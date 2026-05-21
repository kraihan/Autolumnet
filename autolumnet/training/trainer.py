"""End-to-end Trainer: DDP, AMP, EMA, gradient accumulation, checkpointing.

Single class so the training loop and tuning loop share state.
"""
from __future__ import annotations
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..data import SICESingleShot, prepare_sice
from ..losses import LossWeights, VGGPerceptual, compute_loss
from ..models import AutoLumNet
from ..utils import (Logger, ModelEMA, all_reduce_mean, l1_metric,
                     load_checkpoint, main_process_first, psnr_metric,
                     save_checkpoint, ssim_metric)


def _amp_dtype(name: str) -> torch.dtype | None:
    name = (name or "none").lower()
    if name == "bf16": return torch.bfloat16
    if name == "fp16": return torch.float16
    return None


class Trainer:
    def __init__(self, cfg: dict, dist_info: dict, logger: Logger | None = None):
        self.cfg = cfg
        self.dist = dist_info
        self.logger = logger
        self.device = dist_info["device"]
        self.is_main = dist_info["is_main"]

        # ---------- data ----------
        with main_process_first(self.is_main):
            self.data_root = prepare_sice(
                data_root=cfg["data"]["root"],
                train_frac=cfg["data"].get("train_frac", 0.9),
                seed=cfg["data"].get("seed", 42),
                force=cfg["data"].get("force_reprepare", False),
            )

        crop = cfg["data"].get("crop", 256)
        train_ds = SICESingleShot(self.data_root, split="train", crop=crop, train=True)
        test_ds  = SICESingleShot(self.data_root, split="test",  crop=crop, train=False)

        bs       = cfg["train"]["batch_size"]
        per_gpu  = max(1, bs // dist_info["world_size"])
        # effective batch via gradient accumulation if per-GPU memory is the limit
        self.accum_steps = max(1, int(cfg["train"].get("accum_steps", 1)))
        self.micro_bs    = max(1, per_gpu // self.accum_steps)

        sampler = (DistributedSampler(train_ds, shuffle=True, drop_last=True)
                   if dist_info["is_dist"] else None)
        self.train_sampler = sampler
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.micro_bs,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=cfg["train"].get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
            persistent_workers=cfg["train"].get("num_workers", 4) > 0,
        )
        self.test_loader = DataLoader(
            test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
        )

        if self.is_main:
            print(f"[trainer] train scenes={len(train_ds.scenes)}  test scenes={len(test_ds.scenes)}")
            print(f"[trainer] global batch={bs}  per-gpu={per_gpu}  "
                  f"micro_bs={self.micro_bs}  accum_steps={self.accum_steps}  "
                  f"world_size={dist_info['world_size']}")

        # ---------- model ----------
        net = AutoLumNet(
            pretrained=cfg["model"].get("pretrained", True),
            n_bins=cfg["model"].get("n_bins", 64),
            rho=cfg["model"].get("rho", 0.20),
        ).to(self.device)
        self.net_uncompiled = net  # keep a non-DDP, non-compile handle for EMA construction

        if cfg["train"].get("channels_last", True) and torch.cuda.is_available():
            net = net.to(memory_format=torch.channels_last)

        if cfg["train"].get("compile", False) and hasattr(torch, "compile"):
            net = torch.compile(net, mode="default")

        if dist_info["is_dist"]:
            net = DDP(net, device_ids=[dist_info["local_rank"]],
                      find_unused_parameters=False, gradient_as_bucket_view=True)

        self.net = net
        # ---------- loss bundle ----------
        lw = cfg.get("loss", {})
        self.loss_w = LossWeights(
            rec=lw.get("rec", 1.0),
            align=lw.get("align", 0.5),
            perc=lw.get("perc", 0.1),
            smooth=lw.get("smooth", 0.1),
            ssim=lw.get("ssim", 0.2),
        )
        # perceptual loss net
        perc_pretrained = lw.get("perc_pretrained", True)
        self.perc = VGGPerceptual(pretrained=perc_pretrained).to(self.device).eval()
        for p in self.perc.parameters():
            p.requires_grad_(False)

        # canonical target P*
        sigma = lw.get("p_star_sigma", 0.18)
        mean  = lw.get("p_star_mean",  0.50)
        self.p_star = (torch.randn(200_000) * sigma + mean).clamp(0.02, 0.98).to(self.device)

        # ---------- optimisation ----------
        self.opt = torch.optim.AdamW(
            [p for p in self.net.parameters() if p.requires_grad],
            lr=cfg["train"]["lr"],
            betas=tuple(cfg["train"].get("betas", (0.9, 0.999))),
            weight_decay=cfg["train"].get("weight_decay", 1e-2),
        )

        epochs = cfg["train"]["epochs"]
        self.epochs = epochs
        total_optim_steps = epochs * max(1, len(self.train_loader) // self.accum_steps)
        warmup = cfg["train"].get("warmup_steps", min(500, max(1, total_optim_steps // 50)))
        self._warmup = warmup
        self._sched_total = total_optim_steps
        self.sched = CosineAnnealingLR(self.opt, T_max=max(1, total_optim_steps - warmup))
        self.base_lr = cfg["train"]["lr"]

        # ---------- AMP ----------
        self.amp_dtype = _amp_dtype(cfg["train"].get("amp", "bf16"))
        self.scaler = GradScaler(device=self.device.type,
                                 enabled=(self.amp_dtype == torch.float16))

        # ---------- EMA ----------
        self.use_ema = cfg["train"].get("ema", True)
        self.ema = ModelEMA(self.net_uncompiled, decay=cfg["train"].get("ema_decay", 0.999)) \
                   if self.use_ema else None

        # ---------- output ----------
        self.out_dir = Path(cfg["train"]["out_dir"])
        if self.is_main:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.best_score = -math.inf
        self.start_epoch = 1
        self.global_step = 0

    # ------------------------------------------------------------------ resume
    def maybe_resume(self) -> None:
        resume_path = self.cfg["train"].get("resume")
        if not resume_path:
            return
        p = Path(resume_path)
        if not p.exists():
            if self.is_main:
                print(f"[trainer] resume path {p} not found, starting fresh")
            return
        ckpt = load_checkpoint(p, map_location=str(self.device))
        target = self.net.module if hasattr(self.net, "module") else self.net
        target.load_state_dict(ckpt["net"], strict=False)
        if "opt" in ckpt:    self.opt.load_state_dict(ckpt["opt"])
        if "sched" in ckpt:  self.sched.load_state_dict(ckpt["sched"])
        if "scaler" in ckpt: self.scaler.load_state_dict(ckpt["scaler"])
        if self.ema is not None and "ema" in ckpt:
            self.ema.load_state_dict(ckpt["ema"])
        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_score  = float(ckpt.get("best_score", -math.inf))
        if self.is_main:
            print(f"[trainer] resumed from epoch {self.start_epoch-1}, step {self.global_step}")

    # ----------------------------------------------------------- single epoch
    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.net.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        t0 = time.time()
        running = {"loss": 0.0, "rec": 0.0, "align": 0.0, "perc": 0.0, "smooth": 0.0}
        n_iter = 0
        self.opt.zero_grad(set_to_none=True)

        for it, (inp, gt, _kind, _name) in enumerate(self.train_loader):
            inp = inp.to(self.device, non_blocking=True)
            gt  = gt .to(self.device, non_blocking=True)
            if self.cfg["train"].get("channels_last", True):
                inp = inp.contiguous(memory_format=torch.channels_last)
                gt  = gt .contiguous(memory_format=torch.channels_last)

            with torch.autocast(device_type=self.device.type,
                                dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                out = self.net(inp)
                loss, parts = compute_loss(out, gt, self.perc, self.p_star, self.loss_w)
                loss = loss / self.accum_steps

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            do_step = (it + 1) % self.accum_steps == 0
            if do_step:
                # LR warmup
                if self.global_step < self._warmup:
                    lr = self.base_lr * (self.global_step + 1) / self._warmup
                    for g in self.opt.param_groups:
                        g["lr"] = lr
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                                   self.cfg["train"].get("grad_clip", 5.0))
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                                   self.cfg["train"].get("grad_clip", 5.0))
                    self.opt.step()
                self.opt.zero_grad(set_to_none=True)
                if self.global_step >= self._warmup:
                    self.sched.step()
                self.global_step += 1
                if self.ema is not None:
                    self.ema.update(self.net)

            running["loss"]   += float(loss.detach()) * self.accum_steps
            for k, v in parts.items():
                running[k]    += v
            n_iter += 1

            if self.is_main and it % max(1, len(self.train_loader) // 10) == 0:
                lr_now = self.opt.param_groups[0]["lr"]
                if self.logger:
                    self.logger.log(
                        {f"train/{k}": running[k] / max(1, n_iter) for k in running}
                        | {"train/lr": lr_now,
                           "train/epoch": epoch,
                           "train/step": self.global_step},
                        step=self.global_step,
                    )

        out = {k: v / max(1, n_iter) for k, v in running.items()}
        out["time"] = time.time() - t0
        return out

    # ----------------------------------------------------- evaluation routine
    @torch.no_grad()
    def evaluate(self, use_ema: bool = True) -> dict[str, float]:
        model = self.ema.ema if (use_ema and self.ema is not None) else self.net
        model.eval()

        ssim_sum = psnr_sum = l1_sum = 0.0
        n = 0
        for inp, gt, _kind, _name in self.test_loader:
            inp = inp.to(self.device, non_blocking=True)
            gt  = gt .to(self.device, non_blocking=True)
            pred = model(inp)["rgb"].clamp(0.0, 1.0)
            ssim_sum += ssim_metric(pred, gt)
            psnr_sum += psnr_metric(pred, gt)
            l1_sum   += l1_metric  (pred, gt)
            n += 1

        # reduce across ranks (rank-0 might not have all batches if test_loader
        # were sharded, but it isn't -- each rank sees the full test set)
        return {
            "test/ssim": ssim_sum / n,
            "test/psnr": psnr_sum / n,
            "test/l1":   l1_sum   / n,
        }

    # --------------------------------------------------- top-level train loop
    def fit(self) -> dict[str, float]:
        self.maybe_resume()
        last_test = {}
        for ep in range(self.start_epoch, self.epochs + 1):
            train_stats = self.train_one_epoch(ep)

            # eval interval
            if ep % self.cfg["train"].get("eval_every", 5) == 0 or ep == self.epochs:
                test_stats = self.evaluate(use_ema=self.use_ema)
                last_test = test_stats
                if self.is_main:
                    msg = (f"epoch {ep:4d}/{self.epochs}  "
                           f"loss={train_stats['loss']:.4f}  "
                           f"ssim={test_stats['test/ssim']:.4f}  "
                           f"psnr={test_stats['test/psnr']:.3f}  "
                           f"l1={test_stats['test/l1']:.4f}  "
                           f"({train_stats['time']:.1f}s)")
                    print(msg)
                    if self.logger:
                        self.logger.log({**test_stats,
                                         **{f"train_epoch/{k}": v for k, v in train_stats.items()}},
                                        step=self.global_step)

                    # selection score: SSIM (configurable)
                    score = test_stats[self.cfg["train"].get("monitor", "test/ssim")]
                    if score > self.best_score:
                        self.best_score = score
                        self._save("best.pt", ep, score)

            # periodic save
            if self.is_main and ep % self.cfg["train"].get("save_every", 20) == 0:
                self._save(f"epoch_{ep:04d}.pt", ep, self.best_score)

        if self.is_main:
            self._save("last.pt", self.epochs, self.best_score)
            # write the EMEF-style result line
            line = (f"ssim = {last_test.get('test/ssim', 0):.6f}, "
                    f"psnr = {last_test.get('test/psnr', 0):.6f}, "
                    f"l1 = {last_test.get('test/l1', 0):.6f}")
            (self.out_dir / "result.txt").write_text(line + "\n")
            if self.logger:
                self.logger.log_text("result", line)
            print(line)
        return last_test

    # ---------------------------------------------------------------- helpers
    def _save(self, fname: str, epoch: int, score: float) -> None:
        if not self.is_main:
            return
        target = self.net.module if hasattr(self.net, "module") else self.net
        state = {
            "net":     target.state_dict(),
            "opt":     self.opt.state_dict(),
            "sched":   self.sched.state_dict(),
            "scaler":  self.scaler.state_dict(),
            "epoch":   epoch,
            "global_step": self.global_step,
            "best_score":  self.best_score,
            "cfg":     dict(self.cfg),
        }
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()
        save_checkpoint(self.out_dir / fname, state)
