"""autolumnet-eval CLI.

Run a trained checkpoint on a SICE-style test split and produce metrics.txt.
"""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .data import SICESingleShot
from .models import AutoLumNet
from .utils import l1_metric, load_checkpoint, psnr_metric, ssim_metric


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("autolumnet-eval")
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt")
    p.add_argument("--data", type=str, required=True, help="Path to SICE/<split> root")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--out",  type=str, default="outputs/eval")
    p.add_argument("--save-images", action="store_true",
                   help="Write per-image input/pred/gt PNGs (EMEF-style naming).")
    p.add_argument("--use-ema", action="store_true",
                   help="Use EMA weights if present in the checkpoint.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    out_dir = Path(args.out); (out_dir / "images").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(args.ckpt, map_location=str(device))
    cfg_model = ckpt.get("cfg", {}).get("model", {})
    net = AutoLumNet(
        pretrained=False,
        n_bins=cfg_model.get("n_bins", 64),
        rho=cfg_model.get("rho", 0.20),
    ).to(device)

    if args.use_ema and "ema" in ckpt:
        net.load_state_dict(ckpt["ema"]["ema"])
        print("[eval] loaded EMA weights")
    else:
        net.load_state_dict(ckpt["net"])
        print("[eval] loaded raw weights")
    net.eval()

    ds = SICESingleShot(Path(args.data).parent, split=args.split, crop=args.crop, train=False) \
         if (Path(args.data).parent / args.split / "gt").exists() else \
         SICESingleShot(args.data, split=args.split, crop=args.crop, train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    ssim_sum = psnr_sum = l1_sum = 0.0
    n = 0
    with torch.no_grad():
        for i, (inp, gt, kind, name) in enumerate(loader):
            inp = inp.to(device); gt = gt.to(device)
            pred = net(inp)["rgb"].clamp(0, 1)
            ssim_sum += ssim_metric(pred, gt)
            psnr_sum += psnr_metric(pred, gt)
            l1_sum   += l1_metric(pred, gt)
            n += 1
            if args.save_images:
                tag = "oe" if int(kind.item()) == 0 else "ue"
                save_image(pred.cpu(), out_dir / "images" / f"{name[0]}_{tag}_fake_B.png")
                save_image(inp.cpu(),  out_dir / "images" / f"{name[0]}_{tag}_real_A.png")
                save_image(gt.cpu(),   out_dir / "images" / f"{name[0]}_{tag}_real_B.png")
            if i < 5 or i % 25 == 0:
                print(f"  processed ({i:04d})  {name[0]}")

    line = (f"ssim = {ssim_sum/n:.6f}, "
            f"psnr = {psnr_sum/n:.6f}, "
            f"l1 = {l1_sum/n:.6f}")
    print(line)
    (out_dir / "result.txt").write_text(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
