"""autolumnet-train CLI.

Single-GPU:
    python -m autolumnet.train --config configs/sice_sota.yaml

Multi-GPU (e.g. 4 GPUs on one node):
    torchrun --nproc_per_node=4 -m autolumnet.train --config configs/sice_sota.yaml

Multi-node (2 nodes x 4 GPUs):
    # on master node (rank 0)
    torchrun --nnodes=2 --node_rank=0 --master_addr=$MASTER_IP \
             --master_port=29500 --nproc_per_node=4 \
             -m autolumnet.train --config configs/sice_sota.yaml
    # on worker node (rank 1)
    torchrun --nnodes=2 --node_rank=1 --master_addr=$MASTER_IP \
             --master_port=29500 --nproc_per_node=4 \
             -m autolumnet.train --config configs/sice_sota.yaml

Override any config key from the CLI:
    ... --batch-size 1024 --epochs 300 --override loss.align=0.3
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

from .training import Trainer
from .utils import (Logger, cleanup_distributed, deep_merge, dump_config,
                    init_distributed, load_config, parse_overrides,
                    seed_everything)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("autolumnet-train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default="configs/sice_sota.yaml",
                   help="YAML config file path")
    p.add_argument("--name", type=str, default=None,
                   help="Run name (default: derived from timestamp)")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Override output directory (default: outputs/<name>)")
    p.add_argument("--data-root", type=str, default=None,
                   help="Where SICE will be downloaded/prepared (default: ./data)")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint to resume from (or 'auto' to look in --out-dir)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None,
                   help="GLOBAL effective batch size across all GPUs")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--amp", type=str, default=None, choices=["none", "fp16", "bf16"])
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-tb", action="store_true")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--override", action="append", default=[],
                   help="Dotted-key override, e.g. --override loss.align=0.3 (repeatable)")
    return p


def args_to_overrides(args: argparse.Namespace) -> dict:
    """Promote top-level flags to nested overrides."""
    extras: dict = {}
    if args.epochs       is not None: extras["train.epochs"]      = args.epochs
    if args.batch_size   is not None: extras["train.batch_size"]  = args.batch_size
    if args.lr           is not None: extras["train.lr"]          = args.lr
    if args.seed         is not None: extras["train.seed"]        = args.seed
    if args.amp          is not None: extras["train.amp"]         = args.amp
    if args.num_workers  is not None: extras["train.num_workers"] = args.num_workers
    if args.data_root    is not None: extras["data.root"]         = args.data_root
    if args.out_dir      is not None: extras["train.out_dir"]     = args.out_dir
    if args.resume       is not None: extras["train.resume"]      = args.resume
    if args.wandb_project is not None: extras["logging.wandb_project"] = args.wandb_project
    if args.no_wandb:                 extras["logging.use_wandb"] = False
    if args.no_tb:                    extras["logging.use_tb"]    = False
    return parse_overrides([f"{k}={v}" for k, v in extras.items()])


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    # 1. distributed
    dist_info = init_distributed()

    # 2. config: deep-merge  YAML  <-  CLI flags  <-  --override   (last wins)
    cli_overrides   = args_to_overrides(args)
    extra_overrides = parse_overrides(args.override)
    cfg = load_config(args.config, overrides=cli_overrides)
    cfg = deep_merge(cfg, extra_overrides)

    # name & out_dir
    run_name = args.name or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    if "train" not in cfg: cfg["train"] = {}
    if not cfg["train"].get("out_dir"):                # handles None/missing/""
        cfg["train"]["out_dir"] = f"outputs/{run_name}"
    cfg["train"].setdefault("seed", 42)

    # autoresume
    if cfg["train"].get("resume") == "auto":
        cand = Path(cfg["train"]["out_dir"]) / "last.pt"
        cfg["train"]["resume"] = str(cand) if cand.exists() else None

    seed_everything(cfg["train"]["seed"], rank=dist_info["rank"])

    # save resolved config (rank 0 only)
    if dist_info["is_main"]:
        Path(cfg["train"]["out_dir"]).mkdir(parents=True, exist_ok=True)
        dump_config(cfg, Path(cfg["train"]["out_dir"]) / "config.yaml")
        print(f"[train] run={run_name}  out={cfg['train']['out_dir']}  "
              f"world={dist_info['world_size']}  host={dist_info['hostname']}")

    # logger
    logging_cfg = cfg.get("logging", {})
    logger = Logger(
        out_dir=cfg["train"]["out_dir"],
        run_name=run_name,
        config=cfg,
        use_wandb=logging_cfg.get("use_wandb", True) and not args.no_wandb,
        use_tb=logging_cfg.get("use_tb", True) and not args.no_tb,
        wandb_project=logging_cfg.get("wandb_project", "autolumnet"),
        wandb_entity=logging_cfg.get("wandb_entity"),
        is_main=dist_info["is_main"],
    )

    # train
    try:
        trainer = Trainer(cfg, dist_info, logger=logger)
        final = trainer.fit()
        if dist_info["is_main"]:
            print(f"[train] DONE.  final test: {final}")
    finally:
        logger.finish()
        cleanup_distributed()

    return 0


if __name__ == "__main__":
    sys.exit(main())
