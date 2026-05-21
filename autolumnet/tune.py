"""autolumnet-tune CLI.

Optuna-based hyperparameter search using ASHA pruning (kills bad trials early).
Search space lives in a separate config (--space) for clarity. Each trial runs
the Trainer for at most `max_epochs` (config), and the pruner stops trials
whose intermediate test/ssim is below the running median.

Single-GPU example:
    python -m autolumnet.tune --config configs/sice_sota.yaml \
        --space configs/tune_space.yaml --n-trials 30

Multi-GPU per trial (4 GPUs per trial, trials run sequentially):
    python -m autolumnet.tune --config configs/sice_sota.yaml \
        --space configs/tune_space.yaml --n-trials 30 --n-gpus 4

Parallel trials across multiple GPUs is non-trivial with DDP; this script keeps
it simple: each trial uses ALL `--n-gpus` GPUs, trials run one-at-a-time. For
multi-trial concurrency, run multiple tune processes pointing at the same
Optuna RDB storage URL (--storage sqlite:///tune.db).
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

try:
    import optuna
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    from optuna.samplers import TPESampler
    _HAVE_OPTUNA = True
except ImportError:
    optuna = None
    _HAVE_OPTUNA = False


def _sample_from_space(trial: "optuna.Trial", space: dict) -> dict:
    """Read a YAML 'space' file and call trial.suggest_* accordingly.

    Space format:
        train.lr:           {type: loguniform, low: 1e-5, high: 1e-3}
        train.batch_size:   {type: categorical, choices: [64, 128, 256]}
        model.rho:          {type: uniform, low: 0.05, high: 0.40}
        loss.align:         {type: loguniform, low: 0.01, high: 2.0}
    """
    out = {}
    for key, spec in space.items():
        t = spec["type"]
        if t == "uniform":
            v = trial.suggest_float(key, spec["low"], spec["high"])
        elif t == "loguniform":
            v = trial.suggest_float(key, spec["low"], spec["high"], log=True)
        elif t == "int":
            v = trial.suggest_int(key, spec["low"], spec["high"], log=spec.get("log", False))
        elif t == "categorical":
            v = trial.suggest_categorical(key, spec["choices"])
        else:
            raise ValueError(f"Unknown space type {t!r} for {key}")
        out[key] = v
    return out


def _objective_subprocess(
    trial: "optuna.Trial",
    base_config: str,
    space: dict,
    n_gpus: int,
    out_root: Path,
    monitor: str,
    epochs_per_trial: int,
) -> float:
    """Run one training trial as a subprocess so DDP cleans up between trials.

    Returns the best monitor metric (e.g. SSIM) observed in the trial.
    """
    sampled = _sample_from_space(trial, space)
    trial_dir = out_root / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # build CLI overrides
    overrides = [f"{k}={v}" for k, v in sampled.items()]
    # cap trial length and write to a unique out_dir
    overrides += [
        f"train.epochs={epochs_per_trial}",
        f"train.out_dir={trial_dir}",
        f"train.eval_every={max(1, epochs_per_trial // 8)}",  # 8 reports per trial
    ]

    cmd = []
    if n_gpus > 1:
        cmd += ["torchrun", f"--nproc_per_node={n_gpus}", "-m", "autolumnet.train"]
    else:
        cmd += [sys.executable, "-m", "autolumnet.train"]
    cmd += ["--config", base_config, "--no-wandb"]   # avoid spamming W&B with trials
    for o in overrides:
        cmd += ["--override", o]

    print(f"\n[tune] trial {trial.number}  cmd: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
        raise optuna.TrialPruned(f"trial {trial.number} crashed")

    # parse result.txt
    res = trial_dir / "result.txt"
    if not res.exists():
        raise optuna.TrialPruned(f"trial {trial.number} produced no result.txt")
    line = res.read_text().strip()
    parts = {}
    for kv in line.split(","):
        k, v = kv.split("=")
        parts[k.strip()] = float(v.strip())
    score = parts.get(monitor.split("/")[-1], parts.get("ssim", 0.0))
    print(f"[tune] trial {trial.number} -> {monitor}={score:.4f}  ({line})")
    return score


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("autolumnet-tune")
    p.add_argument("--config", type=str, required=True,
                   help="Base YAML config (the one you'd train with)")
    p.add_argument("--space",  type=str, default="configs/tune_space.yaml",
                   help="YAML file defining the search space")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--n-gpus",   type=int, default=1)
    p.add_argument("--epochs-per-trial", type=int, default=30,
                   help="Per-trial epoch budget (short, since most ideas die fast)")
    p.add_argument("--monitor", type=str, default="ssim",
                   help="Which metric in result.txt to maximise (ssim/psnr) or minimise (l1)")
    p.add_argument("--direction", type=str, default="maximize",
                   choices=["maximize", "minimize"])
    p.add_argument("--out-dir", type=str, default="outputs/tune")
    p.add_argument("--study-name", type=str, default=None)
    p.add_argument("--storage", type=str, default=None,
                   help="Optuna storage URL (sqlite:///tune.db) for cross-process tuning")
    p.add_argument("--pruner", type=str, default="asha", choices=["asha", "median", "none"])
    return p


def main(argv: list[str] | None = None) -> int:
    if not _HAVE_OPTUNA:
        print("Optuna not installed. Run: pip install optuna", file=sys.stderr)
        return 1
    args = build_argparser().parse_args(argv)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(args.space) as f:
        space = yaml.safe_load(f)

    if args.pruner == "asha":
        pruner = SuccessiveHalvingPruner(min_resource=5, reduction_factor=3)
    elif args.pruner == "median":
        pruner = MedianPruner(n_warmup_steps=5)
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=args.study_name or f"autolumnet_{time.strftime('%Y%m%d_%H%M%S')}",
        storage=args.storage,
        load_if_exists=args.storage is not None,
        direction=args.direction,
        sampler=TPESampler(seed=42, multivariate=True),
        pruner=pruner,
    )

    study.optimize(
        lambda t: _objective_subprocess(
            t, args.config, space, args.n_gpus,
            out_root, args.monitor, args.epochs_per_trial),
        n_trials=args.n_trials,
        gc_after_trial=True,
    )

    print("\n[tune] === study complete ===")
    print(f"  best value : {study.best_value:.4f}")
    print(f"  best params: {study.best_params}")

    # dump best config
    best_path = out_root / "best_params.yaml"
    with open(best_path, "w") as f:
        yaml.safe_dump({"best_value": study.best_value,
                        "best_params": study.best_params,
                        "monitor": args.monitor}, f)
    print(f"  written -> {best_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
