# AutoLumNet

Single-shot exposure correction. Same model that worked in your Kaggle notebook,
re-packaged for serious multi-GPU runs on RunPod, with experiment tracking and
hyperparameter tuning.

## What's in the box

- **Multi-GPU training** via `torchrun` DDP, 1–8 GPUs, single command.
- **Mixed precision** (bf16 default on Ampere/Hopper, fp16 fallback).
- **W&B + TensorBoard** dual logging — W&B for dashboards, TB as offline mirror.
- **Optuna** hyperparameter tuner with ASHA pruning (kills bad trials early).
- **Auto-resume**, checkpoint averaging (EMA + last-K SWA), best-model selection.
- **Auto dataset download** from Kaggle (one-time auth via env vars).
- **Argparse + YAML config** with hierarchical override (`config < cli < env`).
- **Deterministic seeding**, gradient accumulation, channels-last memory format.

## Quick start

```bash
# 1. Install
pip install -e ".[dev]"
# (or: pip install -r requirements.txt)

# 2. Kaggle credentials for auto-download (one time)
export KAGGLE_USERNAME=your_name
export KAGGLE_KEY=your_key

# 3. Single-GPU smoke test
python -m autolumnet.train --config configs/sice_smoke.yaml

# 4. Multi-GPU full run (e.g. 4 x A6000 on RunPod)
torchrun --nproc_per_node=4 -m autolumnet.train \
    --config configs/sice_sota.yaml \
    --batch-size 1024 --epochs 300

# 5. Hyperparameter sweep
python -m autolumnet.tune --config configs/sice_sota.yaml --n-trials 30 --n-gpus 4

# 6. Evaluate
python -m autolumnet.evaluate --ckpt outputs/run_xxx/best.pt --data /data/SICE/test
```

## Layout

```
autolumnet/
  models/       network (monotone tone curve, dual branch, residual decoder)
  losses/       sorted-W2 (1-D OT), VGG perceptual, SSIM, combined objective
  data/         SICE dataset, transforms, auto-download from Kaggle
  training/     trainer, DDP setup, optimisers, schedulers, checkpoints
  utils/        config, logging (W&B + TB), seeding, metrics
  train.py      single-/multi-GPU training entrypoint
  tune.py       Optuna hyperparameter search
  evaluate.py   inference & metrics on a held-out set
configs/        YAML files: smoke, sota, tune-space
scripts/        RunPod helpers (provision, sync, launch)
tests/          unit tests for the theory-critical pieces
```

## Theory traceability

This is the *corrected* AutoLumNet, faithful to the rewritten Section 3. Each
guarantee in the paper maps to a line of code:

| Claim | File | What enforces it |
|---|---|---|
| Lemma 1 strict monotonicity | `models/tone.py` | `softplus(w) + eps` then `cumsum` |
| Prop. 1 global order preservation | `models/network.py` | tone weights from `adaptive_avg_pool2d(...)` |
| Lemma 2 OT map = monotone rearrangement | `losses/sorted_w2.py` | `sort(pred)` vs `sort(target)` |
| Lemma 3 no-extrapolation fusion | `models/fusion.py` | softmax-2 weights |
| Cor. 1 bounded residual | `models/decoder.py` | `rho * tanh(...)` head |

See `docs/THEORY.md` for the full derivation.

## RunPod recipe

A 4×A6000 pod runs SICE at `batch=1024` in roughly 25 minutes per epoch.
See `scripts/runpod_provision.sh` for a one-line pod setup.
