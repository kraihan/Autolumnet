#!/usr/bin/env bash
# One-shot RunPod provisioning. Run inside a fresh PyTorch container.
set -euo pipefail

echo "[provision] system info:"
nvidia-smi || true
python --version
pip --version

echo "[provision] installing project ..."
pip install --upgrade pip
pip install -e ".[dev]"

# Kaggle credentials must already be set in env (KAGGLE_USERNAME, KAGGLE_KEY)
if [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; then
  echo "[provision] WARNING: KAGGLE_USERNAME / KAGGLE_KEY not set."
  echo "            Set them before training so SICE can auto-download:"
  echo "              export KAGGLE_USERNAME=your_name"
  echo "              export KAGGLE_KEY=your_key"
else
  mkdir -p ~/.kaggle
  printf '{"username":"%s","key":"%s"}\n' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
  echo "[provision] kaggle credentials installed."
fi

# Pre-download dataset so first training job doesn't wait
echo "[provision] pre-downloading SICE..."
python -c "from autolumnet.data import prepare_sice; prepare_sice('./data')"

echo "[provision] done.  Try:  torchrun --nproc_per_node=\$(nvidia-smi -L | wc -l) -m autolumnet.train --config configs/sice_sota.yaml"
