#!/usr/bin/env bash
# Convenience launcher: auto-detects GPU count.
# Usage:  scripts/train.sh configs/sice_sota.yaml --batch-size 1024
set -euo pipefail

CONFIG="${1:-configs/sice_sota.yaml}"
shift || true

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
echo "[launch] detected ${NGPU} GPU(s)  config=${CONFIG}  extra args: $*"

if [ "${NGPU}" -le 1 ]; then
  python -m autolumnet.train --config "${CONFIG}" "$@"
else
  torchrun --nproc_per_node="${NGPU}" -m autolumnet.train --config "${CONFIG}" "$@"
fi
