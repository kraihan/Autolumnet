# RunPod recipe

Step-by-step for getting AutoLumNet running on a RunPod GPU pod from a VS Code
remote-SSH session.

## 1. Spin up a pod

Pick a template with PyTorch already installed, e.g. **"RunPod PyTorch 2.4"**.
Hardware suggestions (in order of cost-effectiveness for this model):

- **1 × A6000** (48GB): handles `batch_size=128` at `crop=256`, ~2 min/epoch.
- **4 × A6000** (48GB ea.): `batch_size=512` at `crop=256`, ~25 sec/epoch.
- **8 × H100/A100**: `batch_size=1024` at `crop=256`, ~10 sec/epoch.

For SOTA, the 8-GPU pod for ~6 hours (≈300 epochs) is the sweet spot.

## 2. Connect VS Code

In VS Code → Remote-SSH → Connect. Use the pod's SSH connection string from
the RunPod UI. Once connected, open the project folder.

```bash
git clone https://github.com/<you>/autolumnet.git
cd autolumnet
```

(Or rsync from your laptop:  `rsync -avz ./autolumnet/ runpod:/workspace/autolumnet/`.)

## 3. Provision

```bash
export KAGGLE_USERNAME=your_kaggle_username
export KAGGLE_KEY=your_kaggle_key
bash scripts/runpod_provision.sh
```

This installs the project and pre-downloads SICE so the first training job
doesn't pay the download cost.

## 4. Smoke test

```bash
python -m autolumnet.train --config configs/sice_smoke.yaml --name smoke
```

Two epochs, no W&B, ~3 minutes on any GPU. Confirms the full pipeline.

## 5. Real training

Single GPU:
```bash
python -m autolumnet.train --config configs/sice_sota.yaml --name run1
```

Multi-GPU (auto-detect):
```bash
bash scripts/train.sh configs/sice_sota.yaml --name run1
```

Explicit 8 GPUs:
```bash
torchrun --nproc_per_node=8 -m autolumnet.train \
    --config configs/sice_sota.yaml \
    --batch-size 1024 --lr 5e-4 \
    --name big_run
```

## 6. Track in W&B

Set `WANDB_API_KEY` (or run `wandb login`) before launch. The logger will
post metrics to project `autolumnet` by default; change with `--wandb-project`.

## 7. Hyperparameter sweep

```bash
# 30 trials, 4 GPUs per trial, ~30 minutes each
python -m autolumnet.tune \
    --config configs/sice_sota.yaml \
    --space  configs/tune_space.yaml \
    --n-trials 30 --n-gpus 4 --epochs-per-trial 30 \
    --storage sqlite:///tune.db
```

Each trial runs as a `torchrun` subprocess so DDP cleans up between trials.
Pruner is ASHA by default — bad ideas die at epoch 5.

## 8. Final eval

```bash
python -m autolumnet.evaluate \
    --ckpt outputs/big_run/best.pt \
    --data ./data/SICE \
    --use-ema --save-images
```

Writes per-image PNGs and `result.txt` in `outputs/eval/`.

## 9. Inference on your own images

```python
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from autolumnet.models import AutoLumNet

ckpt = torch.load("outputs/big_run/best.pt", map_location="cuda")
net = AutoLumNet(pretrained=False,
                 n_bins=ckpt["cfg"]["model"]["n_bins"],
                 rho=ckpt["cfg"]["model"]["rho"]).cuda().eval()
net.load_state_dict(ckpt["ema"]["ema"])    # use EMA weights

img = to_tensor(Image.open("dark.jpg").convert("RGB")).unsqueeze(0).cuda()
with torch.no_grad():
    fixed = net(img)["rgb"].clamp(0, 1).squeeze(0).cpu()
to_pil_image(fixed).save("fixed.jpg")
```

## Tips

- **OOM?** Halve `train.batch_size`, double `train.accum_steps` to keep the
  effective batch fixed. Or drop `crop` from 256 to 192.
- **GPU idle?** Bump `num_workers` to 8 and `train.batch_size` higher.
- **Slow first epoch?** That's the VGG download + DataLoader workers warming up.
  Subsequent epochs are 3-5x faster.
- **bf16 vs fp16?** On A100/H100/A6000 bf16 is faster *and* more stable. On
  older Volta/Turing (V100/T4) use fp16.
