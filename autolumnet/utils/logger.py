"""Dual experiment logger: W&B (primary) + TensorBoard (mirror).

Logs metrics, hyperparameters, sample images, and the final result.txt line.
Both backends are optional: if wandb is not installed or `wandb: false` in
config, we fall back to TensorBoard only. If neither is wanted, set both off.
"""
from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Any

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAVE_TB = True
except Exception:
    _HAVE_TB = False

try:
    import wandb
    _HAVE_WANDB = True
except Exception:
    _HAVE_WANDB = False


class Logger:
    """No-op on non-main ranks; logs on rank 0 only."""

    def __init__(
        self,
        out_dir: str | Path,
        run_name: str,
        config: dict,
        use_wandb: bool = True,
        use_tb: bool = True,
        wandb_project: str = "autolumnet",
        wandb_entity: str | None = None,
        is_main: bool = True,
    ):
        self.out_dir = Path(out_dir)
        self.run_name = run_name
        self.is_main = is_main

        self._wandb = None
        self._tb    = None

        if not is_main:
            return

        self.out_dir.mkdir(parents=True, exist_ok=True)

        if use_wandb and _HAVE_WANDB:
            try:
                self._wandb = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=run_name,
                    config=dict(config),
                    dir=str(self.out_dir),
                    settings=wandb.Settings(start_method="thread"),
                )
            except Exception as e:
                print(f"[logger] W&B init failed ({e}); continuing without W&B.")
                self._wandb = None

        if use_tb and _HAVE_TB:
            self._tb = SummaryWriter(log_dir=str(self.out_dir / "tb"))

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.is_main:
            return
        if self._wandb is not None:
            try:
                self._wandb.log(metrics, step=step)
            except Exception:
                pass
        if self._tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(k, v, step or 0)

    def log_image(self, tag: str, image: torch.Tensor, step: int | None = None) -> None:
        """image: (3, H, W) in [0,1]."""
        if not self.is_main:
            return
        if self._wandb is not None:
            try:
                import numpy as np
                arr = (image.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                self._wandb.log({tag: wandb.Image(arr)}, step=step)
            except Exception:
                pass
        if self._tb is not None:
            self._tb.add_image(tag, image.detach().cpu().clamp(0, 1), step or 0)

    def log_text(self, tag: str, text: str, step: int | None = None) -> None:
        if not self.is_main:
            return
        if self._wandb is not None:
            try:
                self._wandb.log({tag: wandb.Html(f"<pre>{text}</pre>")}, step=step)
            except Exception:
                pass
        if self._tb is not None:
            self._tb.add_text(tag, text, step or 0)
        with open(self.out_dir / f"{tag}.txt", "a") as f:
            f.write(text + "\n")

    def finish(self) -> None:
        if not self.is_main:
            return
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
