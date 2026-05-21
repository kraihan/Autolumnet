"""Atomic checkpoint save/load."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic rename


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(str(path), map_location=map_location)
