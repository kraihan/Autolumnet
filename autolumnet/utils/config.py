"""Hierarchical config: YAML file + CLI overrides via dotted keys.

Usage:
    config = load_config("configs/sice_sota.yaml", overrides={"train.batch_size": 1024})
    config.train.batch_size  # 1024

CLI override format:
    --override train.batch_size=1024 --override loss.align=0.3
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class DotDict(dict):
    """Read attribute access on dict + safe deep-set via dotted keys."""

    def __getattr__(self, key: str) -> Any:
        if key in self:
            v = self[key]
            return DotDict(v) if isinstance(v, dict) else v
        raise AttributeError(key)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _deep_merge(a: dict, b: dict) -> dict:
    """Recursively merge b into a (b wins). Returns a new dict."""
    out = copy.deepcopy(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# public alias
deep_merge = _deep_merge


def _parse_value(s: str) -> Any:
    """Coerce a CLI string value to int/float/bool/None/string."""
    low = s.lower()
    if low in {"true", "yes"}:  return True
    if low in {"false", "no"}:  return False
    if low in {"null", "none"}: return None
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def _set_nested(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def parse_overrides(items: list[str] | None) -> dict:
    """Convert a list of 'a.b.c=value' strings into a nested dict."""
    overrides: dict = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"Override must be key=value, got: {it!r}")
        k, v = it.split("=", 1)
        _set_nested(overrides, k.strip(), _parse_value(v.strip()))
    return overrides


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> DotDict:
    """Load a YAML config and apply overrides. Returns a DotDict."""
    cfg: dict = {}
    if path is not None:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return DotDict(cfg)


def dump_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False, default_flow_style=False)
