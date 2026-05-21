"""Auto-download and lay out the SICE dataset in the EMEF-style folder structure.

After preparation:
    {root}/SICE/
        train/{gt, oe, ue}/   # ~517 scenes
        test /{gt, oe, ue}/   # ~58  scenes
    Each scene id `sid` has:
        gt/{sid:03d}_00.png   ground truth
        oe/{sid}.png          over-exposed input
        ue/{sid}.png          under-exposed input
"""
from __future__ import annotations
import os
import re
import random
import shutil
import subprocess
from pathlib import Path


KAGGLE_SLUG = "khan1803115/sice-dataset-for-autolumnet"
RAW_SUBDIR  = "SICE DATASET for Autolumnet"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_img(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def _extract_sid(stem: str) -> int | None:
    m = re.match(r"\s*([0-9]+)", stem)
    return int(m.group(1)) if m else None


def _link(src: Path, dst: Path) -> None:
    """Symlink src -> dst (relative); fall back to copy if symlinks unavailable."""
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(os.path.relpath(src, dst.parent), dst)
    except Exception:
        shutil.copy2(src, dst)


def _remake(p: Path) -> None:
    if p.exists():
        for f in p.iterdir():
            if f.is_symlink() or f.is_file():
                f.unlink()
    p.mkdir(parents=True, exist_ok=True)


def download_sice(data_root: str | Path) -> Path:
    """Download SICE via Kaggle CLI if not already present. Returns raw root path.

    Requires `KAGGLE_USERNAME` and `KAGGLE_KEY` env vars OR ~/.kaggle/kaggle.json.
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    raw_root = data_root / RAW_SUBDIR

    if raw_root.exists() and any(raw_root.iterdir()):
        return raw_root

    print(f"[data] downloading {KAGGLE_SLUG} -> {data_root}")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_SLUG,
             "-p", str(data_root), "--unzip"],
            check=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "Kaggle CLI not found.  Install with `pip install kaggle` and set "
            "KAGGLE_USERNAME / KAGGLE_KEY env vars."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Kaggle download failed.  Check credentials and dataset slug.  Original error: {e}"
        ) from e

    if not raw_root.exists():
        raise RuntimeError(f"Expected {raw_root} after unzip but it's missing")
    return raw_root


def prepare_sice(
    data_root: str | Path,
    train_frac: float = 0.9,
    seed: int = 42,
    force: bool = False,
) -> Path:
    """Auto-download if needed and rearrange into train/test/{gt,oe,ue} layout.
    Returns the prepared root (e.g. `{data_root}/SICE`)."""
    data_root = Path(data_root)
    out_root  = data_root / "SICE"

    if not force and (out_root / "train" / "gt").exists() and \
       any((out_root / "train" / "gt").iterdir()):
        print(f"[data] SICE already prepared at {out_root}")
        return out_root

    raw_root  = download_sice(data_root)
    label_dir = raw_root / "Label"
    hl_dir    = raw_root / "high_light"
    ll_dir    = raw_root / "low_light"
    for d in (label_dir, hl_dir, ll_dir):
        if not d.is_dir():
            raise RuntimeError(f"Expected directory missing in raw dataset: {d}")

    sid2gt, hl_map, ll_map = {}, {}, {}
    for p in label_dir.iterdir():
        if p.is_file() and _is_img(p):
            sid = _extract_sid(p.stem)
            if sid is not None and sid not in sid2gt:
                sid2gt[sid] = p
    for p in hl_dir.iterdir():
        if p.is_file() and _is_img(p):
            sid = _extract_sid(p.stem)
            if sid is not None:
                hl_map.setdefault(sid, []).append(p)
    for p in ll_dir.iterdir():
        if p.is_file() and _is_img(p):
            sid = _extract_sid(p.stem)
            if sid is not None:
                ll_map.setdefault(sid, []).append(p)

    def pick_one(lst: list[Path]) -> Path | None:
        def keyfun(p: Path) -> tuple:
            m = re.match(r".*_(\d+)$", p.stem)
            return (0 if m else 1, int(m.group(1)) if m else 0, p.name.lower())
        return sorted(lst, key=keyfun)[0] if lst else None

    all_sids = sorted(s for s in sid2gt if s in hl_map and s in ll_map)
    rng = random.Random(seed)
    rng.shuffle(all_sids)
    cut = max(1, int(train_frac * len(all_sids)))
    splits = {"train": all_sids[:cut], "test": all_sids[cut:]}

    for split, sids in splits.items():
        gt_d = out_root / split / "gt"; _remake(gt_d)
        oe_d = out_root / split / "oe"; _remake(oe_d)
        ue_d = out_root / split / "ue"; _remake(ue_d)
        for sid in sids:
            _link(sid2gt[sid], gt_d / f"{sid:03d}_00.png")
            _link(pick_one(hl_map[sid]), oe_d / f"{sid}.png")
            _link(pick_one(ll_map[sid]), ue_d / f"{sid}.png")
        print(f"[data] {split}: {len(sids)} scenes -> {out_root/split}")
    return out_root
