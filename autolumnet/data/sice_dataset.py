"""SICE single-shot dataset: one (input, gt) pair per item.

For each scene id `sid` the dataset yields ONE input — either the over- or
under-exposed shot, chosen randomly during training and deterministically
during evaluation (so each scene contributes two test items, one of each
kind, mirroring how the EMEF metric is reported).
"""
from __future__ import annotations
import random
import re
from pathlib import Path

import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


class SICESingleShot(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        crop: int = 256,
        train: bool = True,
        max_train_resize: int | None = None,
    ):
        self.root = Path(root) / split
        self.train = train
        self.crop = crop
        self.max_train_resize = max_train_resize

        gt_files = sorted((self.root / "gt").glob("*.png"))
        self.scenes: list[tuple[Path, Path, Path, int]] = []
        for p in gt_files:
            m = re.match(r"^0*([0-9]+)_\d+\.png$", p.name)
            if not m:
                continue
            sid = int(m.group(1))
            oe = self.root / "oe" / f"{sid}.png"
            ue = self.root / "ue" / f"{sid}.png"
            if oe.exists() and ue.exists():
                self.scenes.append((p, oe, ue, sid))
        if not self.scenes:
            raise RuntimeError(f"No (gt, oe, ue) triplets in {self.root}")

    def __len__(self) -> int:
        # 2 items per scene: train -> 2 random draws, test -> one of each kind
        return len(self.scenes) * 2

    def _resize_min(self, im: Image.Image, size: int) -> Image.Image:
        w, h = im.size
        s = size / min(w, h)
        nw = max(size, int(round(w * s)))
        nh = max(size, int(round(h * s)))
        return im.resize((nw, nh), Image.BICUBIC)

    def __getitem__(self, idx: int):
        scene_idx, kind = divmod(idx, 2)
        gt_p, oe_p, ue_p, sid = self.scenes[scene_idx]
        if self.train:
            kind = random.randint(0, 1)
        inp_p = oe_p if kind == 0 else ue_p

        inp = Image.open(inp_p).convert("RGB")
        gt  = Image.open(gt_p ).convert("RGB")

        inp = self._resize_min(inp, self.crop)
        gt  = self._resize_min(gt,  self.crop)
        w = min(inp.size[0], gt.size[0])
        h = min(inp.size[1], gt.size[1])
        inp = inp.crop((0, 0, w, h))
        gt  = gt .crop((0, 0, w, h))

        if self.train:
            i = random.randint(0, h - self.crop)
            j = random.randint(0, w - self.crop)
            inp = inp.crop((j, i, j + self.crop, i + self.crop))
            gt  = gt .crop((j, i, j + self.crop, i + self.crop))
            if random.random() < 0.5:
                inp = inp.transpose(Image.FLIP_LEFT_RIGHT)
                gt  = gt .transpose(Image.FLIP_LEFT_RIGHT)
        else:
            inp = TF.center_crop(inp, [self.crop, self.crop])
            gt  = TF.center_crop(gt,  [self.crop, self.crop])

        return TF.to_tensor(inp), TF.to_tensor(gt), kind, f"{sid:03d}_00"
