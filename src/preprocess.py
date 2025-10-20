# src/preprocess.py
"""Data pipeline for CIFAR-10-C (severity configurable).

All datasets are expected inside ``.cache/datasets/`` to fully comply with the
specification.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


class CIFAR10CDataset(Dataset):
    _STANDARD_TYPES: List[str] = [
        "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
        "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
        "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
    ]

    def __init__(
        self,
        root: Path,
        severity: int,
        corruption_types,
        transform=None,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.severity = severity

        if corruption_types == "all" or (
            isinstance(corruption_types, list) and "all" in corruption_types
        ):
            self.types = self._STANDARD_TYPES
        else:
            self.types = list(corruption_types)

        labels_path = root / "labels.npy"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"CIFAR-10-C not found under {root}.  Download it from \n"
                "https://zenodo.org/record/2535967 and place it inside .cache/datasets/"
            )
        self.labels = np.load(labels_path)

        self._images: List[np.ndarray] = []
        self._cum_counts: List[int] = []
        start, end = (severity - 1) * 10000, severity * 10000
        for c in self.types:
            arr = np.load(root / f"{c}.npy", mmap_mode="r")[start:end]
            self._images.append(arr)
            self._cum_counts.append(
                arr.shape[0] if not self._cum_counts else self._cum_counts[-1] + arr.shape[0]
            )
        self.length = self._cum_counts[-1]

    # ------------------------------------------------------------------
    def __len__(self):
        return self.length

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        bucket = int(np.searchsorted(self._cum_counts, idx, side="right"))
        prev = 0 if bucket == 0 else self._cum_counts[bucket - 1]
        local_idx = idx - prev
        img_np = self._images[bucket][local_idx]
        label = int(self.labels[local_idx])
        img = T.ToPILImage()(img_np)
        if self.transform:
            img = self.transform(img)
        return img, label


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def build_dataloader(dataset_cfg, cache_dir: Path) -> Tuple[DataLoader, int]:
    root = cache_dir / "datasets" / "CIFAR-10-C"

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=dataset_cfg.preprocessing.normalize.mean,
                std=dataset_cfg.preprocessing.normalize.std,
            ),
        ]
    )

    ds = CIFAR10CDataset(
        root=root,
        severity=int(dataset_cfg.corruption_severity),
        corruption_types=dataset_cfg.corruption_types,
        transform=transform,
    )

    loader = DataLoader(
        ds,
        batch_size=int(dataset_cfg.batch_size),
        shuffle=False,
        num_workers=int(dataset_cfg.num_workers),
        pin_memory=True,
    )
    return loader, 10  # CIFAR-10 has 10 classes