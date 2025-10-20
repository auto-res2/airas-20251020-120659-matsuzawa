# src/preprocess.py
"""Data loading & preprocessing pipeline for CIFAR-10-C."""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

###############################################################################
# Constants -------------------------------------------------------------------
###############################################################################

CIFAR10C_URL = "https://zenodo.org/record/3555552/files/CIFAR-10-C.tar?download=1"
CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]

###############################################################################
# Download helpers ------------------------------------------------------------
###############################################################################

def _download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dst, "wb") as f, tqdm(total=total, unit="iB", unit_scale=True, desc="CIFAR-10-C") as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def _download_and_extract(url: str, cache_dir: Path) -> Path:
    tar_path = cache_dir / "CIFAR-10-C.tar"
    out_dir = cache_dir / "CIFAR-10-C"
    if not out_dir.exists():
        if not tar_path.exists():
            print("Downloading CIFAR-10-C …")
            _download(url, tar_path)
        print("Extracting CIFAR-10-C …")
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=cache_dir)
    return out_dir

###############################################################################
# Dataset wrapper -------------------------------------------------------------
###############################################################################

class NumpyArrayDataset(Dataset):
    """Simple Dataset around CIFAR-10-C numpy arrays."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        label = int(self.labels[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, label

###############################################################################
# Public API ------------------------------------------------------------------
###############################################################################

def load_cifar10c(
    *,
    severity: int = 5,
    corruption_types: Union[str, List[str]] = "all",
    cache_dir: Union[str, Path] = ".cache/",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return images & labels subset of CIFAR-10-C.

    Parameters
    ----------
    severity : int
        Corruption severity in [1,5].
    corruption_types : str | list[str]
        Either "all" or subset of CORRUPTIONS.
    cache_dir : str | Path
        Directory to store / look up cached data.
    """
    if not 1 <= severity <= 5:
        raise ValueError("Severity must be in [1, 5]")

    cache_dir = Path(cache_dir)
    extract_dir = _download_and_extract(CIFAR10C_URL, cache_dir)

    if corruption_types == "all":
        corruption_types = CORRUPTIONS

    images_list = []
    labels_base = np.load(extract_dir / "labels.npy")
    start, end = (severity - 1) * 10_000, severity * 10_000

    for corr in corruption_types:
        corr_arr = np.load(extract_dir / f"{corr}.npy")
        images_list.append(corr_arr[start:end])
    images = np.concatenate(images_list, axis=0)
    labels = np.tile(labels_base[start:end], len(corruption_types))
    return images, labels


def build_dataloader(
    dataset_cfg,
    *,
    split: str = "test",
    cache_dir: str = ".cache/",
) -> DataLoader:
    """Build a DataLoader according to the Hydra ``dataset_cfg`` object."""
    cache_path = Path(cache_dir)

    if dataset_cfg.name.lower() == "cifar-10-c":
        imgs, lbls = load_cifar10c(
            severity=dataset_cfg.severity,
            corruption_types=dataset_cfg.corruption_types,
            cache_dir=cache_path / "cifar10_c",
        )
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=list(dataset_cfg.normalization.mean),
                    std=list(dataset_cfg.normalization.std),
                ),
            ]
        )
        ds = NumpyArrayDataset(imgs, lbls, transform)
        return DataLoader(
            ds,
            batch_size=dataset_cfg.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )

    raise ValueError(f"Unsupported dataset {dataset_cfg.name}")
