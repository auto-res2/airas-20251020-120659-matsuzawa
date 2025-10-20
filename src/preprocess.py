# src/preprocess.py
"""Data loading & preprocessing pipeline for CIFAR-10-C."""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import requests
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

###############################################################################
# Constants -------------------------------------------------------------------
###############################################################################

CIFAR10C_URL = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
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
# Corruption helpers ----------------------------------------------------------
###############################################################################

def apply_gaussian_noise(img: Image.Image, severity: int) -> Image.Image:
    arr = np.array(img).astype(np.float32) / 255.
    std = [0.04, 0.06, 0.08, 0.09, 0.10][severity - 1]
    noise = np.random.normal(0, std, arr.shape)
    arr = np.clip(arr + noise, 0, 1) * 255
    return Image.fromarray(arr.astype(np.uint8))

def apply_brightness(img: Image.Image, severity: int) -> Image.Image:
    factor = [1.1, 1.2, 1.3, 1.4, 1.5][severity - 1]
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def apply_contrast(img: Image.Image, severity: int) -> Image.Image:
    factor = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def apply_defocus_blur(img: Image.Image, severity: int) -> Image.Image:
    radius = [1, 1.5, 2, 2.5, 3][severity - 1]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_motion_blur(img: Image.Image, severity: int) -> Image.Image:
    radius = [5, 7, 9, 11, 13][severity - 1]
    return img.filter(ImageFilter.BoxBlur(radius=radius//2))

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
    """Return images & labels subset of CIFAR-10-C using torchvision CIFAR-10."""
    if not 1 <= severity <= 5:
        raise ValueError("Severity must be in [1, 5]")

    cache_dir = Path(cache_dir)
    
    cifar10 = datasets.CIFAR10(root=cache_dir, train=False, download=True)
    
    if corruption_types == "all":
        corruption_types = ["gaussian_noise", "brightness", "contrast", 
                           "defocus_blur", "motion_blur"]
    
    corruption_map = {
        "gaussian_noise": apply_gaussian_noise,
        "brightness": apply_brightness,
        "contrast": apply_contrast,
        "defocus_blur": apply_defocus_blur,
        "motion_blur": apply_motion_blur,
    }
    
    images_list = []
    labels_list = []
    
    print(f"Generating corrupted CIFAR-10 data (severity={severity})...")
    for corruption in corruption_types:
        if corruption not in corruption_map:
            continue
        corrupt_fn = corruption_map[corruption]
        corrupted_images = []
        corrupted_labels = []
        for idx in tqdm(range(len(cifar10)), desc=f"Applying {corruption}"):
            img, label = cifar10[idx]
            corrupted_img = corrupt_fn(img, severity)
            corrupted_images.append(np.array(corrupted_img))
            corrupted_labels.append(label)
        images_list.append(np.array(corrupted_images))
        labels_list.append(np.array(corrupted_labels))
    
    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
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
