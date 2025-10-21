from typing import List, Union

import torch
import torchvision.transforms as T
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

# CIFAR-10 statistics -----------------------------------------------------------------------------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

########################################################################################################################
# Synthetic corruption functions (simplified CIFAR-10-C alternatives)
########################################################################################################################

def apply_gaussian_noise(img: Image.Image, severity: int) -> Image.Image:
    """Apply Gaussian noise to image."""
    arr = np.array(img).astype(np.float32)
    std = [0.04, 0.06, 0.08, 0.09, 0.10][severity - 1] * 255
    noise = np.random.normal(0, std, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def apply_gaussian_blur(img: Image.Image, severity: int) -> Image.Image:
    """Apply Gaussian blur to image."""
    radius = [0.5, 0.75, 1.0, 1.25, 1.5][severity - 1]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_brightness(img: Image.Image, severity: int) -> Image.Image:
    """Adjust image brightness."""
    factor = [1.3, 1.5, 1.7, 1.9, 2.1][severity - 1]
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def apply_contrast(img: Image.Image, severity: int) -> Image.Image:
    """Adjust image contrast."""
    factor = [0.5, 0.4, 0.3, 0.2, 0.1][severity - 1]
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def apply_jpeg_compression(img: Image.Image, severity: int) -> Image.Image:
    """Apply JPEG compression artifacts."""
    import io
    quality = [80, 65, 50, 35, 20][severity - 1]
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

# Map of available corruption types
CORRUPTION_FUNCTIONS = {
    'gaussian_noise': apply_gaussian_noise,
    'gaussian_blur': apply_gaussian_blur,
    'brightness': apply_brightness,
    'contrast': apply_contrast,
    'jpeg_compression': apply_jpeg_compression,
}

########################################################################################################################
# Dataset wrapper (CIFAR-10 with synthetic corruptions)
########################################################################################################################


class CIFAR10C(Dataset):
    """CIFAR-10 test set with synthetic corruptions applied on-the-fly."""

    def __init__(
        self,
        split: str,
        severity: int,
        corruption_types: Union[List[str], str] = "all",
        cache_dir: str = ".cache/",
    ) -> None:
        # Load standard CIFAR-10 test set
        ds_all = load_dataset("cifar10", split="test", cache_dir=cache_dir)
        self.ds = ds_all
        self.severity = severity
        
        # Determine which corruptions to apply
        if corruption_types == "all":
            self.corruption_types = list(CORRUPTION_FUNCTIONS.keys())
        else:
            self.corruption_types = [c for c in corruption_types if c in CORRUPTION_FUNCTIONS]
            if not self.corruption_types:
                # Fallback to all if none are valid
                self.corruption_types = list(CORRUPTION_FUNCTIONS.keys())
        
        self.n_classes = 10

    def __len__(self) -> int:  # noqa: D401
        # Replicate dataset by number of corruption types
        return len(self.ds) * len(self.corruption_types)

    def __getitem__(self, idx):
        # Determine which corruption type and base image to use
        corruption_idx = idx % len(self.corruption_types)
        base_idx = idx // len(self.corruption_types)
        
        record = self.ds[base_idx]
        img = record["img"]
        label = int(record["label"])
        
        # Apply the selected corruption
        corruption_type = self.corruption_types[corruption_idx]
        corruption_fn = CORRUPTION_FUNCTIONS[corruption_type]
        corrupted_img = corruption_fn(img, self.severity)
        
        return corrupted_img, label

########################################################################################################################
# Dataloader builder
########################################################################################################################

def build_transform(image_size: int, normalization: str):
    if normalization == "cifar":
        mean, std = CIFAR_MEAN, CIFAR_STD
    else:  # ImageNet statistics (timm default)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    return T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


def build_dataloader(run_cfg):  # expects run_cfg containing dataset.*, training.*, other.*
    tf = build_transform(run_cfg.dataset.image_size, run_cfg.dataset.normalization)

    base_ds = CIFAR10C(
        split=run_cfg.dataset.split,
        severity=run_cfg.dataset.corruption_severity,
        corruption_types=run_cfg.dataset.corruption_types,
        cache_dir=".cache/",
    )

    class _Wrapped(Dataset):
        def __init__(self, inner_ds, transform):
            self.inner_ds = inner_ds
            self.transform = transform

        def __len__(self):
            return len(self.inner_ds)

        def __getitem__(self, i):
            img, label = self.inner_ds[i]
            return self.transform(img), torch.tensor(label, dtype=torch.long)

    ds = _Wrapped(base_ds, tf)
    loader = DataLoader(
        ds,
        batch_size=run_cfg.training.batch_size,
        shuffle=run_cfg.training.shuffle_stream,
        num_workers=run_cfg.other.num_workers,
        pin_memory=True,
    )
    return loader