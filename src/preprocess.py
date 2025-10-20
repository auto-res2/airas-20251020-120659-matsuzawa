"""src/preprocess.py – dataset loading & preprocessing pipeline."""
from __future__ import annotations

import warnings
from typing import Iterator, List, Tuple

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.datasets import CIFAR10

try:
    from robustbench.data import load_cifar10c

    _ROBUSTBENCH_AVAILABLE = True
    CORRUPTIONS_ALL = [
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
except ImportError:  # pragma: no cover
    _ROBUSTBENCH_AVAILABLE = False
    CORRUPTIONS_ALL: List[str] = []

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
#                         helpers
# ---------------------------------------------------------------------------

def _np_to_tensor(arr):
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float().div(255.0)

# ---------------------------------------------------------------------------
#                    public builder
# ---------------------------------------------------------------------------

def build_dataloader(cfg, device: torch.device) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    name = str(cfg.dataset.name).lower()
    batch_size = int(cfg.dataset.batch_size)
    severity = int(getattr(cfg.dataset, "corruption_severity", 5))

    if "cifar" in name and "c" in name and _ROBUSTBENCH_AVAILABLE:
        corruption_types = cfg.dataset.corruption_types
        if corruption_types in {"all", "*", None}:
            corruption_types = CORRUPTIONS_ALL
        elif isinstance(corruption_types, str):
            corruption_types = [corruption_types]

        xs, ys = [], []
        for c in corruption_types:
            x_np, y_np = load_cifar10c(c, severity, data_dir=".cache/cifar10c")
            xs.append(_np_to_tensor(x_np))
            ys.append(torch.from_numpy(y_np))
        images = torch.cat(xs, dim=0)
        labels = torch.cat(ys).long()
        dataset: Dataset = TensorDataset(images, labels)
    else:
        if not _ROBUSTBENCH_AVAILABLE:
            warnings.warn("RobustBench not found – falling back to clean CIFAR-10 test set.")
        transform = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        dataset = CIFAR10(root=".cache/cifar10", train=False, download=True, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=getattr(cfg.dataset, "shuffle", False),
        num_workers=getattr(cfg.dataset, "num_workers", 4),
        pin_memory=True,
        drop_last=False,
    )

    def _iter():
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if x.max() <= 1.0:  # ensure normalisation for CIFAR-10-C
                x = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)(x)
            yield x, y

    return _iter()
