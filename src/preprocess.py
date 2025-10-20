from typing import List

import torch
import torchvision.transforms as T
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

# CIFAR-10 statistics -----------------------------------------------------------------------------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

########################################################################################################################
# Dataset wrapper (CIFAR-10-C)
########################################################################################################################


class CIFAR10C(Dataset):
    """HuggingFace ‘cifar10_corrupted’ wrapper with corruption filtering."""

    def __init__(
        self,
        split: str,
        severity: int,
        corruption_types: List[str] | str = "all",
        cache_dir: str = ".cache/",
    ) -> None:
        self.ds = load_dataset("cifar10_corrupted", split=split, cache_dir=cache_dir)
        self.ds = self.ds.filter(lambda e: e["corruption_severity"] == severity)
        if corruption_types != "all":
            allowed = set(corruption_types)
            self.ds = self.ds.filter(lambda e: e["corruption_type"] in allowed)
        self.ds = self.ds.remove_columns([c for c in self.ds.column_names if c not in {"image", "label"}])
        self.n_classes = 10

    def __len__(self) -> int:  # noqa: D401
        return len(self.ds)

    def __getitem__(self, idx):
        record = self.ds[idx]
        return record["image"], int(record["label"])

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