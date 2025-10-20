# src/model.py
"""Model factory for all experiments (currently ResNet-18 only)."""
from __future__ import annotations

import timm
import torch.nn as nn


def create_model(model_cfg, n_classes: int) -> nn.Module:
    name = model_cfg.name.lower()
    ckpt = getattr(model_cfg, "pretrained_checkpoint", None)
    if name == "resnet18":
        model = timm.create_model(
            ckpt if ckpt is not None else "resnet18.a1_in1k",
            pretrained=True,
            num_classes=n_classes,
            cache_dir=".cache/",
        )
    else:
        raise ValueError(f"Unsupported model '{model_cfg.name}'")
    return model


def enable_bn_adaptation_params(model: nn.Module) -> None:
    """Freeze all parameters except BN affine (weight & bias)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                m.weight.requires_grad_(True)
            if m.bias is not None:
                m.bias.requires_grad_(True)