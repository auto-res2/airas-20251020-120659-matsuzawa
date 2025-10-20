# src/model.py
"""Model definitions & utilities (timm ResNet-18 + helpers for TENT/CW-TENT)."""

from __future__ import annotations

import os
from pathlib import Path

# Ensure timm model weights are stored locally
os.environ.setdefault("TORCH_HOME", ".cache/torch")
Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)

import torch.nn as nn
import timm

###############################################################################
# Model factory ----------------------------------------------------------------
###############################################################################

def build_model(model_cfg):
    """Create a timm model with the specified configuration."""
    model_name = getattr(model_cfg, "checkpoint", model_cfg.name)
    model = timm.create_model(model_name, pretrained=model_cfg.pretrained, num_classes=10)
    return model

###############################################################################
# TENT/CW-TENT helpers ---------------------------------------------------------
###############################################################################

def freeze_non_bn_parameters(model: nn.Module):
    """Freeze all parameters except BatchNorm affine weights & biases."""
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.weight is not None:
                m.weight.requires_grad_(True)
            if m.bias is not None:
                m.bias.requires_grad_(True)


def initialize_bn_adaptation(model: nn.Module):
    """Set BN layers to train mode & disable running-stats (TENT style)."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()
            m.track_running_stats = False

###############################################################################
# Losses -----------------------------------------------------------------------
###############################################################################

import torch
import torch.nn.functional as F

class EntropyLoss(nn.Module):
    """Standard prediction entropy loss used by TENT."""

    def forward(self, logits: torch.Tensor) -> torch.Tensor:  # noqa: D401
        p = F.softmax(logits, dim=1)
        ent = -(p * torch.clamp(p, min=1e-12).log()).sum(dim=1)
        return ent.mean()


class ConfidenceWeightedEntropyLoss(nn.Module):
    """Confidence-Weighted entropy loss (proposed)."""

    def forward(self, logits: torch.Tensor) -> torch.Tensor:  # noqa: D401
        p = F.softmax(logits, dim=1)
        ent = -(p * torch.clamp(p, min=1e-12).log()).sum(dim=1)
        num_classes = logits.size(1)
        weights = 1.0 - ent / torch.log(torch.tensor(num_classes, device=logits.device, dtype=ent.dtype))
        return (weights * ent).sum() / torch.clamp(weights.sum(), min=1e-12)
