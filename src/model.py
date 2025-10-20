"""src/model.py – backbone creation & adapters."""
from __future__ import annotations

import math
import os

import timm
import torch
import torch.nn as nn

__all__ = ["create_backbone", "TentAdapter", "CWTentAdapter"]

# ---------------------------------------------------------------------------
#                         backbone
# ---------------------------------------------------------------------------

def create_backbone(cfg) -> nn.Module:  # noqa: ANN001 – cfg is Hydra object
    os.environ.setdefault("TORCH_HOME", ".cache")
    model = timm.create_model(
        cfg.model.name,
        pretrained=bool(cfg.model.pretrained),
        num_classes=int(cfg.model.num_classes),
        cache_dir=".cache/",
    )

    # freeze everything ----------------------------------------------------
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            if m.weight is not None:
                m.weight.requires_grad_(True)
            if m.bias is not None:
                m.bias.requires_grad_(True)
    model.eval()
    return model

# ---------------------------------------------------------------------------
#               utility: entropy
# ---------------------------------------------------------------------------

def entropy_from_probs(p: torch.Tensor) -> torch.Tensor:
    return -(p * (p + 1e-8).log()).sum(1)

# ---------------------------------------------------------------------------
#               base adapter
# ---------------------------------------------------------------------------

class _BaseAdapter(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.num_classes = getattr(backbone, "num_classes", None)
        if self.num_classes is None:
            raise ValueError("Backbone missing 'num_classes' attribute")
        self.backbone.train()  # BN layers use batch stats

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        logits = self.backbone(x)
        return logits.softmax(1)

# ---------------------------------------------------------------------------
#                     original TENT
# ---------------------------------------------------------------------------

class TentAdapter(_BaseAdapter):
    def __init__(
        self,
        backbone: nn.Module,
        lr: float,
        momentum: float = 0.9,
        inner_steps: int = 1,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(backbone)
        params = [p for p in backbone.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        self.inner_steps = int(inner_steps)

    def adapt(self, x: torch.Tensor) -> float:
        self.backbone.train()
        last_loss = 0.0
        for _ in range(self.inner_steps):
            logits = self.backbone(x)
            probs = logits.softmax(1)
            loss = entropy_from_probs(probs).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            last_loss = float(loss.item())
        self.backbone.eval()
        return last_loss

# ---------------------------------------------------------------------------
#                   confidence-weighted TENT
# ---------------------------------------------------------------------------

class CWTentAdapter(TentAdapter):
    def adapt(self, x: torch.Tensor) -> float:  # override
        self.backbone.train()
        logits = self.backbone(x)
        probs = logits.softmax(1)
        entropy = entropy_from_probs(probs)
        weights = 1.0 - entropy / math.log(self.num_classes)
        loss = (weights * entropy).sum() / weights.sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.backbone.eval()
        return float(loss.item())
