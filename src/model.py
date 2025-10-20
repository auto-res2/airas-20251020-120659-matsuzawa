import os

import timm
import torch

########################################################################################################################
# Model utilities
########################################################################################################################

os.environ.setdefault("TORCH_HOME", ".cache/")  # timm & torch-hub cache


def build_model(model_cfg):
    """Instantiate a timm model according to the provided cfg."""
    pretrained = bool(getattr(model_cfg, "pretrained", True))
    model = timm.create_model(model_cfg.name, pretrained=pretrained, num_classes=10)
    model.eval()
    return model


def enable_bn_adaptation(model: torch.nn.Module) -> None:
    """Enable gradients for BatchNorm affine parameters only (TENT style)."""
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            for p in m.parameters(recurse=False):
                p.requires_grad_(True)
        else:
            for p in m.parameters(recurse=False):
                p.requires_grad_(False)