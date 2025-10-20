# src/train.py
"""Training script executed as a subprocess by main.py.
Implements baseline TENT and the proposed CW-TENT for CIFAR-10-C.
Everything (models, datasets) is cached in .cache/ so that CI stays fast.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# -----------------------------------------------------------------------------
# Force local cache BEFORE heavy imports --------------------------------------
# -----------------------------------------------------------------------------
os.environ.setdefault("TORCH_HOME", ".cache/torch")

import hydra
import optuna
import torch
import torch.nn.functional as F  # noqa: F401  (used implicitly by losses)
import wandb
from omegaconf import OmegaConf

from src.model import (
    ConfidenceWeightedEntropyLoss,
    EntropyLoss,
    build_model,
    freeze_non_bn_parameters,
    initialize_bn_adaptation,
)
from src.preprocess import build_dataloader

###############################################################################
# Config preprocessing --------------------------------------------------------
###############################################################################

def preprocess_sys_argv():
    """Preprocess sys.argv to handle special config name formats."""
    run_config_map = {
        "(11M)-CIFAR-10-C": "proposed-ResNet-18",
    }
    
    for i, arg in enumerate(sys.argv):
        if arg.startswith("run="):
            config_name = arg[4:]
            if config_name in run_config_map:
                sys.argv[i] = f"run={run_config_map[config_name]}"
                break

###############################################################################
# Utility functions -----------------------------------------------------------
###############################################################################

def topk_acc(output: torch.Tensor, target: torch.Tensor, k: int = 1) -> float:
    """Return top-k accuracy (percentage) for a single batch."""
    with torch.no_grad():
        maxk = k
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct.reshape(-1)[: k * batch_size].float().sum(0)
        return float(correct_k.item()) * 100.0 / batch_size

###############################################################################
# Core routine ----------------------------------------------------------------
###############################################################################

def run_single(cfg, *, enable_wandb: bool = True) -> float:
    """Run adaptation with the parameters in ``cfg`` and return final accuracy."""

    run_cfg = getattr(cfg, "run", cfg)  # support both flattened & nested
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------------
    # WandB initialisation -------------------------------------------------
    # ---------------------------------------------------------------------
    if enable_wandb and cfg.wandb.mode != "disabled":
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=run_cfg.run_id,
            resume="allow",
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )
        print(f"[WandB] Run URL: {wandb_run.url}")
    else:
        os.environ["WANDB_MODE"] = "disabled"
        wandb_run = None

    # ---------------------------------------------------------------------
    # Data ----------------------------------------------------------------
    # ---------------------------------------------------------------------
    dataloader = build_dataloader(run_cfg.dataset, split="test", cache_dir=".cache/")

    # ---------------------------------------------------------------------
    # Model & optimiser ---------------------------------------------------
    # ---------------------------------------------------------------------
    model = build_model(run_cfg.model).to(device)
    model.eval()

    freeze_non_bn_parameters(model)
    initialize_bn_adaptation(model)

    params_to_update = filter(lambda p: p.requires_grad, model.parameters())

    opt_name = run_cfg.training.optimizer.lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            params_to_update,
            lr=run_cfg.training.learning_rate,
            momentum=run_cfg.training.momentum,
            weight_decay=run_cfg.training.weight_decay,
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            params_to_update,
            lr=run_cfg.training.learning_rate,
            weight_decay=run_cfg.training.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer {opt_name}")

    if run_cfg.training.loss == "entropy":
        criterion = EntropyLoss()
    elif run_cfg.training.loss == "confidence_weighted_entropy":
        criterion = ConfidenceWeightedEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss {run_cfg.training.loss}")

    # ---------------------------------------------------------------------
    # Adaptation loop (TENT-style) ----------------------------------------
    # ---------------------------------------------------------------------
    global_step = 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    trial_mode: bool = bool(cfg.get("trial_mode", False))

    for epoch in range(run_cfg.training.epochs):
        for batch_idx, (images, targets) in enumerate(dataloader):
            # In trial-mode process only a few batches to keep CI fast
            if trial_mode and batch_idx >= 2:
                break

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # ----- Inner update(s) ---------------------------------------
            model.train()
            for _ in range(run_cfg.training.inner_steps):
                optimizer.zero_grad(set_to_none=True)
                logits_in = model(images)
                loss = criterion(logits_in)
                loss.backward()
                optimizer.step()
            model.eval()

            # ----- Metrics ----------------------------------------------
            with torch.no_grad():
                logits_out = model(images)
            preds = logits_out.argmax(dim=1)
            acc1 = topk_acc(logits_out, targets, k=1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

            if wandb_run is not None:
                wandb.log(
                    {
                        "train_loss": float(loss.item()),
                        "val_acc_batch": float(acc1),
                        "epoch": epoch,
                    },
                    step=global_step,
                )
            global_step += 1

    # ---------------------------------------------------------------------
    # Final accuracy & WandB summary --------------------------------------
    # ---------------------------------------------------------------------
    all_preds_t = torch.tensor(all_preds)
    all_targets_t = torch.tensor(all_targets)
    final_acc = 100.0 * (all_preds_t == all_targets_t).float().mean().item()
    print(f"[RESULT] {run_cfg.run_id} – Final Top-1 Acc: {final_acc:.2f}%")

    if wandb_run is not None:
        wandb_run.summary["top1_accuracy"] = float(final_acc)
        wandb_run.summary["y_true"] = all_targets  # for later evaluation
        wandb_run.summary["y_pred"] = all_preds
        wandb_run.finish()

    return float(final_acc)

###############################################################################
# Optuna helpers --------------------------------------------------------------
###############################################################################

def _suggest_from_space(trial: optuna.Trial, space: Dict[str, Dict[str, Any]]):
    """Sample parameters from an Optuna search-space description."""
    params: Dict[str, Any] = {}
    for name, spec in space.items():
        typ = spec["type"]
        if typ == "loguniform":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif typ == "uniform":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=False)
        elif typ == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown Optuna param type {typ}")
    return params


def _apply_trial_params(cfg, params: Dict[str, Any]):
    """Write Optuna-suggested parameters back into ``cfg`` (in-place)."""
    run_cfg = getattr(cfg, "run", cfg)
    for k, v in params.items():
        if hasattr(run_cfg.training, k):
            setattr(run_cfg.training, k, v)
        elif hasattr(run_cfg.dataset, k):
            setattr(run_cfg.dataset, k, v)
        elif hasattr(run_cfg.model, k):
            setattr(run_cfg.model, k, v)

###############################################################################
# Hydra entry-point -----------------------------------------------------------
###############################################################################

preprocess_sys_argv()

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):  # noqa: C901  (complex but clear)
    # ---------------- Trial-mode tweaks ----------------------------------
    if bool(cfg.get("trial_mode", False)):
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.run.training.epochs = 1

    # Ensure cache directories exist
    Path(".cache/torch").mkdir(parents=True, exist_ok=True)
    Path(".cache/datasets").mkdir(parents=True, exist_ok=True)

    # ---------------- Hyper-parameter optimisation ----------------------
    if cfg.optuna.n_trials > 0:
        print(f"[Optuna] Running {cfg.optuna.n_trials} trials …")

        def objective(trial: optuna.Trial) -> float:
            cfg_trial = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            params = _suggest_from_space(trial, cfg_trial.optuna.search_space)
            _apply_trial_params(cfg_trial, params)
            cfg_trial.wandb.mode = "disabled"  # disable logging during search
            return run_single(cfg_trial, enable_wandb=False)

        study = optuna.create_study(direction=cfg.optuna.direction)
        study.optimize(objective, n_trials=cfg.optuna.n_trials)
        print(
            f"[Optuna] Best value = {study.best_value:.4f}\n[Optuna] Best params = {study.best_params}"
        )
        _apply_trial_params(cfg, study.best_params)

    # ---------------- Final (potentially best) run ----------------------
    run_single(cfg, enable_wandb=(cfg.wandb.mode != "disabled"))


if __name__ == "__main__":
    main()
