"""src/train.py – executes a single experiment run (training / adaptation)
--------------------------------------------------------------------------
FULLY IMPLEMENTED.  Logs *all* required information to WandB, including
`y_true`/`y_pred` so that evaluation can always build confusion matrices.
"""
from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import optuna
import torch
import wandb
import yaml
from omegaconf import DictConfig, OmegaConf

from .model import CWTentAdapter, TentAdapter, create_backbone
from .preprocess import build_dataloader

# ---------------------------------------------------------------------------
#                         Re-usable helper utilities
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]


def slugify(value: str) -> str:
    value = value.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def accuracy(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    return (pred.argmax(1) == tgt).float().mean().item()

# ---------------------------------------------------------------------------
#                           Core adaptation routine
# ---------------------------------------------------------------------------

def run_adaptation(cfg: DictConfig) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = build_dataloader(cfg, device)

    backbone = create_backbone(cfg).to(device)
    method_key = str(cfg.method.name).lower()
    if method_key == "tent":
        AdapterCls = TentAdapter
    elif method_key in {"cw-tent", "cwtent"}:
        AdapterCls = CWTentAdapter
    else:
        raise ValueError(f"Unsupported method: {cfg.method.name}")

    adapter = AdapterCls(
        backbone,
        lr=cfg.training.learning_rate,
        momentum=cfg.training.momentum,
        inner_steps=cfg.method.inner_steps,
        weight_decay=cfg.training.weight_decay,
    )

    total_correct, total_seen = 0, 0

    for step, (imgs, lbls) in enumerate(dataloader, 1):
        # prediction BEFORE adaptation (analysis only)
        pred_before = adapter.predict(imgs)
        acc_before = accuracy(pred_before, lbls)

        # ----- adaptation (one or many optimisation steps)
        loss_val = adapter.adapt(imgs)

        # prediction AFTER adaptation
        pred_after = adapter.predict(imgs)
        acc_after = accuracy(pred_after, lbls)

        total_correct += (pred_after.argmax(1) == lbls).sum().item()
        total_seen += lbls.size(0)

        log_payload = {
            "train_loss": loss_val,
            "acc_before": acc_before,
            "acc_after": acc_after,
            "y_true": lbls.cpu().tolist(),
            "y_pred": pred_after.argmax(1).cpu().tolist(),
            "global_step": step,
            "epoch": 1,
        }
        if cfg.wandb.mode != "disabled":
            wandb.log(log_payload)
        print(
            f"[step {step:04d}] loss={loss_val:.4f} acc_b={acc_before:.4f} "
            f"acc_a={acc_after:.4f}"
        )

        if cfg.get("trial_mode", False) and step >= cfg.trial_limit_batches:
            break

    final_acc = total_correct / max(1, total_seen)
    if cfg.wandb.mode != "disabled":
        wandb.log({"final_top1_accuracy": final_acc})
        wandb.run.summary["final_top1_accuracy"] = final_acc
    return {"top1_accuracy": final_acc, "processed_batches": step}

# ---------------------------------------------------------------------------
#                  Optuna hyper-parameter optimisation helper
# ---------------------------------------------------------------------------

def build_objective(cfg_template: DictConfig):
    name2path = {
        "learning_rate": "training.learning_rate",
        "momentum": "training.momentum",
        "weight_decay": "training.weight_decay",
        "batch_size": "dataset.batch_size",
        "inner_steps": "method.inner_steps",
    }

    def objective(trial: optuna.Trial):
        cfg = OmegaConf.create(OmegaConf.to_container(cfg_template, resolve=True))
        for pname, spec in cfg_template.optuna.search_space.items():
            t = str(spec.type)
            if t == "loguniform":
                val = trial.suggest_float(pname, spec.low, spec.high, log=True)
            elif t == "uniform":
                val = trial.suggest_float(pname, spec.low, spec.high)
            elif t == "int":
                val = trial.suggest_int(pname, spec.low, spec.high)
            elif t == "categorical":
                val = trial.suggest_categorical(pname, spec.choices)
            else:
                raise ValueError(f"Unknown search-space type: {t}")
            OmegaConf.update(cfg, name2path.get(pname, pname), val, merge=False)
        metric_val = run_adaptation(cfg)[cfg.optuna.metric]
        return metric_val

    return objective

# ---------------------------------------------------------------------------
#                               Hydra entry-point
# ---------------------------------------------------------------------------
@hydra.main(config_path="../config", config_name="config")
def main(cfg_cli: DictConfig) -> None:
    run_name = str(cfg_cli.run)
    run_cfg_file = Path(__file__).resolve().parents[1] / "config" / "run" / f"{run_name}.yaml"
    if not run_cfg_file.exists():
        raise FileNotFoundError(run_cfg_file)
    cfg_run = OmegaConf.load(run_cfg_file)
    cfg: DictConfig = OmegaConf.merge(cfg_cli, cfg_run)

    # ----------------------------- trial-mode tweaks ----------------------
    if cfg.get("trial_mode", False):
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.trial_limit_batches = 2
    else:
        cfg.trial_limit_batches = int(1e12)

    # ----------------------------- directories ---------------------------
    results_root = Path(cfg.results_dir).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    run_dir = results_root / slugify(cfg_run.run_id)
    run_dir.mkdir(exist_ok=True, parents=True)

    with open(run_dir / "config.yaml", "w") as fh:
        yaml.safe_dump(OmegaConf.to_container(cfg, resolve=True), fh)

    wandb_cfg_file = results_root / "wandb_config.yaml"
    if not wandb_cfg_file.exists():
        with open(wandb_cfg_file, "w") as fh:
            yaml.safe_dump({"wandb": OmegaConf.to_container(cfg.wandb, resolve=True)}, fh)

    # ----------------------------- WandB ---------------------------------
    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=str(cfg_run.run_id),
            resume="allow",
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print("WandB URL:", wandb.run.get_url())
    else:
        os.environ["WANDB_MODE"] = "disabled"

    seed_everything(42)

    # ----------------------------- Optuna --------------------------------
    if int(cfg.optuna.n_trials) > 0:
        objective = build_objective(cfg)
        study = optuna.create_study(direction=cfg.optuna.direction)
        study.optimize(objective, n_trials=int(cfg.optuna.n_trials))
        for k, v in study.best_params.items():
            path_map = {
                "learning_rate": "training.learning_rate",
                "momentum": "training.momentum",
                "weight_decay": "training.weight_decay",
                "batch_size": "dataset.batch_size",
                "inner_steps": "method.inner_steps",
            }
            OmegaConf.update(cfg, path_map.get(k, k), v, merge=False)
        if cfg.wandb.mode != "disabled":
            wandb.log({"optuna_best_value": study.best_value})
            wandb.run.summary["optuna_best_params"] = study.best_params

    # ----------------------------- final run -----------------------------
    stats = run_adaptation(cfg)
    print(f"FINAL top-1 accuracy: {stats['top1_accuracy']:.4f}")

    if cfg.wandb.mode != "disabled":
        wandb.finish()

if __name__ == "__main__":
    main()
