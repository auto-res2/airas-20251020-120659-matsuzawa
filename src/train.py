# src/train.py
"""Single-run executor with Hydra, Optuna & WandB integration.

Key fix vs. previous revision
-----------------------------
Hydra composes run–specific YAML files under the ``run`` group, therefore
all *experiment–dependent* keys live under ``cfg.run``.  This revision
*flattens* them back to the root namespace so that the rest of the code
can keep referencing ``cfg.training.*``, ``cfg.dataset.*`` … consistently.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401 – kept for extensibility
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, confusion_matrix

from .model import create_model, enable_bn_adaptation_params
from .preprocess import build_dataloader

# -----------------------------------------------------------------------------
# Optional deps (imported lazily)
# -----------------------------------------------------------------------------
try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None  # type: ignore

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _flatten_run_cfg(cfg: DictConfig) -> None:
    """Copy keys from *cfg.run* to the root level (model, dataset, training …).

    This makes the rest of the code agnostic to where these parameters were
    originally defined.
    """
    if "run" not in cfg:
        return

    OmegaConf.set_struct(cfg, False)  # allow dynamic writes
    for k in ["model", "dataset", "training", "optuna", "seed", "hardware"]:
        if k in cfg.run:
            cfg[k] = cfg.run[k]


def _set_deterministic(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(1).clamp_min(1e-12)
    return -(probs * probs.log()).sum(1)


def _adapt_step(
    model: torch.nn.Module,
    x: torch.Tensor,
    objective: str,
    inner_steps: int,
    optimizer: torch.optim.Optimizer,
) -> None:
    model.train()
    for _ in range(inner_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        H = _entropy_from_logits(logits)
        if objective == "entropy":
            loss = H.mean()
        elif objective == "confidence_weighted_entropy":
            C = logits.size(1)
            with torch.no_grad():
                w = 1.0 - H / np.log(C)
            loss = (w * H).sum() / w.sum()
        else:
            raise ValueError(f"Unknown adaptation objective: {objective}")
        loss.backward()
        optimizer.step()
    model.eval()


def _stream_run(
    cfg: DictConfig,
    hyperparams: Dict[str, float],
    device: torch.device,
    loader: torch.utils.data.DataLoader,
    n_classes: int,
    wb_run=None,
    max_steps_override: int | None = None,
) -> Dict[str, float]:
    """Evaluates the data *stream* once (with optional adaptation)."""
    model = create_model(cfg.model, n_classes)
    enable_bn_adaptation_params(model)
    model.to(device)

    bn_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        bn_params,
        lr=float(hyperparams["learning_rate"]),
        momentum=float(hyperparams["momentum"]),
        weight_decay=float(hyperparams["weight_decay"]),
    )
    inner_steps = int(hyperparams.get("inner_steps", cfg.training.inner_steps))
    objective = cfg.training.objective

    all_preds: List[int] = []
    all_targets: List[int] = []

    max_steps = max_steps_override or len(loader)

    for step, (imgs, targets) in enumerate(loader, 1):
        if step > max_steps:
            break
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Predict BEFORE adaptation
        with torch.no_grad():
            logits = model(imgs)
            preds = logits.argmax(1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

        # Adapt AFTER prediction (when enabled)
        if cfg.training.adaptation and cfg.run.method in {"tent", "cw-tent"}:
            _adapt_step(model, imgs, objective, inner_steps, optimizer)

        # Live logging --------------------------------------------------
        if wb_run is not None and (
            step % int(cfg.training.log_interval) == 0 or step == max_steps
        ):
            acc = accuracy_score(all_targets, all_preds) * 100.0
            wb_run.log({"top1_accuracy": acc, "step": step})

    final_acc = accuracy_score(all_targets, all_preds) * 100.0
    cm = confusion_matrix(all_targets, all_preds)

    if wb_run is not None:
        wb_run.log({"final_top1_accuracy": final_acc, "conf_mat": cm.tolist()})

    return {"final_top1_accuracy": final_acc, "confusion_matrix": cm}


def _suggest_from_space(trial: "optuna.trial.Trial", search_space: Dict) -> Dict[str, float]:
    sampled = {}
    for name, p in search_space.items():
        p_type = p["type"]
        if p_type == "loguniform":
            sampled[name] = trial.suggest_float(name, p["low"], p["high"], log=True)
        elif p_type == "uniform":
            sampled[name] = trial.suggest_float(name, p["low"], p["high"], log=False)
        elif p_type == "categorical":
            sampled[name] = trial.suggest_categorical(name, p["choices"])
        else:
            raise ValueError(f"Unsupported parameter type: {p_type}")
    return sampled


# -----------------------------------------------------------------------------
# HYDRA entry-point
# -----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):  # noqa: C901
    _flatten_run_cfg(cfg)  # align hierarchy (critical fix)

    results_dir = Path(to_absolute_path(cfg.results_dir)).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    _set_deterministic(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Data loader (shared across all trials)
    # ------------------------------------------------------------------
    loader, n_classes = build_dataloader(cfg.dataset, cache_dir=Path(".cache"))

    # ------------------------------------------------------------------
    # WandB setup -------------------------------------------------------
    # ------------------------------------------------------------------
    wb_run = None
    if cfg.wandb.mode != "disabled":
        assert wandb is not None, "wandb is required but not installed"
        wb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            resume="allow",
            mode=cfg.wandb.mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"[WandB] {wb_run.get_url()}")

    # ------------------------------------------------------------------
    # Optuna branch (only if enabled & n_trials > 1 & not trial_mode)
    # ------------------------------------------------------------------
    optuna_cfg = cfg.optuna
    if (
        bool(optuna_cfg.enabled)
        and int(optuna_cfg.n_trials) > 1
        and not cfg.trial_mode
    ):
        assert optuna is not None, "optuna is required but not installed"

        search_space = OmegaConf.to_container(optuna_cfg.search_space, resolve=True)
        direction = optuna_cfg.direction
        n_trials = int(optuna_cfg.n_trials)
        metric_name = optuna_cfg.metric

        print(f"[Optuna] {n_trials} trials | optimising '{metric_name}' ({direction})")

        def _objective(trial):
            params = _suggest_from_space(trial, search_space)
            merged = {
                "learning_rate": cfg.training.learning_rate,
                "momentum": cfg.training.momentum,
                "weight_decay": cfg.training.weight_decay,
                "inner_steps": cfg.training.inner_steps,
            }
            merged.update(params)
            m = _stream_run(
                cfg,
                merged,
                device,
                loader,
                n_classes,
                wb_run=None,  # suppress logging during tuning
            )
            return m[metric_name]

        study = optuna.create_study(direction=direction)
        study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)
        best_params = study.best_params
        print("[Optuna] Best params:", best_params)

        if wb_run is not None:
            wb_run.config.update({"optuna_best_params": best_params}, allow_val_change=True)

        final_params = {
            "learning_rate": cfg.training.learning_rate,
            "momentum": cfg.training.momentum,
            "weight_decay": cfg.training.weight_decay,
            "inner_steps": cfg.training.inner_steps,
        }
        final_params.update(best_params)

        metrics = _stream_run(
            cfg,
            final_params,
            device,
            loader,
            n_classes,
            wb_run=wb_run,
        )
    else:  # ▸ single run (no Optuna)
        hp = {
            "learning_rate": cfg.training.learning_rate,
            "momentum": cfg.training.momentum,
            "weight_decay": cfg.training.weight_decay,
            "inner_steps": cfg.training.inner_steps,
        }
        max_steps = 2 if cfg.trial_mode else None  # quick pass in trial-mode
        metrics = _stream_run(
            cfg,
            hp,
            device,
            loader,
            n_classes,
            wb_run=wb_run,
            max_steps_override=max_steps,
        )

    # ------------------------------------------------------------------
    # Save confusion matrix locally (allowed artefact)
    # ------------------------------------------------------------------
    (results_dir / "confusion_matrix.npy").write_bytes(
        metrics["confusion_matrix"].astype(np.int32).tobytes()
    )

    if wb_run is not None:
        wb_run.finish()


if __name__ == "__main__":
    sys.argv[0] = "train.py"
    main()