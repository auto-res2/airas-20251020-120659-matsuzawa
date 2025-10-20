import math
import os
from pathlib import Path
from typing import Any, Dict, List

import hydra
import optuna
import torch
import torch.nn.functional as F  # noqa: F401 – kept for potential extensions
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.model import build_model, enable_bn_adaptation
from src.preprocess import build_dataloader

########################################################################################################################
# Environment & helpers
########################################################################################################################

os.environ.setdefault("WANDB_CACHE_DIR", ".cache/")
os.environ.setdefault("TORCH_HOME", ".cache/")  # timm & torch hub cache


def set_seed(seed: int) -> None:
    """Deterministic behaviour for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def entropy(p: torch.Tensor) -> torch.Tensor:  # p already softmax
    eps = 1e-8
    return -(p * (p + eps).log()).sum(dim=1)


def accuracy(pred: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
    return (pred.argmax(dim=1) == trg).float().mean()

########################################################################################################################
# Adaptation algorithms
########################################################################################################################

class BaseAdapter:
    """No adaptation – inference only."""

    def __init__(self, model: torch.nn.Module):
        self.model = model.eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).softmax(dim=1)


class TentAdapter(BaseAdapter):
    """Vanilla TENT (Shu et al., 2021)."""

    def __init__(self, model: torch.nn.Module, lr: float, momentum: float, inner_steps: int):
        enable_bn_adaptation(model)
        params = [p for p in model.parameters() if p.requires_grad]
        self.opt = torch.optim.SGD(params, lr=lr, momentum=momentum)
        self.inner_steps = inner_steps
        super().__init__(model)

    def adapt(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        self.model.train()
        output = None
        for _ in range(self.inner_steps):
            self.opt.zero_grad(set_to_none=True)
            output = self.model(x)
            loss = entropy(output.softmax(1)).mean()
            loss.backward()
            self.opt.step()
        self.model.eval()
        assert output is not None
        return output.detach().softmax(1)


class CWTentAdapter(BaseAdapter):
    """Confidence-Weighted TENT (proposed)."""

    def __init__(self, model: torch.nn.Module, lr: float, momentum: float, temperature: float):
        enable_bn_adaptation(model)
        params = [p for p in model.parameters() if p.requires_grad]
        self.opt = torch.optim.SGD(params, lr=lr, momentum=momentum)
        self.temp = temperature
        super().__init__(model)

    def adapt(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = logits.softmax(1)
        H = entropy(probs)
        num_classes = probs.size(1)
        weights = (1.0 - H / math.log(num_classes)).pow(self.temp)
        loss = (weights * H).sum() / weights.sum()
        loss.backward()
        self.opt.step()
        self.model.eval()
        return logits.detach().softmax(1)

########################################################################################################################
# Core routine (single run)
########################################################################################################################

def run_single(cfg: DictConfig) -> Dict[str, Any]:
    """Executes one complete test-time adaptation run."""

    run_cfg = cfg.run  # shorthand

    # ------------------------------------------------------------------
    # Reproducibility & device
    # ------------------------------------------------------------------
    set_seed(int(run_cfg.other.seed))
    device = torch.device(run_cfg.other.device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Model & data
    # ------------------------------------------------------------------
    model = build_model(run_cfg.model)
    model.to(device)

    loader = build_dataloader(run_cfg)

    # Trial-mode: shrink dataset to 2 batches for CI sanity-check
    if cfg.get("trial_mode", False):
        loader = list(loader)[:2]

    # ------------------------------------------------------------------
    # Select adaptation method
    # ------------------------------------------------------------------
    algo = str(run_cfg.training.adaptation_algorithm).lower()
    if algo in {"tent"}:
        adapter = TentAdapter(
            model,
            lr=float(run_cfg.training.learning_rate),
            momentum=float(run_cfg.training.momentum),
            inner_steps=int(run_cfg.training.inner_steps),
        )
    elif algo in {"cw-tent", "cwtent", "cw_tent"}:
        adapter = CWTentAdapter(
            model,
            lr=float(run_cfg.training.learning_rate),
            momentum=float(run_cfg.training.momentum),
            temperature=float(run_cfg.training.weight_temperature),
        )
    else:  # "none"
        adapter = BaseAdapter(model)

    # ------------------------------------------------------------------
    # WandB initialisation (skipped if disabled)
    # ------------------------------------------------------------------
    if cfg.wandb.mode != "disabled":
        run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=run_cfg.run_id,
            resume="allow",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"[WandB] {run.url}")
    else:
        run = None

    # ------------------------------------------------------------------
    # Online loop
    # ------------------------------------------------------------------
    total_correct = 0
    total_samples = 0
    preds_all: List[int] = []
    targets_all: List[int] = []

    iterator = tqdm(enumerate(loader), total=len(loader), desc="Adapting", disable=cfg.get("trial_mode", False))
    for step, (imgs, targets) in iterator:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        probs = adapter.adapt(imgs)
        batch_acc = accuracy(probs, targets).item()

        iterator.set_postfix(acc=f"{batch_acc:.3f}")

        total_correct += (probs.argmax(1) == targets).sum().item()
        total_samples += targets.size(0)

        preds_all.extend(probs.argmax(1).cpu().tolist())
        targets_all.extend(targets.cpu().tolist())

        if run is not None:
            run.log({"batch_acc": batch_acc, "step": step})

    final_acc = total_correct / max(total_samples, 1)

    # ------------------------------------------------------------------
    # Confusion matrix & final logging
    # ------------------------------------------------------------------
    num_classes = int(max(max(preds_all), max(targets_all)) + 1) if preds_all else 1
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(targets_all, preds_all):
        conf_mat[t, p] += 1

    if run is not None:
        run.summary["final_accuracy"] = final_acc
        run.summary["confusion_matrix"] = conf_mat.tolist()
        run.log({"final_accuracy": final_acc})
        run.finish()

    print(f"Final accuracy ({run_cfg.run_id}) = {final_acc:.4f}")
    return {"final_accuracy": final_acc}

########################################################################################################################
# Optuna wrapper (hyper-parameter tuning)
########################################################################################################################

def optuna_objective(trial: optuna.Trial, base_cfg: DictConfig) -> float:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))  # deep copy

    # ------------------------------------------------------------------
    # Inject sampled hyper-parameters
    # ------------------------------------------------------------------
    for hp_name, space in base_cfg.run.optuna.search_space.items():
        s_type = space["type"]
        if s_type == "loguniform":
            sample = trial.suggest_float(hp_name, float(space["low"]), float(space["high"]), log=True)
        elif s_type == "uniform":
            sample = trial.suggest_float(hp_name, float(space["low"]), float(space["high"]))
        elif s_type == "categorical":
            sample = trial.suggest_categorical(hp_name, space["choices"])
        elif s_type == "int":
            sample = trial.suggest_int(hp_name, int(space["low"]), int(space["high"]))
        else:
            raise ValueError(f"Unsupported Optuna space type: {s_type}")
        cfg.run.training[hp_name] = sample

    # Disable WandB during optimisation to avoid run flood
    cfg.wandb.mode = "disabled"

    metrics = run_single(cfg)
    return metrics["final_accuracy"]

########################################################################################################################
# Hydra entry-point
########################################################################################################################

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:  # noqa: D401
    # -------------------------------------------------------------
    # Ensure run group is selected
    # -------------------------------------------------------------
    if "run" not in cfg or cfg.run is None:
        raise ValueError("Specify run=<run_id> on the command line (matches a file in config/run/)")

    # Persist composed config (for evaluate.py)
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, Path(cfg.results_dir) / "config.yaml")

    # -------------------------------------------------------------
    # Hyper-parameter optimisation (Optuna)
    # -------------------------------------------------------------
    if int(cfg.run.optuna.n_trials) > 0 and not cfg.get("trial_mode", False):
        study = optuna.create_study(direction=cfg.run.optuna.direction)
        study.optimize(lambda t: optuna_objective(t, cfg), n_trials=int(cfg.run.optuna.n_trials))
        print("[Optuna] Best parameters:", study.best_params)
        # Update cfg with best parameters and run once more (with WandB enabled)
        for k, v in study.best_params.items():
            cfg.run.training[k] = v
        run_single(cfg)
    else:
        run_single(cfg)


if __name__ == "__main__":
    main()