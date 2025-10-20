# src/main.py
"""Main orchestrator – launches a single experiment run as a subprocess."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf


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
# Hydra entry-point -----------------------------------------------------------
###############################################################################

preprocess_sys_argv()

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg):
    original_cwd = Path(get_original_cwd())

    # Persist full configuration (for evaluate.py)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = results_dir / "config.yaml"
    if not cfg_path.exists():
        OmegaConf.save(config=cfg, f=str(cfg_path))

    task_overrides: List[str] = list(HydraConfig.get().overrides.task)

    # Trial-mode convenience overrides -----------------------------------
    if cfg.get("trial_mode", False):
        if "wandb.mode=disabled" not in task_overrides:
            task_overrides.append("wandb.mode=disabled")
        if "optuna.n_trials=0" not in task_overrides:
            task_overrides.append("optuna.n_trials=0")
        # Make only one epoch during trial-mode
        if "run.training.epochs=1" not in task_overrides:
            task_overrides.append("run.training.epochs=1")

    # Ensure results_dir propagates
    if f"results_dir={cfg.results_dir}" not in task_overrides:
        task_overrides.append(f"results_dir={cfg.results_dir}")

    launch_cmd = [sys.executable, "-u", "-m", "src.train", *task_overrides]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{original_cwd}:{env.get('PYTHONPATH', '')}"

    print("\n[Launcher] Executing:\n  ", " ".join(map(shlex.quote, launch_cmd)))
    subprocess.check_call(launch_cmd, env=env)


if __name__ == "__main__":
    main()
