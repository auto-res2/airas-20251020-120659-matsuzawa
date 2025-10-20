# src/main.py
"""Hydra orchestrator – spawns `src.train` as a *sub-process*."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf


def _flatten_run(cfg):
    """Same flattening as in train.py to access root-level keys."""
    from omegaconf import OmegaConf as _OC

    if "run" not in cfg:
        return
    _OC.set_struct(cfg, False)
    for k in ["model", "dataset", "training", "optuna", "seed", "hardware"]:
        if k in cfg.run:
            cfg[k] = cfg.run[k]


def _serialize_cfg(cfg, results_dir: Path):
    OmegaConf.save(cfg, str(results_dir / "config.yaml"))


def _build_overrides(cfg) -> List[str]:
    o: List[str] = [f"run={cfg.run.run_id}", f"results_dir={cfg.results_dir}", f"trial_mode={str(cfg.trial_mode).lower()}"]

    # Quick-run overrides in trial-mode
    if cfg.trial_mode:
        o += [
            "wandb.mode=disabled",
            "optuna.n_trials=0",
            "training.epochs=1",
            "training.inner_steps=1",
            "dataset.batch_size=8",
        ]
    return o


# -----------------------------------------------------------------------------
# HYDRA entry-point
# -----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    _flatten_run(cfg)  # ensure root paths exist for overrides

    results_dir = Path(to_absolute_path(cfg.results_dir)).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    _serialize_cfg(cfg, results_dir)

    cmd = [sys.executable, "-u", "-m", "src.train"] + _build_overrides(cfg)
    print("[Main] Launching:\n  ", " ".join(cmd))

    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    ret = subprocess.call(cmd, env=env)
    if ret != 0:
        raise RuntimeError(f"Training failed (exit {ret})")

    print("[Main] Completed – artefacts in", results_dir)


if __name__ == "__main__":
    main()