"""src/main.py – orchestration wrapper that launches src.train as subprocess."""
from __future__ import annotations

import subprocess
import sys
from typing import List

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    overrides: List[str] = HydraConfig.get().overrides.task.copy()
    filtered = [o for o in overrides if not o.startswith(("run=", "results_dir=", "trial_mode="))]
    filtered.append(f"run={cfg.run}")
    filtered.append(f"results_dir={cfg.results_dir}")
    if cfg.get("trial_mode", False):
        filtered.append("trial_mode=true")
    cmd: List[str] = [sys.executable, "-u", "-m", "src.train", *filtered]
    print("[main] Launching subprocess:\n  ", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
