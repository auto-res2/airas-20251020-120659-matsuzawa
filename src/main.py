import subprocess
import sys
from typing import List

import hydra
from omegaconf import DictConfig

########################################################################################################################
# Orchestrator – launches src.train as a subprocess
########################################################################################################################

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:  # noqa: D401
    # ------------------------------------------------------------------
    # Retrieve selected run (Hydra stores choice in runtime choices)
    # ------------------------------------------------------------------
    run_choice = cfg.hydra.runtime.choices.get("run")
    if run_choice is None:
        raise ValueError("Specify run=<run_id> corresponding to a YAML in config/run/")

    # ------------------------------------------------------------------
    # Build CLI override list for subprocess
    # ------------------------------------------------------------------
    overrides: List[str] = [f"run={run_choice}", f"results_dir={cfg.results_dir}"]

    if cfg.get("trial_mode", False):
        overrides.extend(["trial_mode=true", "wandb.mode=disabled", "run.optuna.n_trials=0"])

    cmd = [sys.executable, "-u", "-m", "src.train"] + overrides
    print("Launching:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()