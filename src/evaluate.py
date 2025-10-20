import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
import yaml
from scipy import stats

sns.set(style="whitegrid")

########################################################################################################################
# Helper utilities
########################################################################################################################

def mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

########################################################################################################################
# Plot helpers
########################################################################################################################

def plot_learning_curve(df: pd.DataFrame, run_id: str, out_path: Path) -> None:
    if df.empty or "batch_acc" not in df.columns:
        return
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df, x="step", y="batch_acc", marker="o")
    plt.title(f"Learning curve – {run_id}")
    plt.xlabel("Step")
    plt.ylabel("Batch accuracy")
    best = df["batch_acc"].max()
    plt.annotate(f"Best={best:.3f}", xy=(df["step"].iloc[-1], df["batch_acc"].iloc[-1]))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, run_id: str, out_path: Path) -> None:
    if cm.size == 0:
        return
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion matrix – {run_id}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_bar_comparison(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="run_id", y="final_accuracy")
    plt.title("Final accuracy comparison")
    for i, row in df.iterrows():
        plt.text(i, row["final_accuracy"] + 0.005, f"{row['final_accuracy']:.3f}", ha="center")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_batch_acc_distribution(batch_dict: Dict[str, List[float]], out_path: Path) -> None:
    if not batch_dict:
        return
    plt.figure(figsize=(8, 4))
    records = []
    for rid, vals in batch_dict.items():
        for v in vals:
            records.append({"run_id": rid, "batch_acc": v})
    df = pd.DataFrame(records)
    sns.boxplot(data=df, x="run_id", y="batch_acc")
    plt.title("Batch accuracy distribution across runs")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

########################################################################################################################
# Per-run processing
########################################################################################################################

def process_run(api: wandb.Api, entity: str, project: str, run_id: str, out_dir: Path) -> Tuple[Dict, List[float]]:
    run = api.run(f"{entity}/{project}/{run_id}")
    history_df = run.history(keys=["step", "batch_acc"], samples=10000)
    final_accuracy = run.summary.get("final_accuracy")
    cm_list = run.summary.get("confusion_matrix")
    confusion_matrix = np.array(cm_list) if cm_list is not None else np.empty((0, 0))

    mkdir(out_dir)
    metrics = {"run_id": run_id, "final_accuracy": final_accuracy}
    save_json(metrics, out_dir / "metrics.json")

    plot_learning_curve(history_df, run_id, out_dir / "learning_curve.pdf")
    if cm_list is not None:
        plot_confusion_matrix(confusion_matrix, run_id, out_dir / "confusion_matrix.pdf")

    # Print generated file paths for CI visibility
    for p in [out_dir / "metrics.json", out_dir / "learning_curve.pdf", out_dir / "confusion_matrix.pdf"]:
        if p.exists():
            print(str(p))

    batch_vals = history_df["batch_acc"].dropna().tolist() if "batch_acc" in history_df else []
    return metrics, batch_vals

########################################################################################################################
# Aggregated analysis
########################################################################################################################

def aggregated_analysis(all_metrics: List[Dict], batch_dict: Dict[str, List[float]], comparison_dir: Path) -> None:
    mkdir(comparison_dir)

    df = pd.DataFrame(all_metrics)
    save_json(df.to_dict(orient="records"), comparison_dir / "aggregated_metrics.json")

    bar_path = comparison_dir / "final_accuracy_comparison.pdf"
    plot_bar_comparison(df, bar_path)
    if bar_path.exists():
        print(str(bar_path))

    box_path = comparison_dir / "batch_acc_distribution.pdf"
    plot_batch_acc_distribution(batch_dict, box_path)
    if box_path.exists():
        print(str(box_path))

    # Statistical significance (Welch’s t-test)
    sig_results = {}
    run_ids = list(batch_dict.keys())
    for i in range(len(run_ids)):
        for j in range(i + 1, len(run_ids)):
            r1, r2 = run_ids[i], run_ids[j]
            vals1, vals2 = batch_dict[r1], batch_dict[r2]
            if len(vals1) > 1 and len(vals2) > 1:
                t_stat, p_val = stats.ttest_ind(vals1, vals2, equal_var=False)
                sig_results[f"{r1}_vs_{r2}"] = {"t_stat": float(t_stat), "p_value": float(p_val)}
    save_json(sig_results, comparison_dir / "significance_tests.json")
    print(str(comparison_dir / "significance_tests.json"))

########################################################################################################################
# Entry point
########################################################################################################################

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("run_ids", type=str, help='JSON list, e.g. "[\"run1\", \"run2\"]"')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    run_ids: List[str] = json.loads(args.run_ids)

    cfg_path = results_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found at {cfg_path}. Ensure training finished successfully.")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    entity = cfg["wandb"]["entity"]
    project = cfg["wandb"]["project"]

    api = wandb.Api()
    all_metrics: List[Dict] = []
    batch_dict: Dict[str, List[float]] = {}

    for rid in run_ids:
        out_dir = results_dir / rid
        metrics, batch_vals = process_run(api, entity, project, rid, out_dir)
        all_metrics.append(metrics)
        batch_dict[rid] = batch_vals

    aggregated_analysis(all_metrics, batch_dict, results_dir / "comparison")


if __name__ == "__main__":
    main()