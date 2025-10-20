# src/evaluate.py
"""Independent evaluation & visualisation script (unchanged)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
import yaml
from scipy import stats

sns.set_style("whitegrid")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("results_dir", type=str)
    p.add_argument("run_ids", type=str, help="JSON list e.g. '[\"run1\",\"run2\"]'")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_wandb_config(results_dir: Path) -> Dict:
    cfg_file = results_dir / "config.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Global config.yaml not found at {cfg_file}")
    with cfg_file.open() as f:
        return yaml.safe_load(f)


def _export_metrics(df: pd.DataFrame, out_path: Path):
    df.to_json(out_path, orient="records", lines=True)


def _plot_learning_curve(df: pd.DataFrame, run_id: str, metric: str, out: Path):
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=df.index, y=df[metric])
    plt.title(f"{run_id} – {metric}")
    plt.xlabel("Step")
    plt.ylabel(metric)
    if not df[metric].dropna().empty:
        final_val = df[metric].dropna().iloc[-1]
        plt.annotate(f"{final_val:.2f}", xy=(df.index[-1], final_val),
                     xytext=(-40, 10), textcoords="offset points", fontsize=8)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def per_run(api: wandb.Api, entity: str, project: str, run_id: str, out_dir: Path) -> Dict:
    run = api.run(f"{entity}/{project}/{run_id}")
    hist_df = run.history(pandas=True)

    _export_metrics(hist_df, out_dir / "metrics.json")

    figs: List[Path] = []
    for metric in ["top1_accuracy"]:
        if metric in hist_df.columns:
            fig_path = out_dir / f"learning_curve_{metric}.pdf"
            _plot_learning_curve(hist_df, run_id, metric, fig_path)
            figs.append(fig_path)

    cm = run.summary.get("conf_mat")
    if cm is not None:
        cm = np.asarray(cm, dtype=int)
        cls_names = [str(i) for i in range(cm.shape[0])]
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=cls_names, yticklabels=cls_names)
        plt.title(f"{run_id} – Confusion Matrix")
        plt.ylabel("True")
        plt.xlabel("Pred")
        plt.tight_layout()
        cm_path = out_dir / "confusion_matrix.pdf"
        plt.savefig(cm_path)
        plt.close()
        figs.append(cm_path)

    final_acc = run.summary.get("final_top1_accuracy", float("nan"))
    method = run.config.get("run", {}).get("method", "unknown")

    print(f"[Per-run] {run_id}: saved {len(figs)} figure(s)")
    for f in figs:
        print("  •", f)

    return {"run_id": run_id, "method": method, "final_accuracy": final_acc}


# -----------------------------------------------------------------------------
# Aggregated analysis
# -----------------------------------------------------------------------------

def _bar_plot(df: pd.DataFrame, out: Path):
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="run_id", y="final_accuracy", hue="method", palette="viridis")
    plt.xticks(rotation=45, ha="right")
    for i, row in df.iterrows():
        plt.text(i, row.final_accuracy + 0.5, f"{row.final_accuracy:.1f}", ha="center", fontsize=8)
    plt.ylabel("Final Top-1 Accuracy (%)")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def _box_plot(df: pd.DataFrame, out: Path):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="method", y="final_accuracy", palette="Set2")
    sns.stripplot(data=df, x="method", y="final_accuracy", color="black", size=4, jitter=True)
    plt.ylabel("Final Top-1 Accuracy (%)")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def aggregated(per_run_stats: List[Dict], results_dir: Path):
    comp_dir = results_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(per_run_stats)
    df.to_csv(comp_dir / "aggregated_metrics_table.csv", index=False)

    with (comp_dir / "aggregated_metrics.json").open("w") as f:
        json.dump(per_run_stats, f, indent=2)

    _bar_plot(df, comp_dir / "final_accuracy_comparison.pdf")
    _box_plot(df, comp_dir / "boxplot_accuracy.pdf")

    print("\n[Aggregated] artefacts:")
    for p in comp_dir.iterdir():
        print("  •", p)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = _parse_cli()
    results_dir = Path(args.results_dir).expanduser().resolve()
    run_ids = json.loads(args.run_ids)

    wb_cfg = _load_wandb_config(results_dir)
    entity = wb_cfg["wandb"]["entity"]
    project = wb_cfg["wandb"]["project"]

    api = wandb.Api()

    stats: List[Dict] = []
    for rid in run_ids:
        rdir = results_dir / rid
        rdir.mkdir(parents=True, exist_ok=True)
        s = per_run(api, entity, project, rid, rdir)
        stats.append(s)

    aggregated(stats, results_dir)


if __name__ == "__main__":
    main()