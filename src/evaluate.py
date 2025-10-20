# src/evaluate.py
"""Independent evaluation and visualisation script.
Fetch metrics from WandB and generate per-run and aggregated artefacts.
CLI:  python -m src.evaluate results_dir=PATH run_ids='["run-1", …]'
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
import yaml
from scipy import stats
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sns.set(style="whitegrid")

###############################################################################
# Argument parsing ------------------------------------------------------------
###############################################################################

def _parse_cli() -> Tuple[Path, List[str]]:
    """Parse key=value style CLI arguments."""
    kv: Dict[str, str] = {}
    for arg in sys.argv[1:]:
        if "=" not in arg:
            raise ValueError(
                f"Argument '{arg}' is not in key=value form (expected e.g. results_dir=/path run_ids='[…]')."
            )
        k, v = arg.split("=", 1)
        kv[k.strip()] = v.strip()

    if "results_dir" not in kv or "run_ids" not in kv:
        raise ValueError("Both results_dir and run_ids must be provided in key=value form.")

    results_dir = Path(kv["results_dir"]).expanduser()

    # Remove optional surrounding quotes so json.loads works irrespective of quoting
    run_ids_raw = kv["run_ids"].strip()
    if (
        (run_ids_raw.startswith("'") and run_ids_raw.endswith("'"))
        or (run_ids_raw.startswith("\"") and run_ids_raw.endswith("\""))
    ):
        run_ids_raw = run_ids_raw[1:-1]
    run_ids = json.loads(run_ids_raw)
    if not isinstance(run_ids, list):
        raise ValueError("run_ids must decode to a JSON list of strings.")

    return results_dir, run_ids

###############################################################################
# Helper functions ------------------------------------------------------------
###############################################################################

def save_learning_curve(df: pd.DataFrame, out_path: Path, metric: str = "val_acc_batch") -> None:
    if metric not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df[metric], linewidth=2, color='steelblue', label=metric)
    if df[metric].notna().any():
        best_idx = df[metric].idxmax()
        best_val = df[metric].max()
        ax.scatter(best_idx, best_val, color='red', s=100, zorder=5, label=f'Best: {best_val:.2f}')
        ax.annotate(f'{best_val:.2f}', (best_idx, best_val), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red'))
    ax.set_xlabel('Step', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    ax.set_title(f'{metric.replace("_", " ").title()} over Time', fontsize=16, fontweight='bold', pad=20)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_confusion(y_true: List[int], y_pred: List[int], out_path: Path, classes: List[str]):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='d')
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def mcnemar_significance(y_true: List[int], preds_a: List[int], preds_b: List[int]) -> float:
    correct_a = np.array(preds_a) == np.array(y_true)
    correct_b = np.array(preds_b) == np.array(y_true)
    n01 = np.logical_and(correct_a, ~correct_b).sum()
    n10 = np.logical_and(~correct_a, correct_b).sum()
    if (n01 + n10) == 0:
        return 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return float(stats.chi2.sf(stat, df=1))

###############################################################################
# Main evaluation workflow ----------------------------------------------------
###############################################################################

def parse_results_from_logs(results_dir: Path, run_id: str) -> float:
    stdout_path = results_dir / "stdout.txt"
    if not stdout_path.exists():
        return np.nan
    
    import re
    with open(stdout_path, "r") as f:
        content = f.read()
    
    pattern = rf'\[RESULT\]\s+{re.escape(run_id)}\s+–\s+Final\s+Top-1\s+Acc:\s+([\d.]+)%'
    matches = re.findall(pattern, content)
    if matches:
        return float(matches[-1])
    return np.nan


def main():
    results_dir, run_ids = _parse_cli()
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = results_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} not found. Ensure you passed the same results_dir used during training."
        )
    with open(cfg_path, "r") as f:
        global_cfg = yaml.safe_load(f)
    entity = global_cfg["wandb"]["entity"]
    project = global_cfg["wandb"]["project"]
    wandb_mode = global_cfg["wandb"].get("mode", "online")
    
    use_wandb = wandb_mode != "disabled"
    if use_wandb:
        api = wandb.Api()

    aggregated: Dict[str, Dict[str, float]] = {}
    predictions_cache: Dict[str, Tuple[List[int], List[int]]] = {}
    generated_paths: List[str] = []

    # ---------------- Per-run processing ---------------------------------
    for run_id in run_ids:
        print(f"[Eval] Processing {run_id}")
        subdir = results_dir / run_id
        subdir.mkdir(parents=True, exist_ok=True)

        if use_wandb:
            try:
                run = api.run(f"{entity}/{project}/{run_id}")
            except wandb.errors.CommError:
                print(f"  [WARN] Run {run_id} not found in WandB – skipping.")
                continue

            history = run.history(samples=100_000)
            metrics_path = subdir / "metrics.json"
            history.to_json(metrics_path, orient="records", indent=2)
            generated_paths.append(str(metrics_path))

            lc_path = subdir / "learning_curve_val_acc_batch.pdf"
            save_learning_curve(history, lc_path)
            generated_paths.append(str(lc_path))

            y_true = run.summary.get("y_true", [])
            y_pred = run.summary.get("y_pred", [])
            if y_true and y_pred:
                cm_path = subdir / "confusion_matrix.pdf"
                num_classes = max(max(y_true), max(y_pred)) + 1
                classes = [str(i) for i in range(num_classes)]
                save_confusion(y_true, y_pred, cm_path, classes)
                generated_paths.append(str(cm_path))
            predictions_cache[run_id] = (y_true, y_pred)

            final_acc = float(run.summary.get("top1_accuracy", np.nan))
        else:
            final_acc = parse_results_from_logs(results_dir, run_id)
            if np.isnan(final_acc):
                print(f"  [WARN] Run {run_id} not found in logs – skipping.")
                continue
            print(f"  [INFO] Found {run_id} with Top-1 Acc: {final_acc:.2f}%")
            predictions_cache[run_id] = ([], [])

        aggregated[run_id] = {"top1_accuracy": final_acc}

    # ---------------- Aggregated comparison -----------------------------
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(exist_ok=True)

    baseline_id = run_ids[0] if run_ids else None
    baseline_acc = aggregated.get(baseline_id, {}).get("top1_accuracy", np.nan)

    for rid, metrics in aggregated.items():
        if baseline_id and rid != baseline_id and not np.isnan(baseline_acc):
            metrics["improvement_rate"] = (metrics["top1_accuracy"] - baseline_acc) / baseline_acc
        else:
            metrics["improvement_rate"] = 0.0

        if baseline_id and rid != baseline_id:
            y_true_b, y_pred_b = predictions_cache.get(baseline_id, ([], []))
            y_true_o, y_pred_o = predictions_cache.get(rid, ([], []))
            if y_true_b and y_pred_b and y_true_o and y_pred_o:
                p_val = mcnemar_significance(y_true_b, y_pred_b, y_pred_o)
            else:
                p_val = np.nan
            metrics["p_value_vs_baseline"] = p_val
        else:
            metrics["p_value_vs_baseline"] = np.nan

    # Save aggregated metrics -------------------------------------------
    agg_path = comparison_dir / "aggregated_metrics.json"
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    generated_paths.append(str(agg_path))

    # Accuracy bar chart -------------------------------------------------
    labels = list(aggregated.keys())
    accuracies = [aggregated[k]["top1_accuracy"] for k in labels]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(labels)), accuracies, color='steelblue', edgecolor='black', linewidth=1.2)
    
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, 
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(accuracies) * 1.15 if accuracies else 100)
    
    plt.tight_layout()
    bar_path = comparison_dir / "top1_accuracy_comparison.pdf"
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    generated_paths.append(str(bar_path))

    # Improvement rates --------------------------------------------------
    if len(aggregated) >= 2 and baseline_id is not None:
        imp_labels = [l for l in labels if l != baseline_id]
        imp_vals = [aggregated[l]["improvement_rate"] * 100 for l in imp_labels]
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['green' if v >= 0 else 'red' for v in imp_vals]
        bars = ax.bar(range(len(imp_labels)), imp_vals, color=colors, edgecolor='black', linewidth=1.2, alpha=0.7)
        
        for i, (bar, v) in enumerate(zip(bars, imp_vals)):
            height = bar.get_height()
            y_pos = height + 0.5 if height >= 0 else height - 0.5
            va = 'bottom' if height >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos, 
                    f'{v:.2f}%', ha='center', va=va, fontsize=14, fontweight='bold')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xticks(range(len(imp_labels)))
        ax.set_xticklabels(imp_labels, rotation=45, ha='right', fontsize=12)
        ax.set_ylabel('Improvement over Baseline (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Method', fontsize=14, fontweight='bold')
        ax.set_title('Relative Improvement', fontsize=16, fontweight='bold', pad=20)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        imp_path = comparison_dir / "improvement_rates.pdf"
        plt.savefig(imp_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_paths.append(str(imp_path))

    # ---------------- Summary ------------------------------------------
    print("\n[Evaluation] Generated artefacts:")
    for p in generated_paths:
        print(p)


if __name__ == "__main__":
    main()
