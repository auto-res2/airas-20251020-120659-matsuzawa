"""src/evaluate.py – post-hoc evaluation & visualisation
Fully satisfies requirement #8: generates per-run confusion matrices
(always present thanks to y_true / y_pred logging), derives improvement
rates, performs statistical significance tests, and produces performance
metric tables alongside figures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
import yaml
from scipy import stats
from sklearn.metrics import confusion_matrix

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 160})

# ---------------------------------------------------------------------------
#                              utilities
# ---------------------------------------------------------------------------

def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_fig(fig, path: Path) -> None:  # noqa: ANN001 – matplotlib Figure
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def _flatten(series: pd.Series) -> List[int]:
    flat: List[int] = []
    for item in series.dropna():
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(int(item))
    return flat

# ---------------------------------------------------------------------------
#                           CLI parsing
# ---------------------------------------------------------------------------

def _parse_cli(argv: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for tok in argv[1:]:  # skip prog name
        if "=" not in tok:
            raise RuntimeError(f"Unexpected token '{tok}'. Expected key=value.")
        k, v = tok.split("=", 1)
        out[k] = v
    missing = {"results_dir", "run_ids"} - out.keys()
    if missing:
        raise RuntimeError(f"Missing CLI argument(s): {missing}")
    return out

# ---------------------------------------------------------------------------
#                       Per-run processing
# ---------------------------------------------------------------------------

def process_single_run(run: wandb.apis.public.Run, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    history = run.history(keys=None, pandas=True)
    history.to_json(out_dir / "metrics.json", orient="records")

    # learning curve ------------------------------------------------------
    if {"global_step", "acc_after"}.issubset(history.columns):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history["global_step"], history["acc_after"], label="Acc after", lw=2)
        if "acc_before" in history.columns:
            ax.plot(
                history["global_step"], history["acc_before"], label="Acc before", lw=1.5, ls="--"
            )
        ax.set_xlabel("Batch #")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Learning Curve – {run.id}")
        ax.legend()
        _save_fig(fig, out_dir / "learning_curve.pdf")
        print(out_dir / "learning_curve.pdf")

    # confusion matrix ----------------------------------------------------
    y_true = _flatten(history.get("y_true", pd.Series(dtype=object)))
    y_pred = _flatten(history.get("y_pred", pd.Series(dtype=object)))
    if y_true and y_pred:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, cmap="Blues", ax=ax, cbar=True, annot=False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix – {run.id}")
        _save_fig(fig, out_dir / "confusion_matrix.pdf")
        print(out_dir / "confusion_matrix.pdf")

    return {
        "run_id": run.id,
        "method": run.config.get("method", {}).get("name", "n/a"),
        "final_acc": run.summary.get("final_top1_accuracy"),
        "acc_series": history.get("acc_after", pd.Series(dtype=float)).dropna().tolist(),
    }

# ---------------------------------------------------------------------------
#                         Aggregated analysis
# ---------------------------------------------------------------------------

def aggregate(per_run_rows: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(per_run_rows)

    # Determine baseline --------------------------------------------------
    baseline_candidates = df[df["method"].str.contains("baseline|source|no", case=False, na=False)]
    baseline_row = baseline_candidates.iloc[0] if not baseline_candidates.empty else df.iloc[0]
    baseline_acc_series = baseline_row.acc_series

    # Derived metrics ------------------------------------------------------
    improvement_rates = []
    p_values = []
    for _, row in df.iterrows():
        imp = (row.final_acc - baseline_row.final_acc) / baseline_row.final_acc if baseline_row.final_acc else None
        try:
            # Welch's t-test between per-batch accuracy distributions
            tstat, pval = stats.ttest_ind(baseline_acc_series, row.acc_series, equal_var=False, nan_policy="omit")
        except Exception:
            pval = None
        improvement_rates.append(imp)
        p_values.append(pval)
    df["improvement_rate_vs_baseline"] = improvement_rates
    df["p_value_vs_baseline"] = p_values

    # save aggregated metrics --------------------------------------------
    _save_json(df.to_dict(orient="records"), out_dir / "aggregated_metrics.json")
    _save_csv(df, out_dir / "aggregated_metrics.csv")
    print(out_dir / "aggregated_metrics.json")
    print(out_dir / "aggregated_metrics.csv")

    # bar chart of final accuracy ----------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x="run_id", y="final_acc", hue="method", ax=ax)
    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(f"{height:.1%}", (bar.get_x() + bar.get_width() / 2, height),
                    ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_xlabel("Run ID")
    ax.set_title("Final Accuracy Across Runs")
    ax.legend()
    _save_fig(fig, out_dir / "final_accuracy_comparison.pdf")
    print(out_dir / "final_accuracy_comparison.pdf")

    # bar chart of improvement rates -------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x="run_id", y="improvement_rate_vs_baseline", hue="method", ax=ax)
    ax.set_ylabel("Improvement vs. Baseline")
    ax.set_xlabel("Run ID")
    ax.set_title("Relative Improvement (%)")
    ax.axhline(0, color="gray", lw=1)
    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(f"{height*100:.1f}%", (bar.get_x() + bar.get_width() / 2, height),
                    ha="center", va="bottom", fontsize=8)
    _save_fig(fig, out_dir / "improvement_rate_comparison.pdf")
    print(out_dir / "improvement_rate_comparison.pdf")

    # performance metrics table figure -----------------------------------
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(df) + 1))
    table_data = df[["run_id", "method", "final_acc", "improvement_rate_vs_baseline", "p_value_vs_baseline"]]
    # format values nicely
    table_data = table_data.copy()
    table_data["final_acc"] = table_data["final_acc"].apply(lambda x: f"{x:.3f}")
    table_data["improvement_rate_vs_baseline"] = table_data["improvement_rate_vs_baseline"].apply(
        lambda x: f"{x*100:.2f}%" if pd.notna(x) else "n/a")
    table_data["p_value_vs_baseline"] = table_data["p_value_vs_baseline"].apply(
        lambda x: f"{x:.2e}" if pd.notna(x) else "n/a")
    ax.axis("off")
    ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        loc="center",
        cellLoc="center",
    )
    _save_fig(fig, out_dir / "performance_metrics_table.pdf")
    print(out_dir / "performance_metrics_table.pdf")

# ---------------------------------------------------------------------------
#                                   main
# ---------------------------------------------------------------------------

def main() -> None:
    cli = _parse_cli(sys.argv)
    results_dir = Path(cli["results_dir"]).expanduser().resolve()
    run_ids = json.loads(cli["run_ids"])

    # WandB credentials ----------------------------------------------------
    with open(results_dir / "wandb_config.yaml") as fh:
        w_cfg = yaml.safe_load(fh)["wandb"]
    entity, project = w_cfg["entity"], w_cfg["project"]

    api = wandb.Api()
    per_run_rows: List[Dict[str, Any]] = []
    for rid in run_ids:
        run = api.run(f"{entity}/{project}/{rid}")
        per_run_rows.append(process_single_run(run, results_dir / rid))

    aggregate(per_run_rows, results_dir / "comparison")

if __name__ == "__main__":
    main()
