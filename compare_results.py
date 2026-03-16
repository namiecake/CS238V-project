"""
compare_results.py — Compare CTG classifiers across pH thresholds.

Loads saved results from RF, CNN, and XGBoost classifiers at both pH
thresholds (7.05, 7.15) and produces:
  1. A summary table (printed + saved as CSV).
  2. A combined ROC curve with all 6 variants.

Usage:
    python compare_results.py
"""

import pickle
import sys

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load_data import ph_threshold_suffix

BASE_DIR    = Path(__file__).parent
MODEL_DIR   = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR   = BASE_DIR / "plots"

THRESHOLDS  = [7.15, 7.05]
CLASSIFIERS = ["rf", "cnn", "xgb"]


# ─── Loaders ─────────────────────────────────────────────────────────────────

def _load_rf_cv(suffix: str) -> dict | None:
    path = RESULTS_DIR / f"cv_results_{suffix}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_xgb_cv(suffix: str) -> dict | None:
    path = RESULTS_DIR / f"cv_results_xgb_{suffix}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_cnn_cv(suffix: str) -> dict | None:
    import torch
    from cnn_classifier import CTG_CNN
    sys.modules["__main__"].CTG_CNN = CTG_CNN
    path = MODEL_DIR / f"cnn_classifier_{suffix}.pt"
    if not path.exists():
        return None
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    y_true = bundle["y_true"]
    y_prob = bundle["y_prob"]
    y_pred = bundle.get("y_pred", (y_prob >= 0.5).astype(int))
    auc    = bundle.get("auc", None)
    if auc is None:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_prob)
    sensitivity = bundle.get("sensitivity", None)
    specificity = bundle.get("specificity", None)
    if sensitivity is None or specificity is None:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"y_true": y_true, "y_prob": y_prob,
            "auc": auc, "sensitivity": sensitivity, "specificity": specificity}


def _load_cv(clf_name: str, suffix: str) -> dict | None:
    if clf_name == "rf":
        return _load_rf_cv(suffix)
    elif clf_name == "xgb":
        return _load_xgb_cv(suffix)
    elif clf_name == "cnn":
        return _load_cnn_cv(suffix)
    return None


def _load_violations(tag: str) -> dict | None:
    path = RESULTS_DIR / f"violations_{tag}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_perturbation(tag: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / f"perturbation_results_{tag}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ─── Summary Table ───────────────────────────────────────────────────────────

def build_summary_table() -> pd.DataFrame:
    rows: list[dict] = []

    for threshold in THRESHOLDS:
        suffix = ph_threshold_suffix(threshold)
        for clf_name in CLASSIFIERS:
            tag = f"{clf_name}_{suffix}"

            cv = _load_cv(clf_name, suffix)
            if cv is None:
                print(f"  [SKIP] No results for {clf_name.upper()} @ pH<{threshold}")
                continue

            violations = _load_violations(tag)
            n_spec1 = len(violations["spec1_tachycardia"])  if violations else "N/A"
            n_spec2 = len(violations["spec2_monotonicity"]) if violations else "N/A"
            n_spec3 = len(violations["spec3_noise"])        if violations else "N/A"

            perturb = _load_perturbation(tag)
            if perturb is not None and len(perturb) > 0:
                n_tested  = len(perturb)
                n_flipped = int(perturb["flipped"].sum())
                flip_rate = n_flipped / n_tested
                mean_mag  = (
                    perturb.loc[perturb["flipped"], "perturbation_magnitude"].mean()
                    if n_flipped > 0 else np.nan
                )
            else:
                flip_rate = np.nan
                mean_mag  = np.nan

            rows.append({
                "Classifier":          clf_name.upper(),
                "pH Threshold":        f"<{threshold}",
                "AUC":                 round(cv["auc"], 3),
                "Sensitivity":         round(cv["sensitivity"], 3),
                "Specificity":         round(cv["specificity"], 3),
                "Spec1 Violations":    n_spec1,
                "Spec2 Violations":    n_spec2,
                "Spec3 Violations":    n_spec3,
                "Flip Rate":           round(flip_rate, 3) if not np.isnan(flip_rate) else np.nan,
                "Mean Flip Magnitude": round(mean_mag, 2)  if not np.isnan(mean_mag)  else np.nan,
            })

    return pd.DataFrame(rows)


# ─── Combined ROC Plot ───────────────────────────────────────────────────────

def plot_combined_roc():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))

    colors  = {"rf": "#2980b9", "cnn": "#e74c3c", "xgb": "#8e44ad"}
    styles  = {7.15: "-", 7.05: "--"}
    plotted = False

    for threshold in THRESHOLDS:
        suffix = ph_threshold_suffix(threshold)
        for clf_name in CLASSIFIERS:
            cv = _load_cv(clf_name, suffix)
            if cv is None:
                continue
            fpr, tpr, _ = roc_curve(cv["y_true"], cv["y_prob"])
            label = f"{clf_name.upper()} pH<{threshold} (AUC={cv['auc']:.3f})"
            ax.plot(fpr, tpr, color=colors[clf_name],
                    linestyle=styles[threshold], linewidth=2, label=label)
            plotted = True

    if not plotted:
        print("No ROC data available to plot.")
        plt.close()
        return

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Combined ROC Curves — RF / CNN / XGBoost × Both Thresholds",
                 fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "combined_roc_curves.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved combined ROC curve to {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("COMPARISON — CTG Classifier Validation Results (RF / CNN / XGBoost)")
    print("=" * 70)

    summary = build_summary_table()

    if len(summary) == 0:
        print("\nNo results found. Run the pipeline first (python run_all.py).")
        return

    print("\n" + summary.to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "comparison_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved summary to {csv_path}")

    plot_combined_roc()
    print("\nDone.")


if __name__ == "__main__":
    main()