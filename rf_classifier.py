"""
rf_classifier.py — Train a Random Forest CTG classifier.

Trains a feature-based classifier on extracted CTG features to predict
fetal distress (pathological pH).  This is one of two "systems under
validation" — the other is the 1D CNN in cnn_classifier.py.

Usage:
    python rf_classifier.py                        # default pH < 7.15
    python rf_classifier.py --ph-threshold 7.05
"""

import argparse
import pickle

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load_data import load_all_data, DEFAULT_PH_THRESHOLD, ph_threshold_suffix

# ─── Constants ───────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

FEATURE_COLS = [
    "baseline_fhr",
    "variability_stv",
    "variability_ltv",
    "mean_fhr",
    "std_fhr",
    "min_fhr",
    "max_fhr",
    "n_accelerations",
    "n_decelerations",
    "pct_tachycardia",
    "pct_bradycardia",
    "pct_low_variability",
    "signal_quality",
    "uc_mean",
    "uc_std",
    "uc_max",
]


# ─── Data Preparation ───────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame):
    """Prepare feature matrix and labels, filling NaN with column medians."""
    df_valid = df[df["label_binary"] >= 0].copy()

    X = df_valid[FEATURE_COLS].copy()
    y = df_valid["label_binary"].values

    for col in FEATURE_COLS:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)

    record_ids = df_valid["record_id"].values
    return X.values, y, record_ids, X.columns.tolist()


# ─── Training ────────────────────────────────────────────────────────────────

def train_and_evaluate(X, y):
    """Train Random Forest with 5-fold stratified CV.

    ``cross_val_predict`` ensures every sample is predicted exactly once
    (when it appears in the held-out test fold).  We call it twice — once
    for hard labels, once for probabilities — because sklearn does not
    return both in a single call.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)
    y_prob = cross_val_predict(clf, X_scaled, y, cv=cv, method="predict_proba")[
        :, 1
    ]

    auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(
        y, y_pred, target_names=["Normal", "Pathological"]
    )

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("=" * 60)
    print("Random Forest — 5-fold Stratified CV")
    print("=" * 60)
    print(report)
    print(f"AUC-ROC:     {auc:.3f}")
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"Confusion Matrix:\n{cm}")

    # Retrain on all data for use in falsification
    clf.fit(X_scaled, y)

    results = {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": cm,
    }
    return clf, scaler, results


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_roc_curve(y_true, y_prob, auc, suffix: str):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"RF (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Random Forest CTG Classifier")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / f"roc_curve_rf_{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved ROC curve to {path}")


def plot_feature_importance(clf, feature_names, suffix: str):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if not hasattr(clf, "feature_importances_"):
        return
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(sorted_idx)), importances[sorted_idx], align="center")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Feature Importance — Random Forest")
    fig.tight_layout()
    path = PLOTS_DIR / f"feature_importance_rf_{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved feature importance to {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main(ph_threshold: float = DEFAULT_PH_THRESHOLD):
    suffix = ph_threshold_suffix(ph_threshold)

    print(f"Loading data (pH threshold = {ph_threshold})...")
    df = load_all_data(ph_threshold=ph_threshold)

    X, y, record_ids, feature_names = prepare_data(df)
    print(f"Dataset: {len(y)} samples, {X.shape[1]} features")
    print(f"Class balance: {np.sum(y == 0)} normal, {np.sum(y == 1)} pathological\n")

    clf, scaler, results = train_and_evaluate(X, y)

    # ── Save model bundle ──
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "classifier": clf,
        "scaler": scaler,
        "feature_names": feature_names,
        "best_name": "RandomForest",
        "ph_threshold": ph_threshold,
    }
    model_path = MODEL_DIR / f"ctg_classifier_{suffix}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"\nSaved model to {model_path}")

    # ── Save CV predictions (used by compare_results.py) ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_results = {
        "record_ids": record_ids,
        "y_true": y,
        **results,
    }
    cv_path = RESULTS_DIR / f"cv_results_{suffix}.pkl"
    with open(cv_path, "wb") as f:
        pickle.dump(cv_results, f)
    print(f"Saved CV results to {cv_path}")

    # ── Plots ──
    plot_roc_curve(y, results["y_prob"], results["auc"], suffix)
    plot_feature_importance(clf, feature_names, suffix)

    return clf, scaler, results


def parse_args():
    parser = argparse.ArgumentParser(description="Train Random Forest CTG classifier.")
    parser.add_argument(
        "--ph-threshold",
        type=float,
        default=DEFAULT_PH_THRESHOLD,
        help=f"pH threshold for binary labeling (default: {DEFAULT_PH_THRESHOLD})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(ph_threshold=args.ph_threshold)
