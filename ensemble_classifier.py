"""
ensemble_classifier.py — Soft-voting ensemble of RF + XGBoost for CTG.

Combines the two feature-based classifiers by averaging their predicted
probabilities (soft voting). This often beats either model alone because
RF and XGBoost make different errors — RF is conservative (misses
pathological), XGBoost is more aggressive. Averaging smooths this out.

Interesting for falsification: does combining classifiers reduce or
redistribute specification violations?

Usage:
    python ensemble_classifier.py
    python ensemble_classifier.py --ph-threshold 7.05
"""

import argparse
import pickle

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load_data import load_all_data, DEFAULT_PH_THRESHOLD, ph_threshold_suffix

# ─── Constants ───────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
MODEL_DIR   = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR   = BASE_DIR / "plots"

FEATURE_COLS = [
    "baseline_fhr", "variability_stv", "variability_ltv",
    "mean_fhr", "std_fhr", "min_fhr", "max_fhr",
    "n_accelerations", "n_decelerations",
    "pct_tachycardia", "pct_bradycardia", "pct_low_variability",
    "signal_quality", "uc_mean", "uc_std", "uc_max",
]


# ─── Data Preparation ────────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame):
    df_valid = df[df["label_binary"] >= 0].copy()
    X = df_valid[FEATURE_COLS].copy()
    y = df_valid["label_binary"].values
    for col in FEATURE_COLS:
        X[col] = X[col].fillna(X[col].median())
    record_ids = df_valid["record_id"].values
    return X.values, y, record_ids, list(X.columns)


# ─── Training ────────────────────────────────────────────────────────────────

def train_and_evaluate(X, y):
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_neg, n_pos = np.bincount(y)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight="balanced", random_state=42,
    )
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=n_neg / n_pos,
        eval_metric="logloss", random_state=42, verbosity=0,
    )

    # Soft voting: average the predicted probabilities
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        voting="soft",
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Manual CV so we can get probabilities
    n = len(y)
    y_pred = np.zeros(n, dtype=int)
    y_prob = np.zeros(n)

    for train_idx, test_idx in cv.split(X_scaled, y):
        ensemble.fit(X_scaled[train_idx], y[train_idx])
        y_prob[test_idx] = ensemble.predict_proba(X_scaled[test_idx])[:, 1]
        y_pred[test_idx] = (y_prob[test_idx] >= 0.5).astype(int)

    auc = roc_auc_score(y, y_prob)
    cm  = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("=" * 60)
    print("RF + XGBoost Soft Voting Ensemble — 5-fold Stratified CV")
    print("=" * 60)
    print(classification_report(y, y_pred, target_names=["Normal", "Pathological"]))
    print(f"AUC-ROC:     {auc:.3f}")
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"Confusion Matrix:\n{cm}")

    # Retrain on all data for falsification
    ensemble.fit(X_scaled, y)

    return ensemble, scaler, {
        "y_pred": y_pred, "y_prob": y_prob,
        "auc": auc, "sensitivity": sensitivity, "specificity": specificity,
        "confusion_matrix": cm,
    }


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_roc_curve(y_true, y_prob, auc, suffix):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, linewidth=2, color="#e67e22",
            label=f"Ensemble RF+XGB (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — RF+XGBoost Ensemble")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / f"roc_curve_ensemble_{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved ROC curve to {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main(ph_threshold: float = DEFAULT_PH_THRESHOLD):
    suffix = ph_threshold_suffix(ph_threshold)

    print(f"Loading data (pH threshold = {ph_threshold})...")
    df = load_all_data(ph_threshold=ph_threshold)

    X, y, record_ids, feature_names = prepare_data(df)
    print(f"Dataset: {len(y)} samples, {X.shape[1]} features")
    print(f"Class balance: {np.sum(y==0)} normal, {np.sum(y==1)} pathological\n")

    ensemble, scaler, results = train_and_evaluate(X, y)

    # ── Save model ──
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "classifier":    ensemble,
        "scaler":        scaler,
        "feature_names": feature_names,
        "best_name":     "Ensemble_RF_XGB",
        "ph_threshold":  ph_threshold,
    }
    model_path = MODEL_DIR / f"ensemble_classifier_{suffix}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"\nSaved model to {model_path}")

    # ── Save CV results ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_results = {"record_ids": record_ids, "y_true": y, **results}
    cv_path = RESULTS_DIR / f"cv_results_ensemble_{suffix}.pkl"
    with open(cv_path, "wb") as f:
        pickle.dump(cv_results, f)
    print(f"Saved CV results to {cv_path}")

    plot_roc_curve(y, results["y_prob"], results["auc"], suffix)


def parse_args():
    parser = argparse.ArgumentParser(description="Train RF+XGBoost ensemble.")
    parser.add_argument(
        "--ph-threshold", type=float, default=DEFAULT_PH_THRESHOLD,
        help=f"pH threshold (default: {DEFAULT_PH_THRESHOLD})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(ph_threshold=args.ph_threshold)