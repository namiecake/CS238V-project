"""
train_classifier.py — Train baseline CTG classifiers.

Trains a feature-based classifier on extracted CTG features to predict
fetal distress (pathological pH). This is the "system under validation."

Usage:
    python train_classifier.py
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt

from load_data import load_all_data

# ─── Constants ───────────────────────────────────────────────────────────────

MODEL_DIR = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = Path(__file__).parent / "plots"

FEATURE_COLS = [
    "baseline_fhr", "variability_stv", "variability_ltv",
    "mean_fhr", "std_fhr", "min_fhr", "max_fhr",
    "n_accelerations", "n_decelerations",
    "pct_tachycardia", "pct_bradycardia", "pct_low_variability",
    "signal_quality",
    "uc_mean", "uc_std", "uc_max",
]


# ─── Data Preparation ───────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame):
    """Prepare feature matrix and labels, handling NaN values."""
    # Filter to records with valid labels
    df_valid = df[df["label_binary"] >= 0].copy()

    X = df_valid[FEATURE_COLS].copy()
    y = df_valid["label_binary"].values

    # Fill NaN features with column medians
    for col in FEATURE_COLS:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)

    record_ids = df_valid["record_id"].values
    return X.values, y, record_ids, X.columns.tolist()


# ─── Training ────────────────────────────────────────────────────────────────

def train_and_evaluate(X, y, record_ids):
    """
    Train classifiers with cross-validation and return the best one.

    Uses stratified 5-fold CV so every record appears in exactly one test fold.
    Returns the trained model along with cross-validated predictions for analysis.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define classifiers to compare
    classifiers = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            random_state=42
        ),
        "SVM_RBF": SVC(
            kernel="rbf", class_weight="balanced", probability=True,
            random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("=" * 60)
    print("CLASSIFIER COMPARISON (5-fold Stratified CV)")
    print("=" * 60)

    for name, clf in classifiers.items():
        print(f"\n── {name} ──")

        # Cross-validated predictions (each sample predicted when in test set)
        y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)
        y_prob = cross_val_predict(clf, X_scaled, y, cv=cv, method="predict_proba")[:, 1]

        # Metrics
        auc = roc_auc_score(y, y_prob)
        cm = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred, target_names=["Normal", "Pathological"])

        print(report)
        print(f"AUC-ROC: {auc:.3f}")
        print(f"Confusion Matrix:\n{cm}")

        # Compute sensitivity (recall for pathological) and specificity
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"Sensitivity (detect distress): {sensitivity:.3f}")
        print(f"Specificity (avoid false alarm): {specificity:.3f}")

        results[name] = {
            "model": clf,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "auc": auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "confusion_matrix": cm,
        }

    # Select best model by AUC
    best_name = max(results, key=lambda k: results[k]["auc"])
    print(f"\n{'=' * 60}")
    print(f"Best model: {best_name} (AUC={results[best_name]['auc']:.3f})")
    print(f"{'=' * 60}")

    # Retrain best model on all data for falsification
    best_clf = classifiers[best_name]
    best_clf.fit(X_scaled, y)

    return best_clf, scaler, results, best_name


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_roc_curves(results: dict, y_true: np.array):
    """Plot ROC curves for all classifiers."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_true, res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — CTG Classifier Comparison", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_curves.png", dpi=150)
    plt.close()
    print(f"Saved ROC curves to {PLOTS_DIR / 'roc_curves.png'}")


def plot_feature_importance(clf, feature_names: list):
    """Plot feature importance for tree-based classifiers."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        sorted_idx = np.argsort(importances)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.barh(range(len(sorted_idx)), importances[sorted_idx], align="center")
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.set_xlabel("Feature Importance", fontsize=12)
        ax.set_title("Feature Importance — CTG Classifier", fontsize=14)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "feature_importance.png", dpi=150)
        plt.close()
        print(f"Saved feature importance to {PLOTS_DIR / 'feature_importance.png'}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # Load data
    print("Loading data...")
    df = load_all_data()

    # Prepare features
    X, y, record_ids, feature_names = prepare_data(df)
    print(f"\nDataset: {len(y)} samples, {X.shape[1]} features")
    print(f"Class balance: {np.sum(y==0)} normal, {np.sum(y==1)} pathological")

    # Train and evaluate
    best_clf, scaler, results, best_name = train_and_evaluate(X, y, record_ids)

    # Save model and associated objects
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "classifier": best_clf,
        "scaler": scaler,
        "feature_names": feature_names,
        "best_name": best_name,
    }
    with open(MODEL_DIR / "ctg_classifier.pkl", "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"\nSaved model to {MODEL_DIR / 'ctg_classifier.pkl'}")

    # Save cross-val predictions for falsification
    cv_results = {
        "record_ids": record_ids,
        "y_true": y,
        "results": results,
    }
    with open(RESULTS_DIR / "cv_results.pkl", "wb") as f:
        pickle.dump(cv_results, f)

    # Plots
    plot_roc_curves(results, y)
    plot_feature_importance(best_clf, feature_names)

    return best_clf, scaler, results


if __name__ == "__main__":
    main()
