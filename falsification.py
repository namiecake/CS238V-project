"""
falsification.py — Optimization-based falsification of CTG classifiers.

Supports three classifier backends: rf, cnn, xgb.

Two approaches:
1. Real-data search: Check all records for specification violations.
2. Perturbation-based: Use differential evolution to find minimal
   perturbations to correctly-classified samples that cause violations.

Usage:
    python falsification.py rf  --ph-threshold 7.15
    python falsification.py cnn --ph-threshold 7.15
    python falsification.py xgb --ph-threshold 7.15
"""

import argparse
import pickle
import sys

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import differential_evolution
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load_data import (
    load_all_data,
    load_all_raw_signals,
    load_raw_signals,
    compute_fhr_features,
    compute_uc_features,
    DEFAULT_PH_THRESHOLD,
    ph_threshold_suffix,
    SAMPLING_RATE,
)
from specifications import (
    spec_tachycardia_low_variability,
    spec_monotonicity,
    spec_noise_robustness,
)

MODEL_DIR   = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR   = Path(__file__).parent / "plots"


# ─── Predict-function factories ──────────────────────────────────────────────

def _feature_predict_fn(bundle):
    """Shared factory for any feature-based classifier (RF, XGBoost)."""
    clf           = bundle["classifier"]
    scaler        = bundle["scaler"]
    feature_names = bundle["feature_names"]

    def predict_fn(fhr):
        feats     = compute_fhr_features(fhr)
        uc_feats  = {"uc_mean": 0.0, "uc_std": 0.0, "uc_max": 0.0}
        all_feats = {**feats, **uc_feats}
        vec = []
        for fname in feature_names:
            val = all_feats.get(fname, 0.0)
            if isinstance(val, float) and np.isnan(val):
                val = 0.0
            vec.append(val)
        X        = np.array(vec).reshape(1, -1)
        X_scaled = scaler.transform(X)
        label    = int(clf.predict(X_scaled)[0])
        prob     = float(clf.predict_proba(X_scaled)[0, 1])
        return label, prob

    return predict_fn


def make_rf_predict_fn(suffix: str):
    path = MODEL_DIR / f"ctg_classifier_{suffix}.pkl"
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return _feature_predict_fn(bundle)


def make_xgb_predict_fn(suffix: str):
    path = MODEL_DIR / f"xgb_classifier_{suffix}.pkl"
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return _feature_predict_fn(bundle)


def make_cnn_predict_fn(suffix: str):
    import torch
    from cnn_classifier import CTG_CNN, CNNPredictor
    sys.modules["__main__"].CTG_CNN = CTG_CNN
    path   = MODEL_DIR / f"cnn_classifier_{suffix}.pt"
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    pred   = CNNPredictor(bundle["model"])
    return lambda fhr: pred(fhr)


def load_predict_fn(clf_name: str, suffix: str):
    if clf_name == "rf":
        return make_rf_predict_fn(suffix)
    elif clf_name == "cnn":
        return make_cnn_predict_fn(suffix)
    elif clf_name == "xgb":
        return make_xgb_predict_fn(suffix)
    else:
        raise ValueError(f"Unknown classifier: {clf_name!r}. Choose rf / cnn / xgb.")


# ═══════════════════════════════════════════════════════════════════════════════
# Approach 1: Real-data search
# ═══════════════════════════════════════════════════════════════════════════════

def search_real_data(predict_fn, df, raw_signals: dict, tag: str):
    print("\n" + "=" * 70)
    print(f"APPROACH 1: Real-data search  [{tag}]")
    print("=" * 70)

    violations = {
        "spec1_tachycardia":  [],
        "spec2_monotonicity": [],
        "spec3_noise":        [],
    }
    all_results = []
    records = df[df["label_binary"] >= 0]
    n = len(records)

    for i, (_, row) in enumerate(records.iterrows()):
        record_id  = int(row["record_id"])
        ph         = row["pH"]
        true_label = int(row["label_binary"])

        if record_id not in raw_signals:
            continue
        fhr = raw_signals[record_id]["fhr"].astype(np.float64)

        print(f"  [{i+1}/{n}] Record {record_id} (pH={ph:.2f}):",
              end="", flush=True)

        r1 = spec_tachycardia_low_variability(fhr, predict_fn)
        if not r1["satisfied"]:
            violations["spec1_tachycardia"].append(
                {"record_id": record_id, "pH": ph, **r1}
            )
            print(" SPEC1", end="")

        for wtype in ["increase_tachycardia", "decrease_variability",
                      "add_deceleration"]:
            r2 = spec_monotonicity(fhr, predict_fn,
                                   worsening_type=wtype, delta=15.0)
            if not r2["satisfied"]:
                violations["spec2_monotonicity"].append(
                    {"record_id": record_id, "pH": ph,
                     "worsening_type": wtype, **r2}
                )
                print(f" SPEC2({wtype[:4]})", end="")

        r3 = spec_noise_robustness(fhr, predict_fn,
                                   noise_std=3.0, n_trials=20)
        if not r3["satisfied"]:
            violations["spec3_noise"].append(
                {"record_id": record_id, "pH": ph, **r3}
            )
            print(f" SPEC3(flip={r3['flip_rate']:.0%})", end="")

        all_results.append({
            "record_id":        record_id,
            "pH":               ph,
            "true_label":       true_label,
            "spec1_satisfied":  r1["satisfied"],
            "spec1_robustness": r1["robustness"],
            "spec3_satisfied":  r3["satisfied"],
            "spec3_flip_rate":  r3["flip_rate"],
        })
        print()

    n_checked = len(all_results)
    print(f"\n{'─' * 70}")
    print(f"Records checked: {n_checked}")
    print(f"Spec 1 violations: {len(violations['spec1_tachycardia'])} "
          f"({len(violations['spec1_tachycardia'])/n_checked:.1%})")
    print(f"Spec 2 violations: {len(violations['spec2_monotonicity'])} "
          f"({len(violations['spec2_monotonicity'])/n_checked:.1%})")
    print(f"Spec 3 violations: {len(violations['spec3_noise'])} "
          f"({len(violations['spec3_noise'])/n_checked:.1%})")

    return violations, pd.DataFrame(all_results)


# ═══════════════════════════════════════════════════════════════════════════════
# Approach 2: Perturbation-based falsification
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_perturbation(fhr: np.ndarray, params) -> np.ndarray:
    baseline_shift, var_offset, noise_scale = params
    var_scale = 1.0 + var_offset
    fhr_p = fhr.copy()
    valid = fhr_p > 50
    if not np.any(valid):
        return fhr_p
    mean_fhr = np.mean(fhr_p[valid])
    fhr_p[valid] += baseline_shift
    fhr_p[valid] = mean_fhr + (fhr_p[valid] - mean_fhr) * var_scale
    np.random.seed(42)
    fhr_p[valid] += np.random.randn(np.sum(valid)) * noise_scale
    return fhr_p


def perturbation_falsification(
    predict_fn,
    df,
    raw_signals: dict,
    tag: str,
    max_records: int   = 50,
    max_noise_std: float = 20.0,
):
    print("\n" + "=" * 70)
    print(f"APPROACH 2: Perturbation falsification  [{tag}]")
    print("=" * 70)

    results = []
    count   = 0

    for _, row in df[df["label_binary"] >= 0].iterrows():
        if count >= max_records:
            break
        record_id  = int(row["record_id"])
        true_label = int(row["label_binary"])
        ph         = row["pH"]

        if record_id not in raw_signals:
            continue
        fhr = raw_signals[record_id]["fhr"].astype(np.float64)

        orig_label, orig_prob = predict_fn(fhr)
        if orig_label != true_label:
            continue

        count += 1
        print(f"  [{count}/{max_records}] Record {record_id} "
              f"(pH={ph:.2f}, label={true_label}, "
              f"p={orig_prob:.3f}):", end="", flush=True)

        def objective(params):
            fhr_p     = _apply_perturbation(fhr, params)
            new_label, new_prob = predict_fn(fhr_p)
            margin = (new_prob - 0.5 if orig_label == 0
                      else 0.5 - new_prob)
            mag = np.sqrt(params[0]**2 + (params[1]*10)**2 + params[2]**2)
            return margin + 0.01 * mag

        bounds = [
            (-max_noise_std, max_noise_std),
            (-0.5, 0.5),
            (0, max_noise_std),
        ]
        result = differential_evolution(
            objective, bounds, maxiter=100, tol=1e-3, seed=42, workers=1
        )

        fhr_p              = _apply_perturbation(fhr, result.x)
        new_label, new_prob = predict_fn(fhr_p)
        flipped            = new_label != orig_label

        baseline_shift, var_offset, noise_scale = result.x
        mag = np.sqrt(baseline_shift**2 + (var_offset*10)**2 + noise_scale**2)

        results.append({
            "record_id":             record_id,
            "pH":                    ph,
            "true_label":            true_label,
            "orig_prob":             orig_prob,
            "new_prob":              new_prob,
            "flipped":               flipped,
            "baseline_shift":        baseline_shift,
            "var_scale":             1.0 + var_offset,
            "noise_scale":           noise_scale,
            "perturbation_magnitude": mag,
        })

        if flipped:
            print(f" FLIPPED! (Δbase={baseline_shift:+.1f}, "
                  f"var={1+var_offset:.2f}, noise={noise_scale:.1f}, "
                  f"p: {orig_prob:.3f}→{new_prob:.3f})")
        else:
            print(f" robust (mag={mag:.1f})")

    df_out = pd.DataFrame(results)
    if len(df_out) > 0:
        n_flip = int(df_out["flipped"].sum())
        print(f"\n{'─' * 70}")
        print(f"Tested: {len(df_out)}  |  "
              f"Flipped: {n_flip} ({n_flip/len(df_out):.0%})")
        if n_flip > 0:
            mean_mag = df_out.loc[df_out["flipped"],
                                  "perturbation_magnitude"].mean()
            print(f"Mean flip magnitude: {mean_mag:.2f}")
    return df_out


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_violations(violations: dict, results_df: pd.DataFrame, tag: str):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if "spec3_flip_rate" in results_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(results_df["pH"], results_df["spec3_flip_rate"],
                   c=results_df["true_label"], cmap="RdYlGn_r",
                   alpha=0.6, s=40)
        ax.axvline(x=7.15, color="red", linestyle="--", label="pH=7.15")
        ax.set_xlabel("Umbilical cord pH")
        ax.set_ylabel("Noise flip rate (σ=3 bpm)")
        ax.set_title(f"Noise Robustness vs Fetal Outcome [{tag}]")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"noise_robustness_{tag}.png", dpi=150)
        plt.close()

    mono = violations.get("spec2_monotonicity", [])
    if mono:
        fig, ax = plt.subplots(figsize=(8, 5))
        types  = [v["worsening_type"] for v in mono]
        unique = sorted(set(types))
        ax.bar(unique, [types.count(t) for t in unique],
               color=["#e74c3c", "#f39c12", "#3498db"])
        ax.set_xlabel("Worsening Type"); ax.set_ylabel("Violations")
        ax.set_title(f"Monotonicity Violations [{tag}]")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"monotonicity_violations_{tag}.png", dpi=150)
        plt.close()


def plot_perturbation_results(perturb_df: pd.DataFrame, tag: str):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if len(perturb_df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax      = axes[0]
    flipped = perturb_df[perturb_df["flipped"]]
    robust  = perturb_df[~perturb_df["flipped"]]
    if len(flipped):
        ax.scatter(flipped["pH"], flipped["perturbation_magnitude"],
                   c="red", label="Flipped", alpha=0.7, s=50)
    if len(robust):
        ax.scatter(robust["pH"], robust["perturbation_magnitude"],
                   c="green", label="Robust", alpha=0.5, s=30)
    ax.axvline(x=7.15, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Umbilical cord pH")
    ax.set_ylabel("Perturbation magnitude")
    ax.set_title(f"Min Perturbation to Flip [{tag}]")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(perturb_df["orig_prob"], perturb_df["new_prob"],
               c=perturb_df["flipped"].map({True: "red", False: "green"}),
               alpha=0.6, s=40)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Original P(pathological)")
    ax.set_ylabel("Perturbed P(pathological)")
    ax.set_title(f"Probability Shift [{tag}]")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = PLOTS_DIR / f"perturbation_analysis_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved perturbation plot to {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main(clf_name: str, ph_threshold: float):
    suffix = ph_threshold_suffix(ph_threshold)
    tag    = f"{clf_name}_{suffix}"

    print(f"Loading data (pH threshold={ph_threshold})...")
    df = load_all_data(ph_threshold=ph_threshold)

    print("Loading raw signals...")
    raw_signals = load_all_raw_signals()

    print(f"Loading {clf_name.upper()} model ({suffix})...")
    predict_fn = load_predict_fn(clf_name, suffix)

    violations, real_df = search_real_data(
        predict_fn, df, raw_signals, tag
    )
    perturb_df = perturbation_falsification(
        predict_fn, df, raw_signals, tag, max_records=50
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / f"violations_{tag}.pkl", "wb") as f:
        pickle.dump(violations, f)
    real_df.to_csv(
        RESULTS_DIR / f"real_data_results_{tag}.csv", index=False
    )
    perturb_df.to_csv(
        RESULTS_DIR / f"perturbation_results_{tag}.csv", index=False
    )
    print(f"\nSaved results to results/violations_{tag}.pkl and CSVs")

    plot_violations(violations, real_df, tag)
    plot_perturbation_results(perturb_df, tag)

    print(f"\n{'=' * 70}")
    print(f"FALSIFICATION COMPLETE  [{tag}]")
    print(f"{'=' * 70}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Falsify a CTG classifier."
    )
    parser.add_argument(
        "classifier",
        choices=["rf", "cnn", "xgb"],
        help="Which classifier to falsify.",
    )
    parser.add_argument(
        "--ph-threshold",
        type=float,
        default=DEFAULT_PH_THRESHOLD,
        help=f"pH labeling threshold (default: {DEFAULT_PH_THRESHOLD})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(clf_name=args.classifier, ph_threshold=args.ph_threshold)