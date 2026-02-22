"""
falsification.py — Optimization-based falsification of CTG classifiers.

This is the core validation module. It searches for inputs that cause
the CTG classifier to violate safety specifications.

Two approaches:
1. Real-data search: Check all records for specification violations.
2. Perturbation-based: Use CMA-ES to find minimal perturbations to
   correctly-classified samples that cause violations.

Usage:
    python falsification.py
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt

from load_data import (
    load_all_data, load_raw_signals, compute_fhr_features, compute_uc_features,
    SAMPLING_RATE
)
from specifications import (
    spec_tachycardia_low_variability,
    spec_monotonicity,
    spec_noise_robustness,
    make_predict_fn,
)

MODEL_DIR = Path(__file__).parent / "models"
RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = Path(__file__).parent / "plots"


# ─── Load trained model ─────────────────────────────────────────────────────

def load_model():
    """Load the trained classifier bundle."""
    with open(MODEL_DIR / "ctg_classifier.pkl", "rb") as f:
        model_bundle = pickle.load(f)
    return model_bundle


def make_full_predict_fn(model_bundle):
    """Create a predict function that extracts both FHR and UC features."""
    clf = model_bundle["classifier"]
    scaler = model_bundle["scaler"]
    feature_names = model_bundle["feature_names"]

    def predict_fn(fhr, uc=None):
        fhr_feats = compute_fhr_features(fhr)
        uc_feats = compute_uc_features(uc) if uc is not None else {
            "uc_mean": 0, "uc_std": 0, "uc_max": 0
        }
        all_feats = {**fhr_feats, **uc_feats}

        feature_vec = []
        for fname in feature_names:
            val = all_feats.get(fname, 0.0)
            if isinstance(val, float) and np.isnan(val):
                val = 0.0
            feature_vec.append(val)

        X = np.array(feature_vec).reshape(1, -1)
        X_scaled = scaler.transform(X)
        label = clf.predict(X_scaled)[0]
        prob = clf.predict_proba(X_scaled)[0, 1]
        return int(label), float(prob)

    # Also create a version that only takes FHR (for specs that only use FHR)
    def predict_fn_fhr_only(fhr):
        return predict_fn(fhr, uc=None)

    return predict_fn, predict_fn_fhr_only


# ═══════════════════════════════════════════════════════════════════════════════
# Approach 1: Search over real data for specification violations
# ═══════════════════════════════════════════════════════════════════════════════

def search_real_data(model_bundle, df):
    """
    Check every record in the dataset for specification violations.

    This is the simplest form of falsification — just check whether the
    existing data contains any counterexamples to our specifications.
    """
    print("\n" + "=" * 70)
    print("APPROACH 1: Searching real data for specification violations")
    print("=" * 70)

    predict_fn, predict_fn_fhr = make_full_predict_fn(model_bundle)
    violations = {
        "spec1_tachycardia": [],
        "spec2_monotonicity": [],
        "spec3_noise": [],
    }
    all_results = []

    records = df[df["label_binary"] >= 0]

    for _, row in records.iterrows():
        record_id = int(row["record_id"])
        fhr, uc, metadata = load_raw_signals(record_id)
        if fhr is None:
            continue

        ph = row["pH"]
        true_label = row["label_binary"]

        print(f"\n  Record {record_id} (pH={ph:.2f}, true={true_label}):", end="")

        # ── Spec 1: Tachycardia + low variability ──
        result1 = spec_tachycardia_low_variability(fhr, predict_fn_fhr)
        if not result1["satisfied"]:
            violations["spec1_tachycardia"].append({
                "record_id": record_id, "pH": ph, **result1
            })
            print(f" SPEC1-VIOLATED", end="")

        # ── Spec 2: Monotonicity (test all worsening types) ──
        for wtype in ["increase_tachycardia", "decrease_variability", "add_deceleration"]:
            result2 = spec_monotonicity(fhr, predict_fn_fhr, worsening_type=wtype, delta=15.0)
            if not result2["satisfied"]:
                violations["spec2_monotonicity"].append({
                    "record_id": record_id, "pH": ph,
                    "worsening_type": wtype, **result2
                })
                print(f" SPEC2-VIOLATED({wtype})", end="")

        # ── Spec 3: Noise robustness ──
        result3 = spec_noise_robustness(fhr, predict_fn_fhr, noise_std=3.0, n_trials=20)
        if not result3["satisfied"]:
            violations["spec3_noise"].append({
                "record_id": record_id, "pH": ph, **result3
            })
            print(f" SPEC3-VIOLATED(flip_rate={result3['flip_rate']:.0%})", end="")

        all_results.append({
            "record_id": record_id,
            "pH": ph,
            "true_label": true_label,
            "spec1_satisfied": result1["satisfied"],
            "spec1_robustness": result1["robustness"],
            "spec2_satisfied_tachy": True,  # Will be overwritten if violated
            "spec3_satisfied": result3["satisfied"],
            "spec3_flip_rate": result3["flip_rate"],
        })
        print(" ✓" if (result1["satisfied"] and result3["satisfied"]) else "")

    # Summary
    print(f"\n{'─' * 70}")
    print("REAL DATA SEARCH SUMMARY")
    print(f"{'─' * 70}")
    print(f"Records checked: {len(records)}")
    print(f"Spec 1 (tachycardia+low var) violations: {len(violations['spec1_tachycardia'])}")
    print(f"Spec 2 (monotonicity) violations:        {len(violations['spec2_monotonicity'])}")
    print(f"Spec 3 (noise robustness) violations:    {len(violations['spec3_noise'])}")

    return violations, pd.DataFrame(all_results)


# ═══════════════════════════════════════════════════════════════════════════════
# Approach 2: Perturbation-based falsification (optimization)
# ═══════════════════════════════════════════════════════════════════════════════

def perturbation_falsification(
    model_bundle, df,
    max_records: int = 50,
    max_noise_std: float = 20.0,
):
    """
    Use optimization to find minimal perturbations that cause spec violations.

    For each correctly-classified record, search for the smallest perturbation
    that flips the classification. This characterizes the classifier's
    robustness boundary.

    The perturbation is parameterized as:
      - baseline_shift: constant offset to FHR (bpm)
      - variability_scale: multiplicative scaling of variability
      - noise_std: additive Gaussian noise level

    We minimize perturbation magnitude subject to causing a classification flip.
    """
    print("\n" + "=" * 70)
    print("APPROACH 2: Perturbation-based falsification")
    print("=" * 70)

    predict_fn, predict_fn_fhr = make_full_predict_fn(model_bundle)
    falsification_results = []

    records = df[df["label_binary"] >= 0]
    count = 0

    for _, row in records.iterrows():
        if count >= max_records:
            break

        record_id = int(row["record_id"])
        true_label = row["label_binary"]
        ph = row["pH"]

        fhr, uc, _ = load_raw_signals(record_id)
        if fhr is None:
            continue

        # Get original prediction
        orig_label, orig_prob = predict_fn_fhr(fhr)

        # Only try to flip correctly-classified samples
        if orig_label != true_label:
            continue

        count += 1
        print(f"\n  Record {record_id} (pH={ph:.2f}, pred_prob={orig_prob:.3f}):", end="")

        # ── Objective: minimize perturbation magnitude while flipping classification ──
        def objective(params):
            """
            params = [baseline_shift, var_scale_offset, noise_seed_scale]
            Returns negative robustness (we want to minimize = find violations).
            """
            baseline_shift = params[0]
            var_scale = 1.0 + params[1]  # 1.0 = no change
            noise_scale = abs(params[2])

            # Apply perturbation
            fhr_perturbed = fhr.copy()
            valid = fhr_perturbed > 50
            mean_fhr = np.mean(fhr_perturbed[valid]) if np.any(valid) else 140

            # Shift baseline
            fhr_perturbed[valid] += baseline_shift

            # Scale variability around mean
            fhr_perturbed[valid] = mean_fhr + (fhr_perturbed[valid] - mean_fhr) * var_scale

            # Add structured noise
            np.random.seed(42)  # Deterministic for optimization
            fhr_perturbed[valid] += np.random.randn(np.sum(valid)) * noise_scale

            try:
                _, prob = predict_fn_fhr(fhr_perturbed)
            except Exception:
                return 100.0

            # We want to flip: if originally normal (prob < 0.5), maximize prob
            # If originally pathological (prob > 0.5), minimize prob
            if orig_label == 0:
                flip_score = 0.5 - prob  # negative when prob > 0.5 (flipped!)
            else:
                flip_score = prob - 0.5  # negative when prob < 0.5 (flipped!)

            # Regularize by perturbation magnitude
            perturbation_mag = np.sqrt(
                baseline_shift**2 + (params[1] * 10)**2 + noise_scale**2
            )
            return flip_score + 0.01 * perturbation_mag

        # Search bounds
        bounds = [
            (-30, 30),       # baseline_shift (bpm)
            (-0.5, 0.5),     # var_scale_offset
            (0, max_noise_std),  # noise_scale
        ]

        result = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            maxiter=100,
            tol=1e-6,
            polish=True,
        )

        # Check if we found a flip
        best_params = result.x
        baseline_shift, var_offset, noise_scale = best_params
        var_scale = 1.0 + var_offset

        # Apply best perturbation
        fhr_perturbed = fhr.copy()
        valid = fhr_perturbed > 50
        mean_fhr = np.mean(fhr_perturbed[valid]) if np.any(valid) else 140
        fhr_perturbed[valid] += baseline_shift
        fhr_perturbed[valid] = mean_fhr + (fhr_perturbed[valid] - mean_fhr) * var_scale
        np.random.seed(42)
        fhr_perturbed[valid] += np.random.randn(np.sum(valid)) * noise_scale

        new_label, new_prob = predict_fn_fhr(fhr_perturbed)
        flipped = new_label != orig_label

        perturbation_magnitude = np.sqrt(
            baseline_shift**2 + (var_offset * 10)**2 + noise_scale**2
        )

        falsification_results.append({
            "record_id": record_id,
            "pH": ph,
            "true_label": true_label,
            "orig_prob": orig_prob,
            "new_prob": new_prob,
            "flipped": flipped,
            "baseline_shift": baseline_shift,
            "var_scale": var_scale,
            "noise_scale": noise_scale,
            "perturbation_magnitude": perturbation_magnitude,
        })

        if flipped:
            print(f" FLIPPED! (Δbaseline={baseline_shift:+.1f}, "
                  f"var_scale={var_scale:.2f}, noise={noise_scale:.1f}, "
                  f"prob: {orig_prob:.3f}→{new_prob:.3f})")
        else:
            print(f" robust (min perturbation tested: {perturbation_magnitude:.1f})")

    results_df = pd.DataFrame(falsification_results)

    # Summary
    print(f"\n{'─' * 70}")
    print("PERTURBATION FALSIFICATION SUMMARY")
    print(f"{'─' * 70}")
    n_flipped = results_df["flipped"].sum()
    print(f"Records tested: {len(results_df)}")
    print(f"Successfully flipped: {n_flipped} ({n_flipped/len(results_df):.0%})")
    if n_flipped > 0:
        flipped_df = results_df[results_df["flipped"]]
        print(f"Mean perturbation magnitude for flips: {flipped_df['perturbation_magnitude'].mean():.2f}")
        print(f"Min perturbation magnitude for flip: {flipped_df['perturbation_magnitude'].min():.2f}")

    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis & Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_violations(violations: dict, results_df: pd.DataFrame):
    """Plot analysis of specification violations."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Noise robustness vs pH ──
    if "spec3_flip_rate" in results_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(results_df["pH"], results_df["spec3_flip_rate"],
                   c=results_df["true_label"], cmap="RdYlGn_r", alpha=0.6, s=40)
        ax.axvline(x=7.15, color="red", linestyle="--", label="pH=7.15 threshold")
        ax.set_xlabel("Umbilical cord pH", fontsize=12)
        ax.set_ylabel("Noise flip rate (σ=3 bpm)", fontsize=12)
        ax.set_title("Noise Robustness vs Fetal Outcome", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "noise_robustness_vs_ph.png", dpi=150)
        plt.close()

    # ── Monotonicity violations distribution ──
    mono_violations = violations.get("spec2_monotonicity", [])
    if mono_violations:
        fig, ax = plt.subplots(figsize=(8, 5))
        types = [v["worsening_type"] for v in mono_violations]
        unique_types = list(set(types))
        counts = [types.count(t) for t in unique_types]
        ax.bar(unique_types, counts, color=["#e74c3c", "#f39c12", "#3498db"])
        ax.set_xlabel("Worsening Type", fontsize=12)
        ax.set_ylabel("Number of Violations", fontsize=12)
        ax.set_title("Monotonicity Violations by Worsening Type", fontsize=14)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "monotonicity_violations.png", dpi=150)
        plt.close()


def plot_perturbation_results(perturb_df: pd.DataFrame):
    """Plot perturbation falsification results."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if len(perturb_df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Perturbation magnitude vs pH ──
    ax = axes[0]
    flipped = perturb_df[perturb_df["flipped"]]
    robust = perturb_df[~perturb_df["flipped"]]
    if len(flipped) > 0:
        ax.scatter(flipped["pH"], flipped["perturbation_magnitude"],
                   c="red", label="Flipped", alpha=0.7, s=50)
    if len(robust) > 0:
        ax.scatter(robust["pH"], robust["perturbation_magnitude"],
                   c="green", label="Robust", alpha=0.5, s=30)
    ax.axvline(x=7.15, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Umbilical cord pH", fontsize=12)
    ax.set_ylabel("Perturbation magnitude", fontsize=12)
    ax.set_title("Minimum Perturbation to Flip Classification", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Probability shift ──
    ax = axes[1]
    ax.scatter(perturb_df["orig_prob"], perturb_df["new_prob"],
               c=perturb_df["flipped"].map({True: "red", False: "green"}),
               alpha=0.6, s=40)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Original P(pathological)", fontsize=12)
    ax.set_ylabel("Perturbed P(pathological)", fontsize=12)
    ax.set_title("Classification Probability: Original vs Perturbed", fontsize=13)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "perturbation_analysis.png", dpi=150)
    plt.close()
    print(f"Saved perturbation analysis to {PLOTS_DIR / 'perturbation_analysis.png'}")


def plot_example_violation(record_id: int, fhr: np.array, violation_details: str):
    """Plot a specific FHR tracing that caused a violation."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    time_minutes = np.arange(len(fhr)) / SAMPLING_RATE / 60
    fig, ax = plt.subplots(figsize=(12, 4))

    valid = fhr > 50
    fhr_plot = fhr.copy()
    fhr_plot[~valid] = np.nan

    ax.plot(time_minutes, fhr_plot, "b-", linewidth=0.5, alpha=0.8)
    ax.axhline(y=160, color="red", linestyle="--", alpha=0.5, label="Tachycardia (160)")
    ax.axhline(y=110, color="orange", linestyle="--", alpha=0.5, label="Bradycardia (110)")
    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("FHR (bpm)", fontsize=12)
    ax.set_title(f"Record {record_id} — {violation_details}", fontsize=12)
    ax.set_ylim(60, 220)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"violation_record_{record_id}.png", dpi=150)
    plt.close()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # Load model and data
    print("Loading model and data...")
    model_bundle = load_model()
    df = load_all_data()

    print(f"Model: {model_bundle['best_name']}")
    print(f"Records: {len(df)}")

    # ── Approach 1: Real data search ──
    violations, real_results_df = search_real_data(model_bundle, df)

    # ── Approach 2: Perturbation-based falsification ──
    perturb_df = perturbation_falsification(model_bundle, df, max_records=50)

    # ── Save all results ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "violations.pkl", "wb") as f:
        pickle.dump(violations, f)
    real_results_df.to_csv(RESULTS_DIR / "real_data_results.csv", index=False)
    perturb_df.to_csv(RESULTS_DIR / "perturbation_results.csv", index=False)

    # ── Plots ──
    plot_violations(violations, real_results_df)
    plot_perturbation_results(perturb_df)

    # Plot example violations
    for spec_name, spec_violations in violations.items():
        for v in spec_violations[:3]:  # Plot up to 3 examples per spec
            rid = v["record_id"]
            fhr, _, _ = load_raw_signals(rid)
            if fhr is not None:
                plot_example_violation(rid, fhr, v.get("details", spec_name))

    print(f"\n{'=' * 70}")
    print("FALSIFICATION COMPLETE")
    print(f"Results saved to {RESULTS_DIR}/")
    print(f"Plots saved to {PLOTS_DIR}/")
    print(f"{'=' * 70}")

    return violations, real_results_df, perturb_df


if __name__ == "__main__":
    main()
