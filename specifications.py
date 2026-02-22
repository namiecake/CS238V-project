"""
specifications.py — Formalize safety specifications for CTG interpretation.

Implements safety properties inspired by FIGO guidelines, expressed as
predicates over FHR signals and classifier outputs. These are used by
the falsification module to find violations.

Safety Specifications:
1. TACHYCARDIA + LOW VARIABILITY → must classify pathological
2. MONOTONICITY → worsening signal should not improve classification
3. NOISE ROBUSTNESS → small noise should not flip classification

Each specification returns a robustness score:
  - Negative = specification VIOLATED (falsification success!)
  - Positive = specification SATISFIED
  - Magnitude = how far from the boundary
"""

import numpy as np
from typing import Callable

SAMPLING_RATE = 4  # Hz


# ═══════════════════════════════════════════════════════════════════════════════
# Specification 1: Tachycardia + Low Variability → Pathological
# ═══════════════════════════════════════════════════════════════════════════════
#
# FIGO: If FHR baseline >160 bpm (tachycardia) and variability <5 bpm
# (minimal) persists for a sustained period, this is pathological.
#
# Formal: □[t, t+T] (FHR > 160 ∧ variability < 5) → classify(pathological)
#
# Robustness: If the antecedent is satisfied and the classifier says
# "normal," the specification is violated. The robustness score measures
# how strongly the classifier contradicts the clinical indicator.

def spec_tachycardia_low_variability(
    fhr: np.array,
    classifier_predict_fn: Callable,
    duration_minutes: float = 10.0,
    fhr_threshold: float = 160.0,
    var_threshold: float = 5.0,
    fs: int = SAMPLING_RATE,
) -> dict:
    """
    Check Specification 1: Sustained tachycardia + low variability must
    trigger pathological classification.

    Args:
        fhr: Raw FHR signal (possibly perturbed).
        classifier_predict_fn: Function that takes FHR array → (pred_label, pred_prob).
            pred_label: 0=normal, 1=pathological
            pred_prob: probability of pathological class
        duration_minutes: Required duration of the pattern.
        fhr_threshold: FHR threshold for tachycardia (bpm).
        var_threshold: Variability threshold for "minimal" (bpm).
        fs: Sampling rate.

    Returns:
        dict with:
            - "satisfied": bool
            - "robustness": float (negative = violated)
            - "antecedent_met": bool (whether the concerning pattern exists)
            - "details": str
    """
    window_samples = int(duration_minutes * 60 * fs)

    # Check if the concerning pattern exists in the signal
    antecedent_met = False
    max_pattern_strength = -np.inf

    for start in range(0, len(fhr) - window_samples + 1, fs * 60):  # slide by 1 min
        window = fhr[start:start + window_samples]
        valid = window[window > 50]  # exclude artifacts

        if len(valid) < window_samples * 0.5:
            continue

        mean_fhr = np.mean(valid)
        variability = np.std(valid)

        # Pattern strength: how far into concerning territory
        fhr_margin = mean_fhr - fhr_threshold
        var_margin = var_threshold - variability

        if fhr_margin > 0 and var_margin > 0:
            antecedent_met = True
            pattern_strength = min(fhr_margin, var_margin)
            max_pattern_strength = max(max_pattern_strength, pattern_strength)

    if not antecedent_met:
        return {
            "satisfied": True,  # Antecedent not met → spec vacuously true
            "robustness": np.inf,
            "antecedent_met": False,
            "details": "Pattern (tachycardia + low variability) not present in signal."
        }

    # Antecedent IS met → check that classifier says pathological
    pred_label, pred_prob = classifier_predict_fn(fhr)

    if pred_label == 1:  # Correctly classified as pathological
        robustness = pred_prob  # How confident (higher = more robust)
        return {
            "satisfied": True,
            "robustness": robustness,
            "antecedent_met": True,
            "details": f"Correctly classified as pathological (p={pred_prob:.3f})."
        }
    else:
        # VIOLATION: concerning pattern exists but classified as normal
        robustness = pred_prob - 0.5  # Negative when confident in "normal"
        return {
            "satisfied": False,
            "robustness": robustness,
            "antecedent_met": True,
            "details": (
                f"VIOLATION: Tachycardia + low variability present "
                f"(strength={max_pattern_strength:.1f}) but classified as "
                f"normal (p_pathological={pred_prob:.3f})."
            )
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Specification 2: Monotonicity — Worsening FHR should not improve classification
# ═══════════════════════════════════════════════════════════════════════════════
#
# If we make the FHR signal "worse" (increase tachycardia, decrease variability,
# add decelerations), the probability of pathological classification should
# not decrease.
#
# Formal: worsen(FHR) → P(pathological | worsen(FHR)) ≥ P(pathological | FHR)
#
# Robustness = P(path | worsened) - P(path | original). Negative = violated.

def spec_monotonicity(
    fhr: np.array,
    classifier_predict_fn: Callable,
    worsening_type: str = "increase_tachycardia",
    delta: float = 10.0,
    fs: int = SAMPLING_RATE,
) -> dict:
    """
    Check Specification 2: Worsening the signal should not decrease
    the probability of pathological classification.

    Args:
        fhr: Original FHR signal.
        classifier_predict_fn: FHR → (label, prob_pathological).
        worsening_type: How to worsen the signal:
            "increase_tachycardia" — shift baseline up by delta bpm
            "decrease_variability" — smooth signal to reduce variability
            "add_deceleration" — insert a late deceleration pattern
        delta: Magnitude of worsening.

    Returns:
        dict with robustness info.
    """
    # Get original prediction
    _, prob_original = classifier_predict_fn(fhr)

    # Worsen the signal
    fhr_worsened = _apply_worsening(fhr, worsening_type, delta, fs)

    # Get worsened prediction
    _, prob_worsened = classifier_predict_fn(fhr_worsened)

    robustness = prob_worsened - prob_original

    if robustness >= 0:
        return {
            "satisfied": True,
            "robustness": robustness,
            "prob_original": prob_original,
            "prob_worsened": prob_worsened,
            "worsening_type": worsening_type,
            "details": (
                f"Monotonicity maintained: P(path) {prob_original:.3f} → "
                f"{prob_worsened:.3f} after {worsening_type} (Δ={delta})"
            )
        }
    else:
        return {
            "satisfied": False,
            "robustness": robustness,
            "prob_original": prob_original,
            "prob_worsened": prob_worsened,
            "worsening_type": worsening_type,
            "details": (
                f"VIOLATION: P(path) decreased from {prob_original:.3f} → "
                f"{prob_worsened:.3f} after {worsening_type} (Δ={delta})"
            )
        }


def _apply_worsening(fhr, worsening_type, delta, fs):
    """Apply a clinically-meaningful worsening to the FHR signal."""
    fhr_w = fhr.copy()
    valid_mask = fhr_w > 50

    if worsening_type == "increase_tachycardia":
        # Shift FHR upward (more tachycardic)
        fhr_w[valid_mask] += delta

    elif worsening_type == "decrease_variability":
        # Smooth the signal (reduce beat-to-beat variability)
        if np.sum(valid_mask) > 0:
            mean_val = np.mean(fhr_w[valid_mask])
            fhr_w[valid_mask] = fhr_w[valid_mask] * (1 - delta/100) + mean_val * (delta/100)

    elif worsening_type == "add_deceleration":
        # Insert a late deceleration: drop of delta bpm for 60 seconds
        decel_duration = 60 * fs
        # Place it at a random valid position
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > decel_duration:
            start = valid_indices[len(valid_indices) // 2]
            end = min(start + decel_duration, len(fhr_w))
            # Smooth dip shape
            t = np.linspace(0, np.pi, end - start)
            dip = delta * np.sin(t)
            fhr_w[start:end] -= dip

    return fhr_w


# ═══════════════════════════════════════════════════════════════════════════════
# Specification 3: Noise Robustness
# ═══════════════════════════════════════════════════════════════════════════════
#
# Small perturbations (noise) to the FHR signal should not flip the
# classification from pathological to normal. This tests robustness to
# measurement noise.
#
# Formal: |ε| < threshold → classify(FHR + ε) = classify(FHR)

def spec_noise_robustness(
    fhr: np.array,
    classifier_predict_fn: Callable,
    noise_std: float = 2.0,
    n_trials: int = 20,
) -> dict:
    """
    Check Specification 3: Classification should be robust to small noise.

    Tests whether adding Gaussian noise (simulating measurement artifacts)
    causes the classifier to flip its prediction.

    Args:
        fhr: Original FHR signal.
        classifier_predict_fn: FHR → (label, prob).
        noise_std: Standard deviation of Gaussian noise (bpm).
        n_trials: Number of noise realizations to test.

    Returns:
        dict with robustness info.
    """
    orig_label, orig_prob = classifier_predict_fn(fhr)

    flip_count = 0
    prob_changes = []

    for _ in range(n_trials):
        noise = np.random.randn(len(fhr)) * noise_std
        fhr_noisy = fhr.copy()
        valid_mask = fhr > 50
        fhr_noisy[valid_mask] += noise[valid_mask]

        noisy_label, noisy_prob = classifier_predict_fn(fhr_noisy)
        if noisy_label != orig_label:
            flip_count += 1
        prob_changes.append(noisy_prob - orig_prob)

    flip_rate = flip_count / n_trials
    mean_prob_change = np.mean(np.abs(prob_changes))

    # Robustness: 1 - flip_rate (higher is better, negative if >50% flips)
    robustness = 0.5 - flip_rate

    return {
        "satisfied": flip_rate == 0,
        "robustness": robustness,
        "flip_rate": flip_rate,
        "mean_prob_change": mean_prob_change,
        "noise_std": noise_std,
        "n_trials": n_trials,
        "details": (
            f"Noise robustness (σ={noise_std} bpm): "
            f"{flip_count}/{n_trials} flips ({flip_rate:.0%}), "
            f"mean |ΔP|={mean_prob_change:.4f}"
        )
    }


# ─── Helper: Create classifier predict function from model bundle ────────────

def make_predict_fn(model_bundle: dict, feature_extractor: Callable):
    """
    Create a predict function compatible with specifications.

    The specifications need: FHR array → (label, prob_pathological)
    This bridges between raw signals and the feature-based classifier.

    Args:
        model_bundle: dict with "classifier", "scaler", "feature_names"
        feature_extractor: function that takes FHR array → feature dict

    Returns:
        predict_fn(fhr) → (label, prob_pathological)
    """
    clf = model_bundle["classifier"]
    scaler = model_bundle["scaler"]
    feature_names = model_bundle["feature_names"]

    def predict_fn(fhr):
        from load_data import compute_fhr_features, compute_uc_features
        # Extract features from the (possibly perturbed) FHR signal
        fhr_features = feature_extractor(fhr)

        # Build feature vector in correct order
        feature_vec = []
        for fname in feature_names:
            val = fhr_features.get(fname, 0.0)
            if np.isnan(val) if isinstance(val, float) else False:
                val = 0.0
            feature_vec.append(val)

        X = np.array(feature_vec).reshape(1, -1)
        X_scaled = scaler.transform(X)

        label = clf.predict(X_scaled)[0]
        prob = clf.predict_proba(X_scaled)[0, 1]  # P(pathological)
        return int(label), float(prob)

    return predict_fn


if __name__ == "__main__":
    # Quick demo: check specs on a synthetic signal
    print("Specification module loaded successfully.")
    print("Specs available:")
    print("  1. spec_tachycardia_low_variability")
    print("  2. spec_monotonicity")
    print("  3. spec_noise_robustness")
    print("\nRun falsification.py to apply these to the trained classifier.")
