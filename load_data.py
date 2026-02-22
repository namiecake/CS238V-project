"""
load_data.py — Download and preprocess the CTU-UHB CTG database.

This module handles:
1. Downloading records from PhysioNet (streaming, no bulk download needed)
2. Parsing header files for clinical metadata (pH, Apgar, etc.)
3. Extracting FHR and UC signals
4. Computing features for classification
5. Labeling based on umbilical cord blood pH

Usage:
    python load_data.py          # Downloads data and saves processed features
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path

# ─── Constants ───────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
PN_DIR = "ctu-uhb-ctgdb/1.0.0"  # PhysioNet directory for streaming
SAMPLING_RATE = 4  # Hz
RECORD_IDS = list(range(1001, 1507))  # Records 1001-1506 (not all exist)

# pH thresholds for classification (standard clinical cutoffs)
PH_PATHOLOGICAL = 7.05   # Below this = acidemia / pathological
PH_SUSPICIOUS = 7.15     # Between 7.05-7.15 = suspicious (pre-acidemia)
PH_NORMAL = 7.20         # Above 7.20 = normal


# ─── Header Parsing ─────────────────────────────────────────────────────────

def parse_header_comments(record_name: str, pn_dir: str = PN_DIR) -> dict:
    """Parse clinical metadata from a CTU-UHB header file's comments."""
    try:
        header = wfdb.rdheader(record_name, pn_dir=pn_dir)
    except Exception as e:
        print(f"  Could not read header for {record_name}: {e}")
        return {}

    metadata = {"record_id": record_name}
    for comment in header.comments:
        comment = comment.strip()
        # Match patterns like "pH           7.19" or "Apgar1       9"
        match = re.match(r"^([\w\.\s]+?)\s{2,}(.+)$", comment)
        if match:
            key = match.group(1).strip()
            val = match.group(2).strip()
            # Try to convert to numeric
            try:
                val = float(val)
                if val == int(val):
                    val = int(val)
            except ValueError:
                pass
            metadata[key] = val
    return metadata


# ─── Signal Loading ──────────────────────────────────────────────────────────

def load_record(record_name: str, pn_dir: str = PN_DIR):
    """
    Load a single CTG record from PhysioNet.

    Returns:
        fhr: np.array — Fetal heart rate signal (bpm), 4 Hz
        uc: np.array — Uterine contraction signal, 4 Hz
        metadata: dict — Clinical parameters from header
    """
    try:
        record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
    except Exception as e:
        return None, None, {}

    fhr = record.p_signal[:, 0]  # Channel 0 = FHR
    uc = record.p_signal[:, 1]   # Channel 1 = UC

    metadata = parse_header_comments(record_name, pn_dir)
    return fhr, uc, metadata


# ─── Feature Extraction ─────────────────────────────────────────────────────

def compute_fhr_features(fhr: np.array, fs: int = SAMPLING_RATE) -> dict:
    """
    Compute clinically-relevant FHR features from a recording.

    Features are based on FIGO guidelines for CTG interpretation:
    - Baseline FHR (mean of the signal excluding artifacts)
    - Variability (std dev of short-term fluctuations)
    - Accelerations (count of episodes where FHR rises >15bpm for >15s)
    - Decelerations (count of episodes where FHR drops >15bpm for >15s)
    - Signal quality (fraction of valid samples)
    - Summary statistics
    """
    features = {}

    # ── Handle missing data (FHR=0 means no signal) ──
    valid_mask = fhr > 50  # FHR below 50 is artifact/missing
    fhr_clean = fhr.copy()
    fhr_clean[~valid_mask] = np.nan

    features["signal_quality"] = np.mean(valid_mask)
    if features["signal_quality"] < 0.1:
        # Too little valid signal — return NaN features
        for key in ["baseline_fhr", "variability_stv", "variability_ltv",
                     "mean_fhr", "std_fhr", "min_fhr", "max_fhr",
                     "n_accelerations", "n_decelerations",
                     "pct_tachycardia", "pct_bradycardia",
                     "pct_low_variability"]:
            features[key] = np.nan
        return features

    valid_fhr = fhr_clean[valid_mask]

    # ── Baseline estimation (windowed median, 10-min windows) ──
    window_size = 10 * 60 * fs  # 10 minutes in samples
    baselines = []
    for i in range(0, len(fhr_clean), window_size):
        chunk = fhr_clean[i:i + window_size]
        valid_chunk = chunk[~np.isnan(chunk)]
        if len(valid_chunk) > 0:
            baselines.append(np.median(valid_chunk))
    features["baseline_fhr"] = np.mean(baselines) if baselines else np.nan

    # ── Variability ──
    # Short-term variability (STV): mean absolute difference between successive samples
    diffs = np.abs(np.diff(valid_fhr))
    features["variability_stv"] = np.mean(diffs)

    # Long-term variability (LTV): std of 1-minute means
    minute_samples = 60 * fs
    minute_means = []
    for i in range(0, len(fhr_clean), minute_samples):
        chunk = fhr_clean[i:i + minute_samples]
        valid_chunk = chunk[~np.isnan(chunk)]
        if len(valid_chunk) > minute_samples // 2:
            minute_means.append(np.mean(valid_chunk))
    features["variability_ltv"] = np.std(minute_means) if len(minute_means) > 1 else np.nan

    # ── Basic statistics ──
    features["mean_fhr"] = np.nanmean(fhr_clean)
    features["std_fhr"] = np.nanstd(fhr_clean)
    features["min_fhr"] = np.nanmin(valid_fhr)
    features["max_fhr"] = np.nanmax(valid_fhr)

    # ── Accelerations & Decelerations ──
    # Acceleration: FHR > baseline + 15 bpm for >= 15 seconds
    # Deceleration: FHR < baseline - 15 bpm for >= 15 seconds
    baseline = features["baseline_fhr"]
    if not np.isnan(baseline):
        min_duration = 15 * fs  # 15 seconds in samples

        accel_mask = fhr_clean > (baseline + 15)
        decel_mask = fhr_clean < (baseline - 15)

        features["n_accelerations"] = _count_episodes(accel_mask, min_duration)
        features["n_decelerations"] = _count_episodes(decel_mask, min_duration)
    else:
        features["n_accelerations"] = np.nan
        features["n_decelerations"] = np.nan

    # ── Clinical thresholds (fraction of time) ──
    features["pct_tachycardia"] = np.nanmean(fhr_clean > 160)   # Tachycardia >160
    features["pct_bradycardia"] = np.nanmean(fhr_clean < 110)   # Bradycardia <110
    features["pct_low_variability"] = _pct_low_variability(fhr_clean, fs)

    return features


def _count_episodes(mask: np.array, min_samples: int) -> int:
    """Count contiguous True episodes in a boolean mask of at least min_samples length."""
    mask = np.asarray(mask, dtype=bool)
    # Replace NaN positions with False
    mask[np.isnan(mask)] = False
    count = 0
    run_length = 0
    for val in mask:
        if val:
            run_length += 1
        else:
            if run_length >= min_samples:
                count += 1
            run_length = 0
    if run_length >= min_samples:
        count += 1
    return count


def _pct_low_variability(fhr_clean: np.array, fs: int, window_min: int = 1) -> float:
    """
    Compute fraction of 1-minute windows with variability < 5 bpm.
    Low variability (<5 bpm bandwidth per minute) is a FIGO concern.
    """
    window_samples = window_min * 60 * fs
    low_var_count = 0
    total_windows = 0
    for i in range(0, len(fhr_clean), window_samples):
        chunk = fhr_clean[i:i + window_samples]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) > window_samples // 2:
            total_windows += 1
            bandwidth = np.max(valid) - np.min(valid)
            if bandwidth < 5:
                low_var_count += 1
    return low_var_count / total_windows if total_windows > 0 else np.nan


# ─── UC Feature Extraction ──────────────────────────────────────────────────

def compute_uc_features(uc: np.array, fs: int = SAMPLING_RATE) -> dict:
    """Compute basic uterine contraction features."""
    features = {}
    valid_mask = ~np.isnan(uc) & (uc != 0)
    if np.sum(valid_mask) < fs * 60:  # Less than 1 minute of data
        features["uc_mean"] = np.nan
        features["uc_std"] = np.nan
        features["uc_max"] = np.nan
        return features

    uc_valid = uc[valid_mask]
    features["uc_mean"] = np.mean(uc_valid)
    features["uc_std"] = np.std(uc_valid)
    features["uc_max"] = np.max(uc_valid)
    return features


# ─── Labeling ────────────────────────────────────────────────────────────────

def ph_to_label(ph: float) -> int:
    """
    Convert pH to a binary label for classification.
      0 = Normal (pH >= 7.15)
      1 = Pathological (pH < 7.15)

    We use 7.15 as the cutoff for binary classification because:
    - pH < 7.05 is clearly pathological but very rare (~5% of dataset)
    - pH < 7.15 captures "pre-acidemic" cases that warrant concern
    - This gives a more balanced dataset for training
    """
    if np.isnan(ph):
        return -1  # Unknown
    return 1 if ph < 7.15 else 0


def ph_to_three_class(ph: float) -> int:
    """Three-class label: 0=Normal, 1=Suspicious, 2=Pathological."""
    if np.isnan(ph):
        return -1
    if ph >= PH_NORMAL:
        return 0
    elif ph >= PH_PATHOLOGICAL:
        return 1
    else:
        return 2


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def load_all_data(max_records: int = None, cache: bool = True) -> pd.DataFrame:
    """
    Load all CTU-UHB records, extract features, and return a DataFrame.

    Args:
        max_records: If set, only load this many records (for debugging).
        cache: If True, save/load from pickle cache.

    Returns:
        DataFrame with features, labels, and metadata for each recording.
    """
    cache_path = DATA_DIR / "ctg_features.pkl"

    if cache and cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        return pd.read_pickle(cache_path)

    print("Downloading and processing CTU-UHB database from PhysioNet...")
    print("(This streams records individually — no bulk download needed)")
    print()

    all_rows = []
    loaded = 0

    for record_id in RECORD_IDS:
        if max_records and loaded >= max_records:
            break

        record_name = str(record_id)
        print(f"  Loading record {record_name}...", end=" ")

        fhr, uc, metadata = load_record(record_name)
        if fhr is None:
            print("SKIP (not found)")
            continue

        # Extract pH for labeling
        ph = metadata.get("pH", np.nan)
        if isinstance(ph, str):
            try:
                ph = float(ph)
            except ValueError:
                ph = np.nan

        # Compute features
        fhr_features = compute_fhr_features(fhr)
        uc_features = compute_uc_features(uc)

        # Combine everything
        row = {
            "record_id": record_id,
            "pH": ph,
            "label_binary": ph_to_label(ph),
            "label_3class": ph_to_three_class(ph),
            **metadata,
            **fhr_features,
            **uc_features,
        }
        all_rows.append(row)
        loaded += 1
        print(f"OK (pH={ph:.2f}, FHR baseline={fhr_features['baseline_fhr']:.1f})")

    df = pd.DataFrame(all_rows)
    print(f"\nLoaded {len(df)} records total.")
    print(f"Label distribution (binary): {dict(df['label_binary'].value_counts())}")

    # Cache
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache_path)
    print(f"Saved to {cache_path}")

    return df


def load_raw_signals(record_id: int) -> tuple:
    """Load raw FHR and UC signals for a single record (for plotting/analysis)."""
    fhr, uc, metadata = load_record(str(record_id))
    return fhr, uc, metadata


if __name__ == "__main__":
    df = load_all_data()
    print("\n── Dataset Summary ──")
    print(f"Records: {len(df)}")
    print(f"pH range: {df['pH'].min():.2f} - {df['pH'].max():.2f}")
    print(f"\nBinary labels:")
    print(f"  Normal (pH >= 7.15):      {(df['label_binary'] == 0).sum()}")
    print(f"  Pathological (pH < 7.15): {(df['label_binary'] == 1).sum()}")
    print(f"\nFeature columns: {[c for c in df.columns if c not in ['record_id', 'pH', 'label_binary', 'label_3class']]}")
