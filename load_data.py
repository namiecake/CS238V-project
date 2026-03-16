"""
load_data.py — Download and preprocess the CTU-UHB CTG database.

This module handles:
1. Downloading records from PhysioNet (streaming, no bulk download needed)
2. Parsing header files for clinical metadata (pH, Apgar, etc.)
3. Extracting FHR and UC signals
4. Computing features for classification
5. Labeling based on umbilical cord blood pH

Usage:
    python load_data.py                      # default pH < 7.15
    python load_data.py --ph-threshold 7.05
"""

import argparse
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
PH_SUSPICIOUS   = 7.15   # Between 7.05–7.15 = suspicious (pre-acidemia)
PH_NORMAL       = 7.20   # Above 7.20 = normal

# Default threshold used when no --ph-threshold flag is supplied
DEFAULT_PH_THRESHOLD = PH_SUSPICIOUS  # 7.15


def ph_threshold_suffix(threshold: float) -> str:
    """Return a filesystem-safe suffix for a given pH threshold.

    Examples:
        7.15  →  "ph715"
        7.05  →  "ph705"
    """
    return "ph" + str(threshold).replace(".", "")


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
        match = re.match(r"^([\w\.\s]+?)\s{2,}(.+)$", comment)
        if match:
            key = match.group(1).strip()
            val = match.group(2).strip()
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
    """Load a single CTG record from PhysioNet.

    Returns:
        fhr: np.ndarray — Fetal heart rate signal (bpm), 4 Hz
        uc:  np.ndarray — Uterine contraction signal, 4 Hz
        metadata: dict  — Clinical parameters from header
    """
    try:
        record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
    except Exception:
        return None, None, {}

    fhr = record.p_signal[:, 0]
    uc  = record.p_signal[:, 1]
    metadata = parse_header_comments(record_name, pn_dir)
    return fhr, uc, metadata


# ─── Feature Extraction ─────────────────────────────────────────────────────

def compute_fhr_features(fhr: np.ndarray, fs: int = SAMPLING_RATE) -> dict:
    """Compute clinically-relevant FHR features (FIGO guidelines)."""
    features = {}

    valid_mask = fhr > 50
    fhr_clean = fhr.copy().astype(float)
    fhr_clean[~valid_mask] = np.nan

    features["signal_quality"] = np.mean(valid_mask)
    if features["signal_quality"] < 0.1:
        for key in [
            "baseline_fhr", "variability_stv", "variability_ltv",
            "mean_fhr", "std_fhr", "min_fhr", "max_fhr",
            "n_accelerations", "n_decelerations",
            "pct_tachycardia", "pct_bradycardia", "pct_low_variability",
        ]:
            features[key] = np.nan
        return features

    valid_fhr = fhr_clean[valid_mask]

    # Baseline (windowed median, 10-min windows)
    window_size = 10 * 60 * fs
    baselines = []
    for i in range(0, len(fhr_clean), window_size):
        chunk = fhr_clean[i:i + window_size]
        vc = chunk[~np.isnan(chunk)]
        if len(vc) > 0:
            baselines.append(np.median(vc))
    features["baseline_fhr"] = np.mean(baselines) if baselines else np.nan

    # Variability
    features["variability_stv"] = np.mean(np.abs(np.diff(valid_fhr)))
    minute_samples = 60 * fs
    minute_means = []
    for i in range(0, len(fhr_clean), minute_samples):
        chunk = fhr_clean[i:i + minute_samples]
        vc = chunk[~np.isnan(chunk)]
        if len(vc) > minute_samples // 2:
            minute_means.append(np.mean(vc))
    features["variability_ltv"] = np.std(minute_means) if len(minute_means) > 1 else np.nan

    # Basic statistics
    features["mean_fhr"] = np.nanmean(fhr_clean)
    features["std_fhr"]  = np.nanstd(fhr_clean)
    features["min_fhr"]  = np.nanmin(valid_fhr)
    features["max_fhr"]  = np.nanmax(valid_fhr)

    # Accelerations & decelerations
    baseline = features["baseline_fhr"]
    if not np.isnan(baseline):
        min_dur = 15 * fs
        features["n_accelerations"] = _count_episodes(fhr_clean > (baseline + 15), min_dur)
        features["n_decelerations"]  = _count_episodes(fhr_clean < (baseline - 15), min_dur)
    else:
        features["n_accelerations"] = np.nan
        features["n_decelerations"]  = np.nan

    features["pct_tachycardia"]     = np.nanmean(fhr_clean > 160)
    features["pct_bradycardia"]     = np.nanmean(fhr_clean < 110)
    features["pct_low_variability"] = _pct_low_variability(fhr_clean, fs)

    return features


def _count_episodes(mask: np.ndarray, min_samples: int) -> int:
    mask = np.asarray(mask, dtype=bool)
    mask[np.isnan(mask)] = False
    count = run = 0
    for val in mask:
        if val:
            run += 1
        else:
            if run >= min_samples:
                count += 1
            run = 0
    if run >= min_samples:
        count += 1
    return count


def _pct_low_variability(fhr_clean: np.ndarray, fs: int, window_min: int = 1) -> float:
    window_samples = window_min * 60 * fs
    low_var = total = 0
    for i in range(0, len(fhr_clean), window_samples):
        chunk = fhr_clean[i:i + window_samples]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) > window_samples // 2:
            total += 1
            if (np.max(valid) - np.min(valid)) < 5:
                low_var += 1
    return low_var / total if total > 0 else np.nan


# ─── UC Features ─────────────────────────────────────────────────────────────

def compute_uc_features(uc: np.ndarray, fs: int = SAMPLING_RATE) -> dict:
    """Compute basic uterine contraction features."""
    features = {}
    valid_mask = ~np.isnan(uc) & (uc != 0)
    if np.sum(valid_mask) < fs * 60:
        return {"uc_mean": np.nan, "uc_std": np.nan, "uc_max": np.nan}
    uc_valid = uc[valid_mask]
    features["uc_mean"] = np.mean(uc_valid)
    features["uc_std"]  = np.std(uc_valid)
    features["uc_max"]  = np.max(uc_valid)
    return features


# ─── Labeling ────────────────────────────────────────────────────────────────

def ph_to_label(ph: float, threshold: float = DEFAULT_PH_THRESHOLD) -> int:
    """Binary label: 1 = pathological (pH < threshold), 0 = normal, -1 = unknown."""
    if np.isnan(ph):
        return -1
    return 1 if ph < threshold else 0


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


# ─── Main Data Pipeline ──────────────────────────────────────────────────────

def load_all_data(
    ph_threshold: float = DEFAULT_PH_THRESHOLD,
    max_records: int = None,
    cache: bool = True,
) -> pd.DataFrame:
    """Load all CTU-UHB records, extract features, and return a DataFrame.

    Uses per-threshold cache files (e.g. ctg_features_ph715.pkl) so both
    thresholds can coexist without overwriting each other.  If only the
    legacy single-file cache exists it is re-labeled and promoted.

    Args:
        ph_threshold: pH cutoff for binary labeling (default: 7.15).
        max_records:  Cap on records loaded (for quick debugging).
        cache:        Whether to read/write the pickle cache.
    """
    suffix = ph_threshold_suffix(ph_threshold)
    cache_path = DATA_DIR / f"ctg_features_{suffix}.pkl"
    legacy_cache = DATA_DIR / "ctg_features.pkl"

    # Promote legacy cache to per-threshold cache on first run
    if cache and not cache_path.exists() and legacy_cache.exists():
        print(f"Loading legacy cache from {legacy_cache} "
              f"(re-labeling for pH<{ph_threshold})")
        df = pd.read_pickle(legacy_cache)
        df["label_binary"] = df["pH"].apply(
            lambda ph: ph_to_label(ph, ph_threshold)
        )
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path)
        return df

    if cache and cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        return pd.read_pickle(cache_path)

    print("Downloading and processing CTU-UHB database from PhysioNet...")
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

        ph = metadata.get("pH", np.nan)
        if isinstance(ph, str):
            try:
                ph = float(ph)
            except ValueError:
                ph = np.nan

        fhr_features = compute_fhr_features(fhr)
        uc_features  = compute_uc_features(uc)

        row = {
            "record_id":    record_id,
            "pH":           ph,
            "label_binary": ph_to_label(ph, ph_threshold),
            "label_3class": ph_to_three_class(ph),
            **metadata,
            **fhr_features,
            **uc_features,
        }
        all_rows.append(row)
        loaded += 1
        print(f"OK (pH={ph:.2f}, baseline={fhr_features['baseline_fhr']:.1f})")

    df = pd.DataFrame(all_rows)
    print(f"\nLoaded {len(df)} records.")
    print(f"Labels: {dict(df['label_binary'].value_counts())}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache_path)
    print(f"Saved to {cache_path}")
    return df


def load_raw_signals(record_id: int) -> tuple:
    """Load raw FHR and UC signals for a single record."""
    fhr, uc, metadata = load_record(str(record_id))
    return fhr, uc, metadata


def load_all_raw_signals(cache: bool = True) -> dict:
    """Load and cache raw FHR/UC signals for every record in the dataset.

    Returns a dict:  record_id (int) → {"fhr": np.ndarray, "uc": np.ndarray}

    The cache is shared across pH thresholds because raw signals are
    threshold-independent.
    """
    cache_path = DATA_DIR / "raw_signals.pkl"

    if cache and cache_path.exists():
        print(f"Loading cached raw signals from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("Loading raw signals from PhysioNet (this may take a while)...")
    raw_signals: dict = {}

    for record_id in RECORD_IDS:
        fhr, uc, _ = load_record(str(record_id))
        if fhr is None:
            continue
        raw_signals[record_id] = {
            "fhr": fhr.astype(np.float32),
            "uc":  uc.astype(np.float32),
        }
        print(f"  Loaded {record_id} ({len(fhr)} samples)")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(raw_signals, f)
    print(f"Saved raw signals to {cache_path} ({len(raw_signals)} records)")
    return raw_signals


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and cache CTG data.")
    parser.add_argument(
        "--ph-threshold",
        type=float,
        default=DEFAULT_PH_THRESHOLD,
        help=f"pH threshold for binary labeling (default: {DEFAULT_PH_THRESHOLD})",
    )
    args = parser.parse_args()

    df = load_all_data(ph_threshold=args.ph_threshold)
    load_all_raw_signals()  # pre-cache raw signals while we're here

    print("\n── Dataset Summary ──")
    print(f"Records:   {len(df)}")
    print(f"pH range:  {df['pH'].min():.2f} – {df['pH'].max():.2f}")
    print(f"\nBinary labels (pH < {args.ph_threshold}):")
    print(f"  Normal:       {(df['label_binary'] == 0).sum()}")
    print(f"  Pathological: {(df['label_binary'] == 1).sum()}")