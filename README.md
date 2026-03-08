# Validating Automated Fetal Heart Rate Interpretation Systems

AA228V / CS238V Final Project — Stanford University

## Overview

This project applies specification-based falsification to validate automated cardiotocography (CTG) classification systems. We train two classifiers — a Random Forest on hand-crafted features and a 1D CNN on raw FHR signals — formalize three safety specifications grounded in FIGO clinical guidelines, and use optimization-based falsification to find specification violations.

The ground truth outcome is derived from umbilical cord blood pH at delivery, a clinically accepted proxy for fetal acidemia. We run the full pipeline at two binary classification thresholds (pH < 7.15 and pH < 7.05) to study how class imbalance affects both classifier behavior and specification violations.

## Project Structure

```
├── load_data.py          # Data loading from PhysioNet, feature extraction, labeling
├── rf_classifier.py      # Random Forest training with 5-fold stratified CV
├── cnn_classifier.py     # 1D CNN training with 5-fold stratified CV
├── specifications.py     # 3 safety specifications with robustness scores
├── falsification.py      # Real-data search + perturbation-based falsification
├── run_all.py            # Runs the full pipeline for both pH thresholds
├── compare_results.py    # Generates comparison table and combined ROC curves
├── requirements.txt      # Pinned dependencies
├── data/                 # Cached features and raw signals
├── models/               # Saved model artifacts
├── results/              # Output CSVs, pickles, and summary table
└── plots/                # Generated figures
```

## Setup

Requires Python 3.10+. All computation runs on a laptop CPU (no GPU needed).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dataset (CTU-UHB Intrapartum Cardiotocography Database, 552 recordings) is streamed from PhysioNet on first run and cached locally. The initial download takes ~15–20 minutes; subsequent runs load from cache.

## How to Run

### Option A: Run everything

```bash
python run_all.py
```

This runs the full pipeline for both pH thresholds (7.15 and 7.05) sequentially: data loading, RF training, CNN training, falsification (RF and CNN), and the final comparison.

### Option B: Run step by step

```bash
# 1. Load data and cache raw signals
python load_data.py --ph-threshold 7.15

# 2. Train Random Forest
python rf_classifier.py --ph-threshold 7.15

# 3. Train 1D CNN
python cnn_classifier.py --ph-threshold 7.15

# 4. Run falsification
python falsification.py rf  --ph-threshold 7.15
python falsification.py cnn --ph-threshold 7.15

# Repeat steps 1–4 with --ph-threshold 7.05

# 5. Generate comparison summary and combined ROC plot
python compare_results.py
```

Each threshold produces separate cache files, models, and results (e.g., `ctg_features_ph715.pkl` vs `ctg_features_ph705.pkl`), so nothing is overwritten.

For long runs, use unbuffered output so you can monitor progress:
```bash
caffeinate -s env PYTHONUNBUFFERED=1 python run_all.py 2>&1 | tee run_all.log
```

## Dataset

[CTU-UHB Intrapartum Cardiotocography Database](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/) — 552 intrapartum CTG recordings at 4 Hz with clinical outcome data. After filtering for records with valid pH, 506 recordings are used.

- **pH < 7.15 threshold:** 417 normal, 89 pathological (17.6%)
- **pH < 7.05 threshold:** 481 normal, 25 pathological (4.9%)

## Classifiers

| Classifier | Input | Architecture |
|---|---|---|
| **Random Forest** | 16 hand-crafted features (baseline FHR, variability, accelerations, decelerations, signal quality, UC features) | 200 trees, max depth 10, balanced class weights |
| **1D CNN** | Raw FHR signal (last 30 min, 7200 samples at 4 Hz) | 3 conv blocks (16→32→64 filters, kernel size 15) → global avg pool → FC layers. Weighted sampling + weighted cross-entropy for class imbalance. |

Both are evaluated with 5-fold stratified cross-validation.

## Safety Specifications

All three specifications are inspired by FIGO clinical guidelines and formalized as predicates over FHR signals and classifier outputs. Each returns a robustness score (negative = violated).

| # | Specification | Clinical Basis | Formalization |
|---|---|---|---|
| 1 | **Tachycardia + low variability → pathological** | Sustained tachycardia (>160 bpm) with minimal variability (<5 bpm) for ≥10 min indicates fetal distress | □[t, t+10min] (mean FHR > 160 ∧ std FHR < 5) → classify(pathological) |
| 2 | **Monotonicity** | Worsening FHR patterns should not decrease the assessed risk | worsen(FHR) → P(path \| worsened) ≥ P(path \| original) |
| 3 | **Noise robustness** | Small measurement noise should not flip the classification | \|ε\| < σ → classify(FHR + ε) = classify(FHR) |

Spec 2 tests three worsening modes: increasing tachycardia (+15 bpm), decreasing variability (15% smoothing toward mean), and inserting a late deceleration (15 bpm sinusoidal dip).

## Falsification Approach

1. **Real-data search:** Check all 506 records against all three specifications to find naturally occurring violations.
2. **Perturbation-based falsification:** For each correctly-classified record, use `scipy.optimize.differential_evolution` to find the minimal perturbation (parameterized as baseline shift + variability scaling + additive noise) that flips the classification. This characterizes the robustness boundary.

## Results

### Classification Performance

| Classifier | pH Threshold | AUC | Sensitivity | Specificity |
|---|---|---|---|---|
| RF | < 7.15 | 0.662 | 0.079 | 0.988 |
| CNN | < 7.15 | 0.674 | 0.910 | 0.225 |
| RF | < 7.05 | 0.734 | 0.000 | 0.998 |
| CNN | < 7.05 | 0.744 | 0.640 | 0.663 |

The RF and CNN exhibit opposite failure modes: the RF almost never predicts pathological (near-zero sensitivity), while the CNN over-predicts (low specificity).

### Specification Violations (real-data search, 506 records)

| Classifier | pH Threshold | Spec 1 (tachycardia) | Spec 2 (monotonicity) | Spec 3 (noise) |
|---|---|---|---|---|
| RF | < 7.15 | 4.9% | **63.2%** | 2.2% |
| CNN | < 7.15 | 2.0% | 10.5% | **37.4%** |
| RF | < 7.05 | 5.1% | **55.7%** | 0.2% |
| CNN | < 7.05 | 2.6% | 15.0% | **33.4%** |

The RF massively violates monotonicity (worsening the signal *decreases* its pathological confidence for >60% of records). The CNN is far more susceptible to noise (small perturbations flip predictions for ~35% of records).

### Perturbation Falsification (50 correctly-classified records per variant)

| Classifier | pH Threshold | Flip Rate | Mean Flip Magnitude |
|---|---|---|---|
| RF | < 7.15 | 18% | 4.75 bpm |
| CNN | < 7.15 | 58% | 10.03 bpm |
| RF | < 7.05 | 0% | — |
| CNN | < 7.05 | 88% | 5.36 bpm |

## Key Findings

1. **RF and CNN fail different specifications.** The feature-based RF violates monotonicity severely but is noise-robust. The CNN satisfies monotonicity better but is noise-fragile. Neither satisfies all three specifications.
2. **Monotonicity is the most frequently violated specification.** For the RF, worsening the FHR signal actually decreases pathological confidence in >60% of cases — a fundamental safety concern.
3. **Class imbalance degrades safety properties.** At the stricter pH < 7.05 threshold (only 4.9% pathological), the RF's sensitivity drops to zero and the CNN's perturbation flip rate rises to 88%.
4. **AUC alone is misleading.** Both classifiers have similar AUC (~0.66–0.74), but their operating points and safety profiles are radically different.

## Output Files

After a full run, key outputs include:

- `results/comparison_summary.csv` — headline comparison table
- `plots/combined_roc_curves.png` — ROC curves for all 4 variants
- `results/perturbation_results_{rf,cnn}_ph{715,705}.csv` — per-record perturbation details
- `results/violations_{rf,cnn}_ph{715,705}.pkl` — full violation records
- `plots/perturbation_analysis_*.png` — perturbation magnitude vs pH scatter plots
- `plots/monotonicity_violations_*.png` — violation counts by worsening type
- `plots/violation_*.png` — example FHR tracings for violated records
