# Validating Automated Fetal Heart Rate Interpretation Systems

AA228V/CS238V Final Project — Stanford University

## Overview

This project applies falsification techniques to validate automated cardiotocography (CTG) classification systems. We train a baseline classifier on the CTU-UHB dataset, formalize safety specifications based on FIGO clinical guidelines, and use optimization-based falsification to identify failure modes.

## Project Structure

```
ctg_validation/
├── load_data.py          # Data loading, preprocessing, feature extraction
├── train_classifier.py   # Train & evaluate baseline CTG classifiers
├── specifications.py     # Formalized safety specifications (STL-inspired)
├── falsification.py      # Falsification: real-data search + perturbation-based
├── data/                 # Cached processed data
├── models/               # Trained model artifacts
├── results/              # Falsification results (CSV, pickle)
└── plots/                # Generated figures
```

## Setup

```bash
pip install wfdb scikit-learn numpy pandas matplotlib scipy
```

No extra compute needed — everything runs on a laptop. The CTU-UHB dataset (552 records) is streamed from PhysioNet on demand.

## How to Run

### Step 1: Load data and extract features
```bash
python load_data.py
```
Downloads all 552 CTG records from PhysioNet (streaming), extracts clinically-relevant features (baseline FHR, variability, accelerations, decelerations, etc.), and labels based on umbilical cord blood pH. Results are cached to `data/ctg_features.pkl`.

### Step 2: Train the classifier
```bash
python train_classifier.py
```
Trains Random Forest, Gradient Boosting, and SVM classifiers using 5-fold stratified cross-validation. Saves the best model and generates ROC curves and feature importance plots.

### Step 3: Run falsification
```bash
python falsification.py
```
Runs both falsification approaches:
1. **Real-data search**: checks every record against all 3 safety specifications
2. **Perturbation-based**: uses `differential_evolution` to find minimal perturbations that flip classifications

Generates analysis plots and saves all results.

## Safety Specifications

| # | Specification | FIGO Basis | Formalization |
|---|---------------|------------|---------------|
| 1 | Tachycardia + low variability → pathological | Sustained tachycardia (>160bpm) with minimal variability (<5bpm) indicates distress | □[t,t+T](FHR>160 ∧ var<5) → classify(pathological) |
| 2 | Monotonicity | Worsening signals should not reduce risk assessment | worsen(FHR) → P(path\|worsened) ≥ P(path\|original) |
| 3 | Noise robustness | System should tolerate measurement noise | \|ε\|<σ → classify(FHR+ε) = classify(FHR) |

## Dataset

[CTU-UHB Intrapartum Cardiotocography Database](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/) — 552 CTG recordings with fetal outcome data (umbilical cord blood pH).
