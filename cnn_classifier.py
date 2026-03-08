"""
cnn_classifier.py — 1D CNN classifier for raw CTG signals.

Unlike the feature-based classifier in rf_classifier.py, this operates
directly on the raw FHR time series.  This lets us compare:
  - Feature-based (RF): interpretable, uses domain knowledge
  - CNN: learns patterns directly from the signal, black-box

Usage:
    python cnn_classifier.py                        # default pH < 7.15
    python cnn_classifier.py --ph-threshold 7.05
"""

import argparse
import pickle

import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load_data import (
    load_all_data,
    load_all_raw_signals,
    DEFAULT_PH_THRESHOLD,
    ph_threshold_suffix,
    SAMPLING_RATE,
)

# ─── Constants ───────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"
RESULTS_DIR = BASE_DIR / "results"

TARGET_LENGTH = SAMPLING_RATE * 60 * 30  # 30 min at 4 Hz = 7200 samples


# ─── Signal Preprocessing (shared by Dataset and Predictor) ─────────────────

def clean_signal(fhr: np.ndarray) -> np.ndarray:
    """Replace artifact values (<=50 bpm or NaN) with linear interpolation."""
    fhr = fhr.astype(float)
    invalid = (fhr <= 50) | np.isnan(fhr)

    if np.all(invalid):
        return np.full_like(fhr, 140.0)

    valid_idx = np.where(~invalid)[0]
    if len(valid_idx) > 1:
        fhr[invalid] = np.interp(
            np.where(invalid)[0], valid_idx, fhr[valid_idx]
        )
    elif len(valid_idx) == 1:
        fhr[invalid] = fhr[valid_idx[0]]

    return fhr


def preprocess_signal(fhr: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline: clean → truncate/pad → normalize.

    This is the single source of truth used by both CTGDataset and
    CNNPredictor to guarantee identical preprocessing.
    """
    fhr = clean_signal(fhr.copy())

    # Take last 30 min (closest to delivery) or pad at the start
    if len(fhr) >= TARGET_LENGTH:
        fhr = fhr[-TARGET_LENGTH:]
    else:
        pad_length = TARGET_LENGTH - len(fhr)
        mean_val = np.mean(fhr[fhr > 50]) if np.any(fhr > 50) else 140.0
        fhr = np.concatenate([np.full(pad_length, mean_val), fhr])

    # Normalize to zero mean, unit variance over valid samples
    valid = fhr[fhr > 50]
    if len(valid) > 0:
        mean = np.mean(valid)
        std = np.std(valid) + 1e-6
        fhr = (fhr - mean) / std
    else:
        fhr = np.zeros(TARGET_LENGTH)

    return fhr


# ─── Dataset ─────────────────────────────────────────────────────────────────

class CTGDataset(Dataset):
    """PyTorch dataset for raw CTG signals."""

    def __init__(self, signals: list, labels: np.ndarray):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        fhr = preprocess_signal(self.signals[idx])
        signal_tensor = torch.FloatTensor(fhr).unsqueeze(0)  # (1, 7200)
        label_tensor = torch.LongTensor([self.labels[idx]]).squeeze()
        return signal_tensor, label_tensor


# ─── Model Architecture ─────────────────────────────────────────────────────

class CTG_CNN(nn.Module):
    """1D CNN for CTG classification.

    Input: (batch, 1, 7200) — 30 min of FHR at 4 Hz
    Architecture progressively downsamples to capture patterns from
    beat-to-beat variability (~seconds) to baseline shifts (~minutes).
    """

    def __init__(self, input_length: int = TARGET_LENGTH):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: short-term patterns (beat-to-beat variability)
            nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 7200 → 1800
            # Block 2: medium-term (accelerations / decelerations)
            nn.Conv1d(16, 32, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),  # 1800 → 450
            # Block 3: long-term (baseline shifts)
            nn.Conv1d(32, 64, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # → (batch, 64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ─── Training Loop ───────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * signals.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for signals, labels in loader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = outputs.max(1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ─── Cross-Validation Pipeline ──────────────────────────────────────────────

def load_raw_signal_data(
    df: pd.DataFrame,
    raw_signals: dict | None = None,
) -> tuple:
    """Gather raw FHR signals and labels for records with valid labels.

    If *raw_signals* (from ``load_all_raw_signals()``) is provided, the
    signals are read from that dict instead of re-streaming from PhysioNet.
    """
    if raw_signals is None:
        raw_signals = load_all_raw_signals()

    signals, labels, record_ids = [], [], []
    for _, row in df[df["label_binary"] >= 0].iterrows():
        rid = int(row["record_id"])
        if rid not in raw_signals:
            continue
        signals.append(raw_signals[rid]["fhr"].astype(np.float64))
        labels.append(int(row["label_binary"]))
        record_ids.append(rid)

    return signals, np.array(labels), np.array(record_ids)


def train_cnn_cv(
    signals,
    labels,
    record_ids,
    n_splits: int = 5,
    n_epochs: int = 25,
    lr: float = 1e-3,
    batch_size: int = 16,
):
    """Train CNN with stratified cross-validation.

    Returns cross-validated predictions (each sample predicted once, when
    it was in the held-out test fold) and the per-fold models.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    n = len(labels)
    all_y_true = np.zeros(n)
    all_y_pred = np.zeros(n)
    all_y_prob = np.zeros(n)
    fold_models = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n), labels)):
        print(f"\n  Fold {fold + 1}/{n_splits}")

        train_signals = [signals[i] for i in train_idx]
        test_signals = [signals[i] for i in test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        train_dataset = CTGDataset(train_signals, train_labels)
        test_dataset = CTGDataset(test_signals, test_labels)

        # Weighted sampling to address class imbalance
        class_counts = np.bincount(train_labels, minlength=2)
        if class_counts[1] == 0:
            sample_weights = np.ones(len(train_labels))
        else:
            sample_weights = 1.0 / class_counts[train_labels]
        sampler = WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        model = CTG_CNN().to(device)

        # Loss also weighted to upweight the minority (pathological) class
        if class_counts[1] > 0:
            loss_weight = torch.FloatTensor(
                [1.0, class_counts[0] / class_counts[1]]
            ).to(device)
        else:
            loss_weight = torch.FloatTensor([1.0, 1.0]).to(device)
        criterion = nn.CrossEntropyLoss(weight=loss_weight)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        best_loss = np.inf
        best_model_state = None
        for epoch in range(n_epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            scheduler.step(train_loss)

            if train_loss < best_loss:
                best_loss = train_loss
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }

            if (epoch + 1) % 10 == 0:
                print(
                    f"    Epoch {epoch+1}: loss={train_loss:.4f}, acc={train_acc:.3f}"
                )

        model.load_state_dict(best_model_state)
        y_true, y_pred, y_prob = evaluate(model, test_loader, device)

        all_y_true[test_idx] = y_true
        all_y_pred[test_idx] = y_pred
        all_y_prob[test_idx] = y_prob
        fold_models.append(model.cpu())

    return all_y_true, all_y_pred, all_y_prob, fold_models


# ─── Prediction Function for Falsification ──────────────────────────────────

class CNNPredictor:
    """Wraps a trained CNN for the ``predict_fn(fhr) → (label, prob)``
    interface required by specifications and falsification.

    Uses the same ``preprocess_signal`` function as CTGDataset to
    guarantee identical preprocessing.
    """

    def __init__(self, model, device=None):
        self.model = model.eval()
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

    def __call__(self, fhr: np.ndarray) -> tuple:
        fhr_proc = preprocess_signal(fhr)
        x = torch.FloatTensor(fhr_proc).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(x)
            prob = torch.softmax(output, dim=1)[0, 1].item()
            label = 1 if prob > 0.5 else 0
        return label, prob


# ─── Main ────────────────────────────────────────────────────────────────────

def main(ph_threshold: float = DEFAULT_PH_THRESHOLD):
    suffix = ph_threshold_suffix(ph_threshold)

    print(f"Loading data (pH threshold = {ph_threshold})...")
    df = load_all_data(ph_threshold=ph_threshold)

    print("Loading raw signals for CNN...")
    raw_signals = load_all_raw_signals()
    signals, labels, record_ids = load_raw_signal_data(df, raw_signals)
    print(f"Loaded {len(signals)} recordings")
    print(f"Class balance: {np.sum(labels == 0)} normal, {np.sum(labels == 1)} pathological")

    print("\n" + "=" * 60)
    print("1D CNN — 5-fold Stratified Cross-Validation")
    print("=" * 60)

    y_true, y_pred, y_prob, fold_models = train_cnn_cv(
        signals, labels, record_ids,
        n_splits=5, n_epochs=25, lr=1e-3, batch_size=16,
    )

    # ── Metrics ──
    print("\n" + "=" * 60)
    print("CNN RESULTS")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=["Normal", "Pathological"]))

    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"AUC-ROC:     {auc:.3f}")
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"Confusion Matrix:\n{cm}")

    # ── Save ──
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cnn_bundle = {
        "model": fold_models[0],
        "all_fold_models": fold_models,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "record_ids": record_ids,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ph_threshold": ph_threshold,
    }
    model_path = MODEL_DIR / f"cnn_classifier_{suffix}.pt"
    torch.save(cnn_bundle, model_path)
    print(f"\nSaved CNN to {model_path}")

    # ── Plot ROC ──
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"CNN (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — 1D CNN CTG Classifier")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    roc_path = PLOTS_DIR / f"roc_curve_cnn_{suffix}.png"
    fig.savefig(roc_path, dpi=150)
    plt.close()
    print(f"Saved ROC curve to {roc_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train 1D CNN CTG classifier.")
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
