"""
lstm_classifier.py — Fast CNN+GRU classifier for CTG signals.

Designed to train in ~5 minutes on CPU:
  - Signal downsampled 4 Hz → 1 Hz, last 10 min only (600 samples)
  - CNN compresses 600 → 10 timesteps
  - Single GRU layer, hidden_size=16
  - 10 epochs, batch_size=64

Still called lstm_classifier.py so the rest of the pipeline
(falsification.py, compare_results.py, run_all.py) needs no changes.
The saved class is CTG_LSTM for the same reason.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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

BASE_DIR    = Path(__file__).parent
MODEL_DIR   = BASE_DIR / "models"
PLOTS_DIR   = BASE_DIR / "plots"
RESULTS_DIR = BASE_DIR / "results"

# Use last 10 min at 1 Hz — 600 samples instead of 7200
TARGET_SECONDS = 10 * 60        # 600 s
TARGET_LENGTH  = TARGET_SECONDS  # at 1 Hz


# ─── Signal Preprocessing ────────────────────────────────────────────────────

def clean_signal(fhr: np.ndarray) -> np.ndarray:
    fhr = fhr.astype(float)
    invalid = (fhr <= 50) | np.isnan(fhr)
    if np.all(invalid):
        return np.full_like(fhr, 140.0)
    valid_idx = np.where(~invalid)[0]
    if len(valid_idx) > 1:
        fhr[invalid] = np.interp(np.where(invalid)[0], valid_idx, fhr[valid_idx])
    elif len(valid_idx) == 1:
        fhr[invalid] = fhr[valid_idx[0]]
    return fhr


def preprocess_signal(fhr: np.ndarray) -> np.ndarray:
    """Clean → downsample 4→1 Hz → take last 10 min → normalize."""
    fhr = clean_signal(fhr.copy())

    # Downsample 4 Hz → 1 Hz by taking every 4th sample
    fhr = fhr[::SAMPLING_RATE]   # now 1 Hz

    # Take last 10 minutes (600 samples), pad at start if shorter
    if len(fhr) >= TARGET_LENGTH:
        fhr = fhr[-TARGET_LENGTH:]
    else:
        pad_len  = TARGET_LENGTH - len(fhr)
        mean_val = np.mean(fhr[fhr > 50]) if np.any(fhr > 50) else 140.0
        fhr = np.concatenate([np.full(pad_len, mean_val), fhr])

    # Normalize
    valid = fhr[fhr > 50]
    if len(valid) > 0:
        fhr = (fhr - np.mean(valid)) / (np.std(valid) + 1e-6)
    else:
        fhr = np.zeros(TARGET_LENGTH)

    return fhr.astype(np.float32)


# ─── Dataset ─────────────────────────────────────────────────────────────────

class CTGDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels  = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        fhr = preprocess_signal(self.signals[idx])
        return (
            torch.FloatTensor(fhr).unsqueeze(0),           # (1, 600)
            torch.LongTensor([self.labels[idx]]).squeeze(),
        )


# ─── Model ───────────────────────────────────────────────────────────────────

class CTG_LSTM(nn.Module):
    """Tiny CNN + 1-layer GRU — trains in ~1 min/fold on CPU.

    CNN:  (1, 600) → (16, 10)   [stride-based compression ×60]
    GRU:  10 steps, hidden=16
    Head: Linear(16, 2)

    Named CTG_LSTM to stay compatible with falsification.py / compare_results.py.
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # 600 → 60  (stride 10)
            nn.Conv1d(1,  16, kernel_size=11, stride=10, padding=5),
            nn.BatchNorm1d(16), nn.ReLU(),
            # 60 → 10   (stride 6)
            nn.Conv1d(16, 16, kernel_size=7,  stride=6,  padding=3),
            nn.BatchNorm1d(16), nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=16, hidden_size=16, num_layers=1, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        enc = self.encoder(x)           # (batch, 16, ~10)
        seq = enc.permute(0, 2, 1)      # (batch, ~10, 16)
        _, h_n = self.gru(seq)          # h_n: (1, batch, 16)
        return self.classifier(h_n[-1])


# ─── Training helpers ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(signals)
        loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * signals.size(0)
        correct    += out.max(1)[1].eq(labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            out     = model(signals)
            probs   = torch.softmax(out, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(out.max(1)[1].cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ─── Data loading ────────────────────────────────────────────────────────────

def load_raw_signal_data(df: pd.DataFrame, raw_signals: dict | None = None):
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


# ─── Cross-validation ────────────────────────────────────────────────────────

def train_lstm_cv(
    signals,
    labels,
    record_ids,
    n_splits:   int   = 5,
    n_epochs:   int   = 10,
    lr:         float = 1e-3,
    batch_size: int   = 64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n  = len(labels)
    all_y_true = np.zeros(n)
    all_y_pred = np.zeros(n)
    all_y_prob = np.zeros(n)
    fold_models = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n), labels)):
        print(f"\n  Fold {fold + 1}/{n_splits}")

        train_signals = [signals[i] for i in train_idx]
        test_signals  = [signals[i] for i in test_idx]
        train_labels  = labels[train_idx]
        test_labels   = labels[test_idx]

        train_ds = CTGDataset(train_signals, train_labels)
        test_ds  = CTGDataset(test_signals,  test_labels)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

        model = CTG_LSTM().to(device)

        # sqrt-dampened loss weight — avoids the double-correction collapse
        class_counts = np.bincount(train_labels, minlength=2)
        pos_weight   = float(np.sqrt(class_counts[0] / max(class_counts[1], 1)))
        criterion    = nn.CrossEntropyLoss(
            weight=torch.FloatTensor([1.0, pos_weight]).to(device)
        )
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-5
        )

        best_loss  = np.inf
        best_state = None
        for epoch in range(n_epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            scheduler.step()
            if train_loss < best_loss:
                best_loss  = train_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 5 == 0:
                print(f"    Epoch {epoch+1:2d}: loss={train_loss:.4f}, acc={train_acc:.3f}")

        model.load_state_dict(best_state)
        y_true, y_pred, y_prob = evaluate(model, test_loader, device)
        all_y_true[test_idx] = y_true
        all_y_pred[test_idx] = y_pred
        all_y_prob[test_idx] = y_prob
        fold_models.append(model.cpu())

    return all_y_true, all_y_pred, all_y_prob, fold_models


# ─── Predictor (used by falsification.py) ───────────────────────────────────

class LSTMPredictor:
    """predict_fn(fhr) → (label, prob) interface for falsification."""

    def __init__(self, model, device=None):
        self.model  = model.eval()
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

    def __call__(self, fhr: np.ndarray) -> tuple:
        x = (torch.FloatTensor(preprocess_signal(fhr))
             .unsqueeze(0).unsqueeze(0).to(self.device))
        with torch.no_grad():
            prob  = float(torch.softmax(self.model(x), dim=1)[0, 1].item())
            label = 1 if prob > 0.5 else 0
        return label, prob


# ─── Main ────────────────────────────────────────────────────────────────────

def main(ph_threshold: float = DEFAULT_PH_THRESHOLD):
    suffix = ph_threshold_suffix(ph_threshold)

    print(f"Loading data (pH threshold = {ph_threshold})...")
    df = load_all_data(ph_threshold=ph_threshold)

    print("Loading raw signals...")
    raw_signals = load_all_raw_signals()
    signals, labels, record_ids = load_raw_signal_data(df, raw_signals)
    print(f"Loaded {len(signals)} recordings")
    print(f"Class balance: {np.sum(labels==0)} normal, "
          f"{np.sum(labels==1)} pathological")

    print("\n" + "=" * 60)
    print("Compact CNN+GRU — 5-fold Stratified CV")
    print("=" * 60)

    y_true, y_pred, y_prob, fold_models = train_lstm_cv(
        signals, labels, record_ids,
        n_splits=5, n_epochs=10, lr=1e-3, batch_size=64,
    )

    print("\n" + "=" * 60)
    print("GRU RESULTS")
    print("=" * 60)
    print(classification_report(
        y_true, y_pred,
        target_names=["Normal", "Pathological"],
        zero_division=0,
    ))

    auc = roc_auc_score(y_true, y_prob)
    cm  = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"AUC-ROC:     {auc:.3f}")
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"Confusion Matrix:\n{cm}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model":           fold_models[0],
        "all_fold_models": fold_models,
        "y_true":          y_true,
        "y_pred":          y_pred,
        "y_prob":          y_prob,
        "record_ids":      record_ids,
        "auc":             auc,
        "sensitivity":     sensitivity,
        "specificity":     specificity,
        "ph_threshold":    ph_threshold,
    }, MODEL_DIR / f"lstm_classifier_{suffix}.pt")
    print(f"\nSaved to models/lstm_classifier_{suffix}.pt")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, "g-", linewidth=2, label=f"GRU (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Compact CNN+GRU CTG Classifier")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"roc_curve_lstm_{suffix}.png", dpi=150)
    plt.close()
    print(f"Saved ROC curve to plots/roc_curve_lstm_{suffix}.png")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ph-threshold", type=float, default=DEFAULT_PH_THRESHOLD)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(ph_threshold=args.ph_threshold)