"""
Exp7 — Neural steganalysis detector.

Trains on the current NES detector dataset and evaluates on a
held-out set split by sample_id.

Important:
    Clean/stego members of the same pair must never be split
    across train and test.
"""

import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = "artifacts/exp7_detector_dataset.pkl"

SEED = 42
TEST_RATIO = 0.20

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4

HIDDEN_1 = 256
HIDDEN_2 = 64


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Dataset
# ============================================================

class ResidualDataset(Dataset):

    def __init__(self, samples):

        self.features = torch.stack(
            [
                sample["features"].float()
                for sample in samples
            ]
        )

        self.labels = torch.tensor(
            [
                sample["label"]
                for sample in samples
            ],
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


# ============================================================
# Detector
# ============================================================

class Detector(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_1),
            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Linear(HIDDEN_2, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


# ============================================================
# Split by sample_id
# ============================================================

def split_by_sample_id(samples):

    sample_ids = sorted(
        {
            sample["sample_id"]
            for sample in samples
        }
    )

    random.shuffle(sample_ids)

    split_index = int(
        len(sample_ids) * (1.0 - TEST_RATIO)
    )

    train_ids = set(
        sample_ids[:split_index]
    )

    test_ids = set(
        sample_ids[split_index:]
    )

    train_samples = [
        sample
        for sample in samples
        if sample["sample_id"] in train_ids
    ]

    test_samples = [
        sample
        for sample in samples
        if sample["sample_id"] in test_ids
    ]

    return train_samples, test_samples


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device):

    model.eval()

    correct = 0
    total = 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for features, labels in loader:

        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)

        predictions = (
            torch.sigmoid(logits) >= 0.5
        ).float()

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.numel()

        tp += (
            (predictions == 1)
            & (labels == 1)
        ).sum().item()

        tn += (
            (predictions == 0)
            & (labels == 0)
        ).sum().item()

        fp += (
            (predictions == 1)
            & (labels == 0)
        ).sum().item()

        fn += (
            (predictions == 0)
            & (labels == 1)
        ).sum().item()

    accuracy = correct / max(total, 1)

    return {
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("EXP7 — NEURAL STEGANALYSIS DETECTOR")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(f"Loading: {DATASET_PATH}")

    with open(DATASET_PATH, "rb") as f:
        samples = pickle.load(f)

    print(f"Total samples: {len(samples)}")

    # --------------------------------------------------------
    # Verify labels
    # --------------------------------------------------------

    clean_count = sum(
        sample["label"] == 0
        for sample in samples
    )

    stego_count = sum(
        sample["label"] == 1
        for sample in samples
    )

    print(f"Clean: {clean_count}")
    print(f"Stego: {stego_count}")

    # --------------------------------------------------------
    # Split by sample ID
    # --------------------------------------------------------

    train_samples, test_samples = split_by_sample_id(
        samples
    )

    train_ids = {
        sample["sample_id"]
        for sample in train_samples
    }

    test_ids = {
        sample["sample_id"]
        for sample in test_samples
    }

    assert train_ids.isdisjoint(test_ids)

    print()
    print(f"Training samples: {len(train_samples)}")
    print(f"Testing samples:  {len(test_samples)}")
    print(f"Training pairs:   {len(train_ids)}")
    print(f"Testing pairs:    {len(test_ids)}")

    # --------------------------------------------------------
    # Dataset/DataLoader
    # --------------------------------------------------------

    train_dataset = ResidualDataset(
        train_samples
    )

    test_dataset = ResidualDataset(
        test_samples
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    input_dim = train_dataset.features.shape[1]

    model = Detector(
        input_dim=input_dim
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("Training detector...")

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for features, labels in train_loader:

            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(features)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = (
            total_loss / len(train_loader)
        )

        if (
            epoch == 0
            or (epoch + 1) % 5 == 0
            or epoch == EPOCHS - 1
        ):
            metrics = evaluate(
                model,
                test_loader,
                device,
            )

            print(
                f"Epoch {epoch + 1:02d}/{EPOCHS} "
                f"loss={avg_loss:.6f} "
                f"test_acc={metrics['accuracy']:.4f}"
            )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    metrics = evaluate(
        model,
        test_loader,
        device,
    )

    accuracy = metrics["accuracy"]

    print()
    print("=" * 70)
    print("EXP7 FINAL RESULT")
    print("=" * 70)

    print(f"Test accuracy: {accuracy * 100:.2f}%")

    print()
    print("Confusion matrix:")
    print(f"TN: {metrics['tn']}")
    print(f"FP: {metrics['fp']}")
    print(f"FN: {metrics['fn']}")
    print(f"TP: {metrics['tp']}")

    print()

    if accuracy <= 0.55:
        print("Detector gate: PASS")
        print("Accuracy <= 55%")
    else:
        print("Detector gate: FAIL")
        print("Accuracy > 55%")

    print("=" * 70)


if __name__ == "__main__":
    main()