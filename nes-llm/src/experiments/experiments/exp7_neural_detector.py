"""
Experiment 7 (neural) — carrier-centered neural steganalysis stress test.

Kept separate from the statistical security gate because it answers a
different question and can fail when the statistical gate passes.

What this experiment actually measures: a small MLP trained on paired
patches cut from the *same* layers, regions and positions of the same
model. That is a deliberately favourable setting for the adversary — the
only difference between a clean and a stego patch is the embedding — so
this is a stress test of the embedding, not a general claim about
adversarial detectability.

What it does not establish: that a detector trained this way is a
universal adversary against other models, carriers or payload sizes. The
dataset reuses a single model's residual structure heavily.

The threshold is fixed at 55% accuracy. It is never relaxed to convert a
FAIL into a PASS (§16.2, §25 rule 4).
"""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

from src.experiments.artifact_manager import utc_now
from src.experiments.experiment_registry import gate_for
from src.experiments.model_context import ModelContext
from src.experiments.paths import ARTIFACTS_DIR, RESULTS_DIR

EXPERIMENT = "exp7_neural"

DATASET_FILENAME = "exp7_detector_dataset.pkl"

NEURAL_THRESHOLD = 0.55


def dataset_path() -> Path:
    return ARTIFACTS_DIR / DATASET_FILENAME


def existing_result(model_id: str) -> Optional[Dict[str, Any]]:
    """Locate a previously saved neural-detector result.

    Recorded as a real completed experiment with a FAIL verdict. It must
    not be retrained merely to change the answer, and it must never be
    converted to NOT_RUN just because the dataset is gone.
    """
    candidates = [
        RESULTS_DIR / f"{EXPERIMENT}_{model_id.replace('/', '__').lower()}.json",
        RESULTS_DIR / "exp7_neural_detector_results.json",
    ]

    for path in candidates:
        if not path.exists():
            continue

        try:
            data = __import__("json").loads(
                path.read_text(encoding="utf-8")
            )
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        if data.get("model_id") == model_id and "accuracy" in data:
            return data

        item = data.get(model_id)
        if isinstance(item, dict) and "accuracy" in item:
            return item

    return None


def run(context: ModelContext, allow_training: bool = False) -> Dict[str, Any]:
    gate = gate_for(EXPERIMENT)

    # --- Reuse a completed result before considering any work -------
    saved = existing_result(context.model_id)

    if saved is not None:
        accuracy = float(saved["accuracy"])
        passed = accuracy <= gate["max_detector_accuracy"]

        return {
            "experiment": EXPERIMENT,
            "title": "Security (neural steganalysis)",
            "model_id": context.model_id,
            "configuration": {
                "detector": "MLP 4096 -> 256 -> 64 -> 1",
                "loss": "BCEWithLogitsLoss",
                "optimizer": "Adam",
                "dataset": saved.get("dataset_path"),
                "seed": saved.get("seed"),
                "test_ratio": saved.get("test_ratio"),
                "epochs": saved.get("epochs"),
                "batch_size": saved.get("batch_size"),
                "learning_rate": saved.get("learning_rate"),
                "hidden_1": saved.get("hidden_1"),
                "hidden_2": saved.get("hidden_2"),
                "scope": (
                    "carrier-centered stress test on paired patches "
                    "from one model; not a universal-adversary claim"
                ),
            },
            "metrics": {
                "accuracy": accuracy,
                "accuracy_percent": accuracy * 100.0,
                "tp": saved.get("tp"),
                "tn": saved.get("tn"),
                "fp": saved.get("fp"),
                "fn": saved.get("fn"),
                "threshold": saved.get("threshold", NEURAL_THRESHOLD),
                "gate_rule": saved.get(
                    "gate_rule",
                    "accuracy <= 0.55",
                ),
            },
            "thresholds": gate,
            "status": "PASS" if passed else "FAIL",
            "gate_status": "PASS" if passed else "FAIL",
            "reproducibility": context.reproducibility(),
            "notes": (
                "Reused a previously completed neural-detector run. "
                "This is a real FAIL result and is preserved as such: "
                "the embedding passes the statistical gate but a "
                "carrier-centered neural adversary detects it above "
                "threshold."
                if not passed
                else "Reused a previously completed neural-detector run."
            ),
            "source": "reused_artifact",
            "reused_from": saved.get("dataset_path"),
        }

    # --- No saved result --------------------------------------------
    if not allow_training:
        return {
            "experiment": EXPERIMENT,
            "title": "Security (neural steganalysis)",
            "model_id": context.model_id,
            "configuration": {},
            "metrics": {},
            "thresholds": gate,
            "status": "NOT_RUN",
            "gate_status": "NOT_RUN",
            "reproducibility": context.reproducibility(),
            "notes": (
                "No saved neural-detector result for this model. "
                "Training requires building the paired-patch dataset "
                "(expensive); enable it explicitly rather than having "
                "a cross-model run trigger it as a side effect."
            ),
            "source": "run",
        }

    path = dataset_path()

    if not path.exists():
        return {
            "experiment": EXPERIMENT,
            "title": "Security (neural steganalysis)",
            "model_id": context.model_id,
            "configuration": {"dataset_path": str(path)},
            "metrics": {},
            "thresholds": gate,
            "status": "NOT_RUN",
            "gate_status": "NOT_RUN",
            "reproducibility": context.reproducibility(),
            "notes": (
                f"Detector dataset missing at {path}. Build it with "
                "src/steganalysis/build_exp7_detector_dataset.py."
            ),
            "source": "run",
        }

    metrics = _train_and_evaluate(path)

    accuracy = float(metrics["accuracy"])
    passed = accuracy <= gate["max_detector_accuracy"]

    return {
        "experiment": EXPERIMENT,
        "title": "Security (neural steganalysis)",
        "model_id": context.model_id,
        "configuration": {
            "dataset_path": str(path),
            "detector": "MLP 4096 -> 256 -> 64 -> 1",
            "epochs": 30,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "seed": 42,
        },
        "metrics": metrics,
        "thresholds": gate,
        "status": "PASS" if passed else "FAIL",
        "gate_status": "PASS" if passed else "FAIL",
        "reproducibility": context.reproducibility(),
        "notes": (
            "Neural adversary exceeds the 55% gate. Preserved as a "
            "real FAIL."
            if not passed
            else "Neural adversary within threshold."
        ),
        "source": "run",
    }


def _train_and_evaluate(dataset_path: Path) -> Dict[str, Any]:
    """Train the detector using the existing, unmodified Exp7 code.

    The Exp7 module owns the architecture, loss, split and training loop.
    Reimplementing any of it here would create a second definition of the
    same experiment that could drift from the recorded one.
    """
    import torch

    from src.steganalysis.exp7_neural_detector import (
        BATCH_SIZE,
        Detector,
        ResidualDataset,
        evaluate,
        split_by_sample_id,
    )
    from torch.utils.data import DataLoader

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    with open(dataset_path, "rb") as handle:
        samples = pickle.load(handle)

    train_samples, test_samples = split_by_sample_id(samples)

    train_loader = DataLoader(
        ResidualDataset(train_samples),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        ResidualDataset(test_samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    input_dim = len(train_samples[0]["values"])
    model = Detector(input_dim).to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4
    )

    for _ in range(30):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x).squeeze(1), batch_y)
            loss.backward()
            optimizer.step()

    metrics = dict(evaluate(model, test_loader, device))
    metrics["dataset_samples"] = len(samples)
    metrics["train_samples"] = len(train_samples)
    metrics["test_samples"] = len(test_samples)
    metrics["input_dim"] = input_dim
    metrics["device"] = str(device)
    metrics["timestamp"] = utc_now()

    return metrics