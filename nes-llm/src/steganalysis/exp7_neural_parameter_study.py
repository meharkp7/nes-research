"""
Diagnostic study: is the neural-detector FAIL tunable?

The Exp7 neural gate fails at 70.5% accuracy on Qwen2.5-3B. The useful
question is not "how do we make this number smaller" but "is this
number a property of a tunable parameter, or is it structural?"

This sweeps the embedding parameters that could plausibly affect
detectability and trains the same detector on each variant:

    alpha      perturbation floor (EmbeddingConfig.min_magnitude)
    gamma      QACI layer-concentration
    payload    number of carriers touched

If accuracy is flat across the sweep, the FAIL is structural and the
honest conclusion is that the sign-embedding scheme is intrinsically
detectable at carrier positions — not that the parameters need tuning.

This is a separate diagnostic artifact. It does NOT replace, relax or
reinterpret the recorded 70.5% result, and it does not change the 55%
gate.
"""

import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.core.types import EmbeddingConfig  # noqa: E402
from src.embedding.intelligent_embedder import (  # noqa: E402
    IntelligentEmbedder,
)
from src.experiments.residual_source import (  # noqa: E402
    load_cached_residuals,
)
from src.steganalysis.exp7_neural_detector import (  # noqa: E402
    BATCH_SIZE,
    Detector,
    ResidualDataset,
    evaluate,
    split_by_sample_id,
)
from torch.utils.data import DataLoader  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-3B"
FAMILY = "qwen"
NUM_LAYERS = 36

# Deliberately smaller than the recorded 500-pair dataset so a full
# sweep is tractable. Every variant uses the same size, so the sweep
# stays internally comparable; this is NOT the headline 70.5% dataset.
PAIRS = 150
PATCH_SIZE = 4096
EPOCHS = 20
SEED = 42

GATE = 0.55

# Defaults taken from EmbeddingConfig: min_magnitude=0.001, gamma=2.5.
# The sweep varies around the values production actually uses.
DEFAULT_ALPHA = 0.001
DEFAULT_GAMMA = 2.5
DEFAULT_PAYLOAD = 10_000

VARIANTS: List[Dict[str, Any]] = [
    {"label": "baseline (alpha=1e-3, gamma=2.5)", "alpha": DEFAULT_ALPHA, "gamma": DEFAULT_GAMMA, "payload": DEFAULT_PAYLOAD},
    {"label": "alpha 1e-4 (10x smaller)", "alpha": 1e-4, "gamma": DEFAULT_GAMMA, "payload": DEFAULT_PAYLOAD},
    {"label": "alpha 1e-2 (10x larger)", "alpha": 1e-2, "gamma": DEFAULT_GAMMA, "payload": DEFAULT_PAYLOAD},
    {"label": "gamma 1.0 (wider spread)", "alpha": DEFAULT_ALPHA, "gamma": 1.0, "payload": DEFAULT_PAYLOAD},
    {"label": "gamma 5.0 (narrower)", "alpha": DEFAULT_ALPHA, "gamma": 5.0, "payload": DEFAULT_PAYLOAD},
    {"label": "payload 100k (10x carriers)", "alpha": DEFAULT_ALPHA, "gamma": DEFAULT_GAMMA, "payload": 100_000},
]


def build_dataset(
    residuals: Dict[int, torch.Tensor],
    alpha: float,
    gamma: float,
    payload: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Build paired clean/stego patches for one parameter variant."""
    dataset = []

    for sample_id in range(PAIRS):
        config = EmbeddingConfig(
            total_payload_bits=payload,
            model_family=FAMILY,
            num_hidden_layers=NUM_LAYERS,
            gamma=gamma,
            min_magnitude=alpha,
        )

        result = IntelligentEmbedder(config).embed(
            f"EXP7_SWEEP_{sample_id}_" + "A" * 1200,
            residuals,
        )

        # Layers that actually received carriers.
        candidates = [
            layer_id
            for layer_id in residuals
            if result.carrier_indices.get(layer_id)
        ]

        if not candidates:
            continue

        for _ in range(2):
            layer_id = rng.choice(candidates)
            indices = sorted(result.carrier_indices[layer_id])
            position = rng.choice(indices)

            start = (
                position // PATCH_SIZE
            ) * PATCH_SIZE

            clean_patch = residuals[layer_id].flatten()[
                start : start + PATCH_SIZE
            ]
            stego_patch = (
                result.embedded_residuals[layer_id]
                .flatten()[start : start + PATCH_SIZE]
            )

            if len(clean_patch) < PATCH_SIZE:
                continue

            for label, values in ((0, clean_patch), (1, stego_patch)):
                dataset.append(
                    {
                        "sample_id": sample_id,
                        "layer_id": layer_id,
                        "patch_start": int(start),
                        "label": label,
                        # ResidualDataset reads "features" and calls
                        # .float() on it, so store a tensor, not numpy.
                        "features": values.detach().float().clone(),
                    }
                )

    return dataset


def train_and_score(
    dataset: List[Dict[str, Any]],
    device: torch.device,
) -> float:
    if not dataset:
        return float("nan")

    train, test = split_by_sample_id(dataset)

    if not train or not test:
        return float("nan")

    train_loader = DataLoader(
        ResidualDataset(train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        ResidualDataset(test),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = Detector(len(train[0]["values"])).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for _ in range(EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(
                model(batch_x).squeeze(1), batch_y
            )
            loss.backward()
            optimizer.step()

    return float(evaluate(model, test_loader, device)["accuracy"])


def main() -> int:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading residuals from cache...")
    residuals = load_cached_residuals(MODEL_ID, NUM_LAYERS)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"device: {device}\n")

    results = []

    print(f"{'variant':<38}{'accuracy':>10}  gate")
    print("-" * 62)

    for variant in VARIANTS:
        rng = random.Random(SEED)

        dataset = build_dataset(
            residuals,
            alpha=variant["alpha"],
            gamma=variant["gamma"],
            payload=variant["payload"],
            rng=rng,
        )

        accuracy = train_and_score(dataset, device)

        results.append(
            {
                "variant": variant["label"],
                "alpha": variant["alpha"],
                "gamma": variant["gamma"],
                "payload_bits": variant["payload"],
                "accuracy": accuracy,
                "meets_gate": accuracy <= GATE,
                "dataset_samples": len(dataset),
            }
        )

        verdict = "PASS" if accuracy <= GATE else "FAIL"
        print(f"{variant['label']:<38}{accuracy:>9.2%}  {verdict}")

    accuracies = [r["accuracy"] for r in results]
    spread = max(accuracies) - min(accuracies)
    any_pass = any(r["meets_gate"] for r in results)

    print()
    print(f"range across sweep : {min(accuracies):.2%} - {max(accuracies):.2%}")
    print(f"spread             : {spread:.2%}")
    print(f"any variant <= {GATE:.0%} : {any_pass}")

    conclusion = (
        "Detectability is essentially insensitive to alpha, gamma and "
        "payload size across the swept range, so the 70.5% FAIL is a "
        "structural property of sign-based embedding at carrier "
        "positions rather than a tuning problem."
        if not any_pass and spread < 0.05
        else "Detectability varies materially with embedding parameters; "
        "at least one variant reaches the gate."
    )

    print()
    print(conclusion)

    out = Path(__file__).resolve().parents[2] / "results"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "exp7_neural_parameter_study.json"

    target.write_text(
        __import__("json").dumps(
            {
                "experiment": "exp7_neural_parameter_study",
                "title": "Neural detectability vs embedding parameters",
                "model_id": MODEL_ID,
                "gate": GATE,
                "question": (
                    "Is the 70.5% neural-detector FAIL tunable?"
                ),
                "method": {
                    "pairs_per_variant": PAIRS,
                    "patch_size": PATCH_SIZE,
                    "epochs": EPOCHS,
                    "seed": SEED,
                    "note": (
                        "Smaller than the recorded 500-pair dataset; "
                        "internally comparable across variants only."
                    ),
                },
                "variants": results,
                "range": {"min": min(accuracies), "max": max(accuracies)},
                "spread": spread,
                "any_variant_meets_gate": any_pass,
                "conclusion": conclusion,
                "preserves_recorded_result": True,
                "recorded_neural_accuracy": 0.705,
                "interpretation": (
                    "This diagnostic does not replace the recorded "
                    "70.5% result and does not change the 55% gate."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())