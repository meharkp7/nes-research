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
from src.experiments.paths import RESULTS_DIR  # noqa: E402
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

# Matched to the recorded protocol's scale as closely as is affordable.
#
# The first attempt used 120 pairs and every variant scored exactly
# 50.00% -- chance. That was an underpowered study, not evidence of
# undetectability: sign embedding changes only a handful of values in a
# 4096-wide carrier patch, so 120 pairs cannot train a detector to find
# it. The recorded result used 500 pairs and reached 70.5%.
PAIRS = 400
# Patches are sampled from fewer distinct embeddings (see build_dataset).
# Carrier positions come from residual quality, not payload content, so
# extra embeddings mostly change the AES nonce rather than the patches a
# detector sees.
EMBEDDINGS_PER_VARIANT = 20
PATCH_SIZE = 4096
EPOCHS = 30
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
    """Build paired clean/stego patches for one parameter variant.

    Embeddings are amortised across pairs. QACI selects carriers from
    residual quality alone — it does not look at payload content — so the
    carrier positions are identical for every message at a given payload
    size. Re-embedding once per pair (as the recorded dataset builder
    does) costs ~9.4s per call and buys only a fresh AES nonce for the
    bit values, while the patches a detector sees are drawn from the
    same carrier set.

    So: EMBEDDINGS_PER_VARIANT distinct embeddings, each sampled at many
    positions. That trades payload-bit diversity for tractability, and
    the trade is recorded in the artifact. Since all variants use the
    same setting, the sweep stays internally comparable.
    """
    dataset = []

    per_embedding = max(PAIRS // EMBEDDINGS_PER_VARIANT, 1)

    for embed_id in range(EMBEDDINGS_PER_VARIANT):
        config = EmbeddingConfig(
            total_payload_bits=payload,
            model_family=FAMILY,
            num_hidden_layers=NUM_LAYERS,
            gamma=gamma,
            min_magnitude=alpha,
        )

        result = IntelligentEmbedder(config).embed(
            f"EXP7_SWEEP_{embed_id}_" + "A" * 1200,
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

        sample_id_base = embed_id * per_embedding

        for offset in range(per_embedding):
            layer_id = rng.choice(candidates)
            indices = sorted(result.carrier_indices[layer_id])
            position = rng.choice(indices)

            start = (position // PATCH_SIZE) * PATCH_SIZE

            clean_patch = residuals[layer_id].flatten()[
                start : start + PATCH_SIZE
            ]
            stego_patch = (
                result.embedded_residuals[layer_id]
                .flatten()[start : start + PATCH_SIZE]
            )

            if len(clean_patch) < PATCH_SIZE:
                continue

            sample_id = sample_id_base + offset

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

    model = Detector(len(train[0]["features"])).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for _ in range(EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            # Detector.forward already applies squeeze(-1); squeezing
            # again raises on the 1-D result.
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    return float(evaluate(model, test_loader, device)["accuracy"])


def measure_signal_density(
    dataset: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """How much of each patch actually differs, and how many pairs are identical.

    This is the mechanism behind detectability, and it also explains why
    a small study scores exactly 50%: sign embedding rewrites a carrier
    to +/-|r|, which only changes the tensor when the payload bit
    disagrees with the carrier's original sign. So roughly half the
    carriers are left untouched, and some sampled pairs come out
    byte-identical.
    """
    by_id: Dict[int, Dict[int, Any]] = {}

    for sample in dataset:
        by_id.setdefault(sample["sample_id"], {})[sample["label"]] = (
            sample["features"]
        )

    changed_counts = []
    identical = 0

    for pair in by_id.values():
        if 0 not in pair or 1 not in pair:
            continue
        delta = (pair[1] - pair[0]).abs()
        n_changed = int((delta > 0).sum().item())
        changed_counts.append(n_changed)
        if n_changed == 0:
            identical += 1

    return {
        "pairs": len(changed_counts),
        "mean_changed_values_per_patch": (
            sum(changed_counts) / len(changed_counts)
            if changed_counts
            else 0.0
        ),
        "max_changed_values_per_patch": max(changed_counts, default=0),
        "patch_size": PATCH_SIZE,
        "identical_pairs": identical,
        "identical_pair_fraction": (
            identical / len(changed_counts) if changed_counts else 0.0
        ),
    }


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
        density = measure_signal_density(dataset)

        results.append(
            {
                "variant": variant["label"],
                "alpha": variant["alpha"],
                "gamma": variant["gamma"],
                "payload_bits": variant["payload"],
                "accuracy": accuracy,
                "meets_gate": accuracy <= GATE,
                "dataset_samples": len(dataset),
                "signal_density": density,
            }
        )

        verdict = "PASS" if accuracy <= GATE else "FAIL"
        print(
            f"{variant['label']:<38}{accuracy:>9.2%}  {verdict}"
            f"   ({density['mean_changed_values_per_patch']:.0f}"
            f"/{PATCH_SIZE} values changed)"
        )

    accuracies = [r["accuracy"] for r in results]
    spread = max(accuracies) - min(accuracies)
    any_pass = any(r["meets_gate"] for r in results)

    # An all-exactly-50% sweep means the detector learned nothing, which
    # is a power failure, not a security result. Guard against reading it
    # as evidence of undetectability.
    all_at_chance = all(abs(a - 0.5) < 1e-9 for a in accuracies)

    print()
    print(f"range across sweep : {min(accuracies):.2%} - {max(accuracies):.2%}")
    print(f"spread             : {spread:.2%}")
    print(f"any variant <= {GATE:.0%} : {any_pass}")

    if all_at_chance:
        conclusion = (
            "INVALID STUDY: every variant scored exactly 50%, i.e. the "
            "detector learned nothing. The dataset is too weak to draw "
            "any conclusion about detectability. This must not be read "
            "as evidence that the embedding is undetectable."
        )
    elif not any_pass and spread < 0.05:
        conclusion = (
            "Detectability is essentially insensitive to alpha, gamma "
            "and payload size across the swept range, so the 70.5% FAIL "
            "is a structural property of sign-based embedding at "
            "carrier positions rather than a tuning problem."
        )
    else:
        conclusion = (
            "Detectability varies materially with embedding parameters; "
            "at least one variant reaches the gate."
        )

    print()
    print(conclusion)

    out = RESULTS_DIR
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
                    "embeddings_per_variant": EMBEDDINGS_PER_VARIANT,
                    "patch_size": PATCH_SIZE,
                    "epochs": EPOCHS,
                    "seed": SEED,
                    "note": (
                        "Smaller than the recorded 500-pair dataset and "
                        "amortised embeddings; internally comparable "
                        "across variants only, not a replacement for the "
                        "recorded result."
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