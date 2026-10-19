"""
W3.1 — cross-scheme detector.

RESEARCH_PLAN §0 "Not established" #2: *every detectability number
uses a detector trained against the same scheme it tests —
cross-scheme is untested*, and §3 calls it "the claim's main
weakness." This experiment trains a detector on one scheme's
carrier-centred patches and tests it on the other's.

Four evaluations, two of them controls:

    train sign -> test sign   within-scheme control (must clear the
                              gate or the cross numbers mean nothing)
    train sign -> test LWE    cross
    train LWE  -> test LWE    within-scheme control
    train LWE  -> test sign   cross

Placement is carrier-centred for both schemes — exp14 established that
blind placement is chance (50.0%), so placement is held at the
detectable setting and ONLY the scheme varies. That isolates the
question: does structure learned from sign's modifications transfer to
LWE's, and vice versa?

Design, following exp14:

    - pairs cut from the same layers/positions of the same clean
      tensor; only the stego side differs between classes;
    - 6 embeddings per scheme x 72 pairs = 432 pairs per scheme
      (>= the suite's 400-pair floor);
    - split by embedding: 4 embeds train / 2 test, so no stego tensor
      appears on both sides;
    - Detector, epochs, batch, lr, seed imported from exp7's module —
      identical to exp7/exp14's training loop;
    - message length varies per embedding so the Hamilton allocation
      shifts across embeds.

Gate: THRESHOLDS['exp16'].max_cross_scheme_detector_accuracy = 0.55
— exp7_neural's existing number, reused. BOTH cross directions must
be at or below it; the within-scheme controls carry no gate but are
recorded (and claim_audit asserts they clear it, so the cross result
is never read from a broken pipeline).

What either outcome means:

    cross <= 55%  a detector must be trained on the scheme it hunts;
                  the 70.5% FAIL does not generalise across schemes.
    cross > 55%   the schemes share detectable structure and a
                  generic detector catches both.

Sign steganography is the production default; LWE is measured here
because it is the other implemented strategy (and, per exp13, its
key-gating is refuted — this experiment says nothing about that).
"""

import bisect
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.core.types import EmbeddingConfig  # noqa: E402
from src.embedding.intelligent_embedder import IntelligentEmbedder  # noqa: E402
from src.experiments.experiment_registry import gate_for  # noqa: E402
from src.experiments.paths import RESULTS_DIR  # noqa: E402
from src.experiments.residual_source import load_cached_residuals  # noqa: E402

EXPERIMENT = "exp16"

MODEL_ID = "Qwen/Qwen2.5-3B"
FAMILY = "qwen"
NUM_LAYERS = 36

PATCH_SIZE = 4096
PAYLOAD_BITS = 10_000

SCHEMES = ("sign", "lwe")
NUM_EMBEDS = 6          # per scheme: 4 train / 2 test
PATCHES_PER_EMBED = 72  # -> 432 pairs per scheme
TRAIN_EMBEDS = 4

SEED = 42


def _carrier_start(carriers: List[int], rng: random.Random) -> int:
    """exp7/exp14's control rule: aligned patch containing a carrier."""
    carrier = rng.choice(carriers)
    return (carrier // PATCH_SIZE) * PATCH_SIZE


def _count_in_patch(sorted_indices: List[int], start: int) -> int:
    lo = bisect.bisect_left(sorted_indices, start)
    hi = bisect.bisect_left(sorted_indices, start + PATCH_SIZE)
    return hi - lo


def build_scheme_dataset(
    residuals: Dict[int, torch.Tensor],
    scheme: str,
    seed_offset: int,
):
    """Cut TRAIN_EMBEDS + (NUM_EMBEDS - TRAIN_EMBEDS) embeds' pairs.

    Returns (train_samples, test_samples, embed_stats).
    """
    rng = random.Random(SEED + seed_offset)
    train: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    stats: List[Dict[str, Any]] = []

    for k in range(NUM_EMBEDS):
        print(f"[exp16] {scheme} embed {k + 1}/{NUM_EMBEDS} ...",
              flush=True)

        config = EmbeddingConfig(
            total_payload_bits=PAYLOAD_BITS,
            embedding_strategy=scheme,
            model_family=FAMILY,
            num_hidden_layers=NUM_LAYERS,
        )
        message = f"EXP16_{scheme.upper()}_{k:02d}_" + "A" * (1200 + k)
        embed = IntelligentEmbedder(config).embed(message, residuals)

        stego = embed.embedded_residuals
        carrier_layers = sorted(
            lid for lid in embed.carrier_indices
            if embed.carrier_indices[lid]
        )
        bucket = train if k < TRAIN_EMBEDS else test
        carriers_per_patch = 0

        for _ in range(PATCHES_PER_EMBED):
            lid = rng.choice(carrier_layers)
            carriers = sorted(embed.carrier_indices[lid])
            start = _carrier_start(carriers, rng)
            carriers_per_patch += _count_in_patch(carriers, start)

            clean = residuals[lid].reshape(-1)[start:start + PATCH_SIZE]
            stg = stego[lid].reshape(-1)[start:start + PATCH_SIZE]
            if clean.numel() != PATCH_SIZE or stg.numel() != PATCH_SIZE:
                continue

            bucket.append(
                {"features": clean.clone(), "label": 0,
                 "sample_id": k, "layer_id": lid,
                 "patch_start": start}
            )
            bucket.append(
                {"features": stg.clone(), "label": 1,
                 "sample_id": k, "layer_id": lid,
                 "patch_start": start}
            )

        stats.append(
            {
                "embed": k,
                "total_bits": embed.total_bits,
                "carriers_in_patches": carriers_per_patch,
                "split": "train" if k < TRAIN_EMBEDS else "test",
            }
        )
        del stego

    return train, test, stats


def train_detector(samples: List[Dict[str, Any]]):
    """exp7's training loop, unchanged (imported, like exp14)."""
    from torch.utils.data import DataLoader

    from src.steganalysis.exp7_neural_detector import (
        BATCH_SIZE,
        Detector,
        ResidualDataset,
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    random.seed(SEED)
    torch.manual_seed(SEED)

    loader = DataLoader(
        ResidualDataset(samples),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    model = Detector(len(samples[0]["features"])).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for _ in range(30):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    return model, device


def evaluate(model, samples: List[Dict[str, Any]], device) -> Dict[str, Any]:
    from torch.utils.data import DataLoader

    from src.steganalysis.exp7_neural_detector import (
        BATCH_SIZE,
        ResidualDataset,
        evaluate as exp7_evaluate,
    )

    loader = DataLoader(
        ResidualDataset(samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    metrics = dict(exp7_evaluate(model, loader, device))
    metrics["pairs"] = len(samples) // 2
    return metrics


def main() -> int:
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading residuals from cache ...", flush=True)
    residuals = load_cached_residuals(MODEL_ID, NUM_LAYERS)

    data: Dict[str, Dict[str, Any]] = {}
    for offset, scheme in enumerate(SCHEMES):
        print(f"\nBuilding {scheme} dataset ...", flush=True)
        train, test, stats = build_scheme_dataset(
            residuals, scheme, seed_offset=offset * 1000
        )
        data[scheme] = {"train": train, "test": test, "stats": stats}
        print(f"[exp16] {scheme}: {len(train) // 2} train pairs, "
              f"{len(test) // 2} test pairs", flush=True)

    results: Dict[str, Any] = {}
    for train_scheme in SCHEMES:
        print(f"\nTraining detector on {train_scheme} ...", flush=True)
        model, device = train_detector(data[train_scheme]["train"])
        for test_scheme in SCHEMES:
            tag = f"{train_scheme}_to_{test_scheme}"
            metrics = evaluate(
                model, data[test_scheme]["test"], device
            )
            results[tag] = metrics
            print(f"  {tag}: accuracy = {metrics['accuracy']:.4f} "
                  f"(pairs {metrics['pairs']})", flush=True)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    gate = gate_for(EXPERIMENT)
    max_acc = gate["max_cross_scheme_detector_accuracy"]
    cross = {
        "sign_to_lwe": results["sign_to_lwe"]["accuracy"],
        "lwe_to_sign": results["lwe_to_sign"]["accuracy"],
    }
    within = {
        "sign_to_sign": results["sign_to_sign"]["accuracy"],
        "lwe_to_lwe": results["lwe_to_lwe"]["accuracy"],
    }
    passed = all(v <= max_acc for v in cross.values())
    controls_ok = all(v > max_acc for v in within.values())

    artifact = {
        "experiment": EXPERIMENT,
        "title": "W3.1 — cross-scheme detector",
        "model_id": MODEL_ID,
        "family": FAMILY,
        "status": "PASS" if passed else "FAIL",
        "gate": {
            "max_cross_scheme_detector_accuracy": max_acc,
            "measured_cross": cross,
            "status": "PASS" if passed else "FAIL",
            "gate_source": "experiment_registry.THRESHOLDS['exp16']",
        },
        "results": results,
        "within_scheme_controls": within,
        "controls_valid": controls_ok,
        "method": {
            "placement": (
                "carrier-centred (exp7/exp14 control rule); placement "
                "held fixed so only the scheme varies"
            ),
            "embeddings_per_scheme": NUM_EMBEDS,
            "pairs_per_scheme": NUM_EMBEDS * PATCHES_PER_EMBED,
            "split": (
                f"{TRAIN_EMBEDS} embeds train / "
                f"{NUM_EMBEDS - TRAIN_EMBEDS} test, per scheme; no "
                "stego tensor on both sides"
            ),
            "detector": "MLP 4096 -> 256 -> 64 -> 1 (exp7 module)",
            "epochs": 30,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "seed": SEED,
            "payload_bits": PAYLOAD_BITS,
            "residuals": "shared residual cache (same source exp10+)",
            "per_embedding": {
                scheme: data[scheme]["stats"] for scheme in SCHEMES
            },
        },
        "notes": [
            "Within-scheme controls carry no gate; if either does not "
            "clear the 55% line, the cross numbers are uninformative "
            "and claim_audit fails on controls_valid.",
            "exp13 separately refuted LWE's key-gating; this "
            "experiment is about detectability transfer only.",
        ],
    }

    target = RESULTS_DIR / f"{EXPERIMENT}_cross_scheme_detector.json"
    target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    for tag in sorted(results):
        print(f"{tag:16s} accuracy = {results[tag]['accuracy']:.4f}")
    print(f"controls clear gate: {controls_ok}")
    print(f"gate (both cross <= {max_acc}): {artifact['status']}")
    print("=" * 70)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
