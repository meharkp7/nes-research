"""
W3.2 — blind-patch adversary.

exp7_neural's 70.5% FAIL cuts both the clean and the stego patch at a
carrier position: the detector is handed the locations (RESEARCH_PLAN
§0, "Not established" #3 — *clean and stego patches are both cut at
carrier positions*). This experiment removes that hand-out and asks
what an adversary that does not know carrier positions can do.

Two arms, cut from the SAME embeddings:

    blind    patch position drawn uniformly at random, independent of
             the carrier set. Carrier presence inside a blind patch is
             whatever chance provides: carriers are ~0.001% of
             positions, so a 4096-value patch expects ~0.05 carriers
             and ~5% of patches contain any at all.
    control  the exp7 rule — an aligned patch containing a carrier.
             Positive control: if this arm does NOT come out clearly
             above the gate, the blind arm's result is uninformative
             and must be read as such.

Detector, architecture, epochs, batch size, learning rate and seed are
imported from `src.steganalysis.exp7_neural_detector` — the same code
exp7 trained with — so the only variable between arms, and against
exp7, is patch placement.

Split: by embedding (`sample_id` = embed index), so no stego tensor
appears on both sides of train/test. exp7 split by pair while every
pair came from the same model's structure; splitting by embedding is
stricter.

Gate: THRESHOLDS["exp14"].max_blind_detector_accuracy = 0.55 — the
same self-chosen 55% number as exp7_neural (§0 "Not established" #4),
reused rather than invented. Only the blind arm is gated; the control
is a recorded validity check, not a second gate.

Message confidentiality scope is not at issue here; this is a
detectability experiment against the production sign strategy.
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

EXPERIMENT = "exp14"

MODEL_ID = "Qwen/Qwen2.5-3B"
FAMILY = "qwen"
NUM_LAYERS = 36

PATCH_SIZE = 4096
PAYLOAD_BITS = 10_000

# 12 embeddings x 48 patch pairs per arm = 576 pairs per arm (the
# suite's floor is 400). Split 80/20 by embedding: 9 train / 3 test.
NUM_EMBEDS = 12
PATCHES_PER_EMBED = 48

SEED = 42


def _aligned_start(flat_len: int, rng: random.Random) -> int:
    """A uniform random aligned patch start, independent of carriers."""
    return rng.randrange(flat_len // PATCH_SIZE) * PATCH_SIZE


def _count_in_patch(sorted_indices: List[int], start: int) -> int:
    lo = bisect.bisect_left(sorted_indices, start)
    hi = bisect.bisect_left(sorted_indices, start + PATCH_SIZE)
    return hi - lo


def build_datasets(residuals: Dict[int, torch.Tensor]):
    """Cut both arms from NUM_EMBEDS embeddings.

    Returns (blind_pairs, control_pairs, per_embed stats).
    """
    rng = random.Random(SEED)
    blind: List[Dict[str, Any]] = []
    control: List[Dict[str, Any]] = []
    embed_stats: List[Dict[str, Any]] = []

    for k in range(NUM_EMBEDS):
        print(f"embed {k + 1}/{NUM_EMBEDS} ...", flush=True)

        config = EmbeddingConfig(
            total_payload_bits=PAYLOAD_BITS,
            model_family=FAMILY,
            num_hidden_layers=NUM_LAYERS,
        )
        # Varying message length by one byte per embedding changes
        # total_bits, so the Hamilton allocation shifts and the
        # control arm is not carved from one fixed carrier set.
        message = f"EXP14_SAMPLE_{k:02d}_" + "A" * (1200 + k)
        embed = IntelligentEmbedder(config).embed(message, residuals)

        stego = embed.embedded_residuals
        carriers = embed.carrier_indices
        carrier_layers = sorted(
            lid for lid in carriers if carriers[lid]
        )

        blind_with_carrier = 0
        control_carriers_per_patch = 0

        for _ in range(PATCHES_PER_EMBED):
            # ---- blind arm: position knows nothing about carriers
            lid = rng.randrange(NUM_LAYERS)
            flat_len = residuals[lid].reshape(-1).numel()
            start = _aligned_start(flat_len, rng)
            sorted_idx = sorted(carriers.get(lid, []))
            n_car = _count_in_patch(sorted_idx, start)
            blind_with_carrier += n_car > 0

            clean = residuals[lid].reshape(-1)[start:start + PATCH_SIZE]
            stg = stego[lid].reshape(-1)[start:start + PATCH_SIZE]
            if clean.numel() != PATCH_SIZE or stg.numel() != PATCH_SIZE:
                continue

            blind.append(
                {"features": clean.clone(), "label": 0,
                 "sample_id": k, "layer_id": lid,
                 "patch_start": start, "carriers_in_patch": n_car}
            )
            blind.append(
                {"features": stg.clone(), "label": 1,
                 "sample_id": k, "layer_id": lid,
                 "patch_start": start, "carriers_in_patch": n_car}
            )

            # ---- control arm: exp7's rule, aligned patch on a carrier
            ctl_lid = rng.choice(carrier_layers)
            ctl_carrier = rng.choice(carriers[ctl_lid])
            ctl_start = (
                ctl_carrier // PATCH_SIZE
            ) * PATCH_SIZE
            sorted_ctl = sorted(carriers[ctl_lid])
            control_carriers_per_patch += _count_in_patch(
                sorted_ctl, ctl_start
            )

            clean_c = residuals[ctl_lid].reshape(-1)[
                ctl_start:ctl_start + PATCH_SIZE
            ]
            stg_c = stego[ctl_lid].reshape(-1)[
                ctl_start:ctl_start + PATCH_SIZE
            ]
            if clean_c.numel() != PATCH_SIZE or stg_c.numel() != PATCH_SIZE:
                continue

            control.append(
                {"features": clean_c.clone(), "label": 0,
                 "sample_id": k, "layer_id": ctl_lid,
                 "patch_start": ctl_start}
            )
            control.append(
                {"features": stg_c.clone(), "label": 1,
                 "sample_id": k, "layer_id": ctl_lid,
                 "patch_start": ctl_start}
            )

        embed_stats.append(
            {
                "embed": k,
                "total_bits": embed.total_bits,
                "blind_patches_with_carrier": blind_with_carrier,
                "control_carriers_sum": control_carriers_per_patch,
            }
        )
        del stego

    return blind, control, embed_stats


def train_and_evaluate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The exp7 training loop, unchanged, on the given arm."""
    from torch.utils.data import DataLoader

    from src.steganalysis.exp7_neural_detector import (
        BATCH_SIZE,
        Detector,
        ResidualDataset,
        evaluate,
        split_by_sample_id,
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Re-seed before the split so both arms get the same shuffle over
    # the same sample_ids, and the run is reproducible on its own.
    random.seed(SEED)
    torch.manual_seed(SEED)

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

    model = Detector(len(train_samples[0]["features"])).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for _ in range(30):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    metrics = dict(evaluate(model, test_loader, device))
    metrics.update(
        {
            "pairs": len(samples) // 2,
            "train_pairs": len(train_samples) // 2,
            "test_pairs": len(test_samples) // 2,
            "device": str(device),
        }
    )
    return metrics


def main() -> int:
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading residuals from cache ...", flush=True)
    residuals = load_cached_residuals(MODEL_ID, NUM_LAYERS)

    print("Building blind + control datasets ...", flush=True)
    blind, control, embed_stats = build_datasets(residuals)

    print(f"\nblind pairs:    {len(blind) // 2}")
    print(f"control pairs:  {len(control) // 2}")

    print("\nTraining blind adversary ...", flush=True)
    blind_metrics = train_and_evaluate(blind)
    print(f"  accuracy = {blind_metrics['accuracy']:.4f}")

    print("\nTraining carrier-centered control ...", flush=True)
    control_metrics = train_and_evaluate(control)
    print(f"  accuracy = {control_metrics['accuracy']:.4f}")

    gate = gate_for(EXPERIMENT)
    max_acc = gate["max_blind_detector_accuracy"]
    blind_acc = float(blind_metrics["accuracy"])
    passed = blind_acc <= max_acc

    blind_with_carrier = sum(
        s["blind_patches_with_carrier"] for s in embed_stats
    )
    blind_pairs = len(blind) // 2
    # Each pair is (clean, stego) sharing one position.
    blind_positions = blind_pairs

    artifact = {
        "experiment": EXPERIMENT,
        "title": "W3.2 — blind-patch adversary",
        "model_id": MODEL_ID,
        "family": FAMILY,
        "layers": NUM_LAYERS,
        "status": "PASS" if passed else "FAIL",
        "gate": {
            "max_blind_detector_accuracy": max_acc,
            "measured": {"blind_accuracy": blind_acc},
            "status": "PASS" if passed else "FAIL",
        },
        "arms": {
            "blind": {
                "placement": (
                    "uniform random aligned position; knows nothing "
                    "about carrier locations"
                ),
                "metrics": blind_metrics,
                "positions": blind_positions,
                "positions_containing_carrier": blind_with_carrier,
                "fraction_containing_carrier": (
                    blind_with_carrier / blind_positions
                    if blind_positions else None
                ),
            },
            "control": {
                "placement": (
                    "exp7 rule: aligned patch containing a carrier"
                ),
                "metrics": control_metrics,
            },
        },
        "method": {
            "embeddings": NUM_EMBEDS,
            "patch_pairs_per_arm": PATCHES_PER_EMBED,
            "split": (
                "by embedding (sample_id = embed index), 80/20; no "
                "stego tensor on both sides"
            ),
            "detector": "MLP 4096 -> 256 -> 64 -> 1 (exp7 module, unchanged)",
            "epochs": 30,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "seed": SEED,
            "payload_bits": PAYLOAD_BITS,
            "strategy": "sign (production default)",
            "residuals": "shared residual cache (same source exp10/12/13)",
            "gate_source": "experiment_registry.THRESHOLDS['exp14']",
        },
        "per_embedding": embed_stats,
        "notes": [
            "The control arm is a validity check, not a second gate: "
            "if it does not clearly exceed the blind arm, the blind "
            "result carries no information about adversary strength.",
            "exp7_neural's 70.5% FAIL (exp7_neural_*.json) cut both "
            "patches at carrier positions; this experiment changes "
            "only where patches are cut.",
        ],
    }

    target = RESULTS_DIR / f"{EXPERIMENT}_blind_patch_detector.json"
    target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print()
    print("=" * 70)
    print(f"blind accuracy:    {blind_acc:.4f}")
    print(f"control accuracy:  {control_metrics['accuracy']:.4f}")
    print(f"blind positions containing a carrier: "
          f"{blind_with_carrier}/{blind_positions}")
    print(f"gate (<= {max_acc}): {artifact['status']}")
    print("=" * 70)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
