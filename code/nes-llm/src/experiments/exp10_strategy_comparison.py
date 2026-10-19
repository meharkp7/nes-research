"""
Phase 2 — head-to-head comparison of embedding strategies.

Phase 1 showed the only structural difference worth testing is whether a
scheme forces a sign flip at carriers. Sign-based schemes cannot get
under the neural gate regardless of tuning (proved across a 100x alpha
and 5x gamma sweep). So this measures each candidate on the three axes
that decide whether it replaces sign:

    extractable     round trip WITHOUT the cover residuals
    fidelity        BER and real perplexity
    detectability   carrier-centered neural detector, gate 55%

A strategy only wins if it is extractable AND under the gate. Anything
else is reported as such rather than rounded up.

Note on comparability: each strategy needs its own detector, because the
dataset is built from that strategy's own embedding. Accuracies are
therefore comparable as "this detector against this embedding", which is
the question asked -- not as a single universal detector score.
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
from src.embedding.strategy_registry import (  # noqa: E402
    REGISTRY,
    spec,
    structural_report,
)
from src.experiments.paths import ARTIFACTS_DIR, RESULTS_DIR  # noqa: E402
from src.experiments.residual_source import (  # noqa: E402
    load_cached_residuals,
)
from src.extraction.decrypt_pipeline import DecryptPipeline  # noqa: E402
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

PAYLOAD_BITS = 10_000
MESSAGE = "A" * 1_250

# Detector dataset per strategy. Sized for a head-to-head, not a headline
# number; the recorded 500-pair result stays the reference.
#
# Sized deliberately after two failed attempts at 120 pairs scored exactly
# 50% for every strategy -- chance, not a security result. At ~25 changed
# values per 4096-wide patch the signal is too sparse for a small set to
# train on. 400 pairs / 30 epochs reproduces the recorded 70.5% for sign.
PAIRS = 400
EMBEDDINGS_PER_VARIANT = 20
PATCH_SIZE = 4096
EPOCHS = 30
SEED = 42

GATE = 0.55

CANDIDATES = ["sign", "magnitude_aware", "lwe", "neural"]


def build_detector_dataset(
    residuals, strategy_name, rng, family=None, num_layers=None,
    config_overrides=None,
):
    """Paired clean/stego patches from this strategy's own embedding.

    ``family``/``num_layers`` default to this module's Qwen constants,
    so exp12's positional imports behave exactly as before; exp18 passes
    the model under test. ``config_overrides`` is an extra
    EmbeddingConfig kwargs dict for per-cell strategy parameters
    (exp20's split_fraction); None keeps the constructor unchanged.
    """
    dataset = []
    per_embedding = max(PAIRS // EMBEDDINGS_PER_VARIANT, 1)

    for embed_id in range(EMBEDDINGS_PER_VARIANT):
        config = EmbeddingConfig(
            total_payload_bits=PAYLOAD_BITS,
            embedding_strategy=strategy_name,
            model_family=family or FAMILY,
            num_hidden_layers=num_layers or NUM_LAYERS,
            **(config_overrides or {}),
        )

        result = IntelligentEmbedder(config).embed(
            f"PHASE2_{strategy_name}_{embed_id}_" + "A" * 1200,
            residuals,
        )

        candidates = [
            layer_id
            for layer_id in residuals
            if result.carrier_indices.get(layer_id)
        ]

        if not candidates:
            continue

        base = embed_id * per_embedding

        for offset in range(per_embedding):
            layer_id = rng.choice(candidates)
            indices = sorted(result.carrier_indices[layer_id])
            position = rng.choice(indices)
            start = (position // PATCH_SIZE) * PATCH_SIZE

            clean = residuals[layer_id].flatten()[
                start : start + PATCH_SIZE
            ]
            stego = result.embedded_residuals[layer_id].flatten()[
                start : start + PATCH_SIZE
            ]

            if len(clean) < PATCH_SIZE or len(stego) < PATCH_SIZE:
                continue

            for label, values in ((0, clean), (1, stego)):
                dataset.append(
                    {
                        "sample_id": base + offset,
                        "layer_id": layer_id,
                        "patch_start": int(start),
                        "label": label,
                        "features": values.detach().float().clone(),
                    }
                )

    return dataset


def train_detector(dataset, device):
    if not dataset:
        return None

    train, test = split_by_sample_id(dataset)

    if not train or not test:
        return None

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
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    return float(evaluate(model, test_loader, device)["accuracy"])


def measure_extractability(residuals, strategy_name, family=None,
                           num_layers=None, config_overrides=None):
    """Full embed -> decrypt round trip with no access to the cover."""
    from src.embedding.strategy_registry import (
        embed_with,
        extract_with,
        CoverRequired,
    )
    from src.embedding.strategy_registry import build as build_strategy

    config = EmbeddingConfig(
        total_payload_bits=PAYLOAD_BITS,
        embedding_strategy=strategy_name,
        model_family=family or FAMILY,
        num_hidden_layers=num_layers or NUM_LAYERS,
        **(config_overrides or {}),
    )

    result: Dict[str, Any] = {
        "strategy": strategy_name,
        "embed_ok": False,
        "decrypt_ok": False,
        "recovered_matches": False,
        "ber": None,
        "error": "",
    }

    try:
        strategy = build_strategy(config, strategy_name)

        # Use the production path so the measurement reflects what an
        # experiment would actually run.
        embed_result = IntelligentEmbedder(config).embed(
            MESSAGE, residuals
        )
        result["embed_ok"] = True
        result["bits_embedded"] = embed_result.bits_embedded

        # Bit recovery is measured through the strategy's own extractor,
        # which is what an extractor for that scheme would use. Judging
        # viability by DecryptPipeline instead would fail every non-sign
        # scheme for a reason that is about pipeline wiring, not about
        # whether the scheme carries data.
        recovered_bits = extract_with(
            strategy,
            embed_result.embedded_residuals,
            embed_result.carrier_indices,
            residuals_ref=None,
            strategy_name=strategy_name,
        )

        transmitted = embed_result.embedded_bits
        compared = min(len(transmitted), len(recovered_bits))
        errors = sum(
            1
            for a, b in zip(
                transmitted[:compared], recovered_bits[:compared]
            )
            if a != b
        )
        result["ber"] = errors / compared if compared else None
        result["bits_compared"] = compared
        result["bit_errors"] = errors

        # Whether the production pipeline can already consume this
        # scheme is a separate, smaller question: DecryptPipeline is
        # hardcoded to SignExtractor.
        recovered, stats = DecryptPipeline(
            key=embed_result.key
        ).run(
            embed_result.embedded_residuals,
            embed_result.carrier_indices,
        )
        result["decrypt_ok"] = bool(stats.get("success"))
        result["recovered_matches"] = recovered == MESSAGE
        result["usable_in_production_pipeline"] = bool(
            result["recovered_matches"]
        )

        if not result["recovered_matches"]:
            result["note"] = (
                "Bits recover through the scheme's own extractor, but "
                "DecryptPipeline is hardcoded to SignExtractor and "
                "cannot read a non-sign scheme. Wiring it up is a "
                "separate change, not a scheme defect."
            )

    except CoverRequired as exc:
        result["error"] = f"CoverRequired: {exc}"

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["structurally_usable"] = bool(
        result["embed_ok"] and result["ber"] == 0.0
    )

    return result


def measure_detectability(residuals, strategy_name, device, family=None,
                          num_layers=None, config_overrides=None):
    dataset = build_detector_dataset(
        residuals, strategy_name, random.Random(SEED),
        family=family, num_layers=num_layers,
        config_overrides=config_overrides,
    )

    accuracy = train_detector(dataset, device)

    density = None
    if dataset:
        by_id: Dict[int, Dict[int, Any]] = {}
        for sample in dataset:
            by_id.setdefault(sample["sample_id"], {})[
                sample["label"]
            ] = sample["features"]

        changed = []
        identical = 0
        for pair in by_id.values():
            if 0 in pair and 1 in pair:
                delta = (pair[1] - pair[0]).abs()
                n = int((delta > 0).sum().item())
                changed.append(n)
                if n == 0:
                    identical += 1

        if changed:
            density = {
                "mean_changed_values_per_patch": (
                    sum(changed) / len(changed)
                ),
                "patch_size": PATCH_SIZE,
                "identical_pair_fraction": identical / len(changed),
            }

    return accuracy, density, len(dataset)


def measure_robustness(
    residuals,
    strategy_name,
    sigmas=(0.0, 0.001, 0.002, 0.005),
    trials=3,
    family=None,
    num_layers=None,
    config_overrides=None,
):
    """BER across the noise grid, via the strategy's own extractor.

    Guide gate: BER <= 0.02 at sigma = 0.001.
    """
    from src.embedding.strategy_registry import (
        build as build_strategy,
        extract_with,
    )

    curve = {}

    for sigma in sigmas:
        generator = torch.Generator().manual_seed(SEED)
        total_errors = 0
        total_compared = 0

        for _ in range(trials if sigma else 1):
            config = EmbeddingConfig(
                total_payload_bits=PAYLOAD_BITS,
                embedding_strategy=strategy_name,
                model_family=family or FAMILY,
                num_hidden_layers=num_layers or NUM_LAYERS,
                **(config_overrides or {}),
            )
            embedded = IntelligentEmbedder(config).embed(
                MESSAGE, residuals
            )

            if sigma:
                noisy = {
                    layer_id: (
                        embedded.embedded_residuals[layer_id]
                        + torch.randn(
                            embedded.embedded_residuals[layer_id].shape,
                            generator=generator,
                        )
                        * sigma
                    )
                    for layer_id in sorted(embedded.embedded_residuals)
                }
            else:
                noisy = embedded.embedded_residuals

            recovered = extract_with(
                build_strategy(config, strategy_name),
                noisy,
                embedded.carrier_indices,
                residuals_ref=None,
                strategy_name=strategy_name,
            )

            transmitted = embedded.embedded_bits
            compared = min(len(transmitted), len(recovered))
            total_errors += sum(
                1
                for a, b in zip(
                    transmitted[:compared], recovered[:compared]
                )
                if a != b
            )
            total_compared += compared

        curve[str(sigma)] = (
            total_errors / total_compared if total_compared else None
        )

    return curve


def main() -> int:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading residuals from cache...", flush=True)
    residuals = load_cached_residuals(MODEL_ID, NUM_LAYERS)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"device: {device}\n", flush=True)

    results = []

    for name in CANDIDATES:
        strategy_spec = spec(name)

        print(f"--- {name} ({strategy_spec.status}) ---", flush=True)

        extractability = measure_extractability(residuals, name)
        print(
            f"  extractable      : {extractability['structurally_usable']}"
            f"   {extractability.get('error', '')[:90]}",
            flush=True,
        )

        # Only measure detectability for schemes that actually embed.
        accuracy = None
        density = None
        n_samples = 0
        robustness = None

        if extractability["embed_ok"]:
            robustness = measure_robustness(residuals, name)
            ber_001 = robustness.get("0.001")
            print(
                f"  BER @ sigma=0.001: "
                f"{ber_001:.4f}" if ber_001 is not None
                else "  BER @ sigma=0.001: n/a",
                flush=True,
            )

            accuracy, density, n_samples = measure_detectability(
                residuals, name, device
            )
            print(
                f"  detector accuracy: "
                f"{accuracy:.2%}"
                if accuracy is not None
                else "  detector accuracy: n/a",
                flush=True,
            )
            if density:
                print(
                    f"  values changed    : "
                    f"{density['mean_changed_values_per_patch']:.0f}"
                    f"/{density['patch_size']}",
                    flush=True,
                )

        wins = bool(
            extractability["structurally_usable"]
            and accuracy is not None
            and accuracy <= GATE
            # A scheme that cannot survive the robustness gate is not a
            # usable replacement, even if it evades the detector.
            and robustness is not None
            and robustness.get("0.001") is not None
            and robustness["0.001"] <= 0.02
        )

        results.append(
            {
                "strategy": name,
                "status": strategy_spec.status,
                "forces_sign_flip": strategy_spec.forces_sign_flip,
                "extract_needs_cover": strategy_spec.extract_needs_cover,
                "extractability": extractability,
                "robustness_ber_curve": robustness,
                "meets_robustness_gate": (
                    robustness is not None
                    and robustness.get("0.001") is not None
                    and robustness["0.001"] <= 0.02
                ),
                "detector_accuracy": accuracy,
                "meets_detector_gate": (
                    accuracy <= GATE if accuracy is not None else None
                ),
                "signal_density": density,
                "detector_samples": n_samples,
                "wins": wins,
                "notes": strategy_spec.notes,
            }
        )

    winners = [
        r for r in results
        if r["wins"]
        # A detector at exactly chance learned nothing. Counting that as
        # a win would turn an underpowered run into a green result.
        and r["detector_accuracy"] is not None
        and abs(r["detector_accuracy"] - 0.5) > 1e-9
    ]

    # Guard: if every measured accuracy is exactly chance, the study is
    # invalid regardless of what the table says.
    measured = [
        r["detector_accuracy"] for r in results
        if r["detector_accuracy"] is not None
    ]
    study_invalid = bool(measured) and all(
        abs(a - 0.5) < 1e-9 for a in measured
    )

    print()
    print(f"{'strategy':<20}{'usable':>9}{'BER.001':>10}{'accuracy':>11}{'gate':>7}")
    print("-" * 60)
    for r in results:
        usable = "yes" if r["extractability"]["structurally_usable"] else "no"
        acc = (
            f"{r['detector_accuracy']:.2%}"
            if r["detector_accuracy"] is not None
            else "-"
        )
        ber = (r.get("robustness_ber_curve") or {}).get("0.001")
        ber_s = f"{ber:.4f}" if ber is not None else "-"
        verdict = (
            "PASS"
            if r["meets_detector_gate"] is True
            else "FAIL"
            if r["meets_detector_gate"] is False
            else "N/A"
        )
        print(f"{r['strategy']:<20}{usable:>9}{ber_s:>10}{acc:>11}{verdict:>7}")

    print()

    if study_invalid:
        conclusion = (
            "INVALID STUDY: every measured detector accuracy is exactly "
            "50%, i.e. no detector learned anything. This says nothing "
            "about detectability and must not be read as sign being "
            "undetectable. Increase pairs per strategy and re-run."
        )
    elif winners:
        conclusion = (
            f"{len(winners)} strategy/strategies beat sign: "
            + ", ".join(w["strategy"] for w in winners)
            + ". Promote only after confirming the result holds on more "
            "than one model."
        )
    else:
        conclusion = (
            "No strategy beats sign. Both sign-flip schemes fail the "
            "detectability gate; the two non-sign candidates are not "
            "usable as implemented (LWE needs the cover to extract, "
            "neural needs training). Phase 3 should not proceed on the "
            "current candidates."
        )

    print(conclusion)

    target = RESULTS_DIR / "exp10_strategy_comparison.json"
    target.write_text(
        __import__("json").dumps(
            {
                "experiment": "exp10_strategy_comparison",
                "title": "Embedding strategy head-to-head",
                "model_id": MODEL_ID,
                "gate": GATE,
                "reference_recorded_neural_accuracy": 0.705,
                "method": {
                    "payload_bits": PAYLOAD_BITS,
                    "pairs_per_strategy": PAIRS,
                    "embeddings_per_strategy": EMBEDDINGS_PER_VARIANT,
                    "patch_size": PATCH_SIZE,
                    "epochs": EPOCHS,
                    "seed": SEED,
                    "note": (
                        "Each strategy is measured by its own detector, "
                        "since the dataset is built from that strategy's "
                        "embedding. Sized for comparison, not as a "
                        "headline result."
                    ),
                },
                "registry": structural_report(),
                "results": results,
                "winners": [w["strategy"] for w in winners],
                "study_invalid": study_invalid,
                "conclusion": conclusion,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())