"""
exp27 — threat-model boundary (W9.3).

One production embed (sign, 10,000 bits), four conditions around it:

    C1 control         full carrier map + correct key  → must recover
    C2 wrong key       carrier map + 10 distinct wrong keys → 0
                       recoveries, no plaintext ever emitted
                       (AES-GCM authentication is the claim; the
                       count is the evidence). Raw-bit readability
                       with the map is measured separately: the
                       CHANNEL may be readable while the MESSAGE is
                       not — that split is the boundary.
    C3 partial access  only the first 50/25/10% of layers, and a
                       seeded scattered 50/25/10% subset, with the
                       correct key and full carrier map → 0 message
                       recoveries; stream coverage and BER-on-
                       available-bits reported per cell.
    C4 public rule     attacker re-runs the public QACI selection on
                       the stego residuals (exp13's Kerckhoffs shape
                       via exp25's sign implementation) — channel
                       readability measured; message still needs the
                       key, which C2 shows never works.

Gate: experiment_registry.THRESHOLDS['exp27'] — control recovers at
BER 0.0, >=10 wrong keys tested with 0 recoveries, 0 partial-access
recoveries. Channel-level readability is measured and reported, not
gated: this experiment draws the CONFIDENTIALITY boundary.

Usage
-----
    python -m src.experiments.exp27_threat_model_boundary [--model <id>]
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from src.core.types import EmbeddingConfig  # noqa: E402
from src.embedding.intelligent_embedder import IntelligentEmbedder  # noqa: E402
from src.embedding.strategy_registry import (  # noqa: E402
    build as build_strategy,
    extract_with,
)
from src.experiments.exp15_lwe_fidelity import _context_for, _slug  # noqa: E402
from src.experiments.exp25_selection_ablation import (  # noqa: E402
    _ber,
    keyless_qaci_read,
)
from src.experiments.experiment_registry import THRESHOLDS  # noqa: E402
from src.experiments.paths import RESULTS_DIR  # noqa: E402
from src.experiments.residual_source import load_cached_residuals  # noqa: E402
from src.extraction.decrypt_pipeline import DecryptPipeline  # noqa: E402

EXPERIMENT = "exp27"
STRATEGY = "sign"            # production's own scheme
PAYLOAD_BITS = 10_000        # exp10/exp13/exp23's payload
MESSAGE = "A" * 1_250        # 10,000 bits of UTF-8 'A'
SEED = 42
WRONG_KEYS = 10
PARTIAL_FRACTIONS: Tuple[float, ...] = (0.5, 0.25, 0.1)


def log(message: str) -> None:
    print(f"[{EXPERIMENT}] {message}", flush=True)


def wrong_keys(n: int = WRONG_KEYS) -> List[bytes]:
    """Deterministic distinct 32-byte wrong keys (seeded by label)."""
    return [
        hashlib.sha256(f"exp27-wrong-key-{i}".encode()).digest()
        for i in range(n)
    ]


def partial_layer_sets(
    layers: List[int], fraction: float
) -> Dict[str, List[int]]:
    """Prefix (best case: lowest layers leaked first) and a seeded
    scattered subset, both sorted like the extractor walks them."""
    k = max(1, int(len(layers) * fraction))
    tag = int(fraction * 100)
    scattered = sorted(
        random.Random(SEED + tag).sample(layers, k)
    )
    return {
        f"prefix_{tag}pct": list(layers[:k]),
        f"scattered_{tag}pct": scattered,
    }


def stream_slices(carriers: Dict[int, List[int]]) -> Dict[int, Tuple[int, int]]:
    """True stream offsets: base_embedder writes layers in sorted order."""
    offsets: Dict[int, Tuple[int, int]] = {}
    pos = 0
    for lid in sorted(carriers):
        offsets[lid] = (pos, pos + len(carriers[lid]))
        pos += len(carriers[lid])
    return offsets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B",
        help="model id whose residual cache is complete "
             "(default: Qwen2.5-3B, exp3/exp10's anchor)",
    )
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    context = _context_for(args.model)
    log(f"Model: {args.model} (family {context.family}, "
        f"{context.expected_layers} layers)")

    log("loading cached residuals ...")
    residuals = load_cached_residuals(args.model, context.expected_layers)

    config = EmbeddingConfig(
        total_payload_bits=PAYLOAD_BITS,
        embedding_strategy=STRATEGY,
        model_family=context.family,
        num_hidden_layers=(
            context.actual_layers or context.expected_layers
        ),
    )

    log("one production embed ...")
    result = IntelligentEmbedder(config).embed(MESSAGE, residuals)
    emb_r = result.embedded_residuals
    carriers = result.carrier_indices
    transmitted = result.embedded_bits
    strategy = build_strategy(config, STRATEGY)
    offsets = stream_slices(carriers)
    total_stream = sum(len(v) for v in carriers.values())
    layers = sorted(carriers)

    # -- C1 control ---------------------------------------------------
    log("C1: control (full access + correct key) ...")
    control_msg, control_stats = DecryptPipeline(key=result.key).run(
        emb_r, carriers
    )
    control_rec = extract_with(
        strategy, emb_r, carriers,
        residuals_ref=None, strategy_name=STRATEGY,
    )
    control_ber, control_errors, control_n = _ber(transmitted, control_rec)

    # -- C2 wrong key -------------------------------------------------
    log(f"C2: {WRONG_KEYS} wrong keys ...")
    wrong_results = []
    for wk in wrong_keys():
        msg, stats = DecryptPipeline(key=wk).run(emb_r, carriers)
        error = str(stats.get("error", ""))
        if "Header decode" in error:
            kind = "header"
        elif "Decryption failed" in error:
            kind = "gcm_authentication"
        elif stats.get("success"):
            kind = "none_success"
        else:
            kind = "other"
        wrong_results.append({
            "recovered": bool(stats.get("success")),
            "emitted_plaintext": msg == MESSAGE,
            "error_kind": kind,
        })
    wrong_recoveries = sum(r["recovered"] for r in wrong_results)
    plaintext_emitted = sum(r["emitted_plaintext"] for r in wrong_results)
    # Channel readability WITH the map but WITHOUT the key:
    channel_ber_map_no_key = _ber(transmitted, control_rec)[0]

    # -- C3 partial access -------------------------------------------
    log("C3: partial model access ...")
    partial_cells = []
    for fraction in PARTIAL_FRACTIONS:
        for name, subset in partial_layer_sets(layers, fraction).items():
            sub_r = {lid: emb_r[lid] for lid in subset}
            sub_c = {lid: carriers[lid] for lid in subset}
            sub_msg, sub_stats = DecryptPipeline(key=result.key).run(
                sub_r, sub_c
            )
            sub_rec = extract_with(
                strategy, sub_r, sub_c,
                residuals_ref=None, strategy_name=STRATEGY,
            )
            # Expected bits: each available layer's true stream slice,
            # concatenated in the extractor's walk order.
            expected: List[int] = []
            for lid in subset:
                start, end = offsets[lid]
                expected.extend(transmitted[start:end])
            bits_available = sum(len(v) for v in sub_c.values())
            sub_ber, sub_errors, sub_n = _ber(expected, sub_rec)
            cell = {
                "condition": name,
                "fraction_requested": fraction,
                "layers_available": len(subset),
                "layers_total": len(layers),
                "bits_available": bits_available,
                "stream_coverage": (
                    bits_available / total_stream if total_stream else None
                ),
                "ber_on_available_bits": sub_ber,
                "bit_errors_on_available": sub_errors,
                "decrypt_ok": bool(sub_stats.get("success")),
                "recovered_matches": sub_msg == MESSAGE,
                "error": str(sub_stats.get("error", "")) or None,
            }
            partial_cells.append(cell)
            log(f"  {name}: coverage={cell['stream_coverage']:.3f} "
                f"ber_available={sub_ber} "
                f"decrypt_ok={cell['decrypt_ok']}")
    partial_recoveries = sum(c["recovered_matches"] for c in partial_cells)

    # -- C4 public-rule attacker --------------------------------------
    log("C4: keyless attacker re-runs the public QACI rule ...")
    keyless_exact = keyless_qaci_read(
        emb_r, carriers, transmitted, config, len(transmitted)
    )
    keyless_nominal = keyless_qaci_read(
        emb_r, carriers, transmitted, config, config.total_payload_bits
    )

    # -- gate ---------------------------------------------------------
    gate = dict(THRESHOLDS[EXPERIMENT])
    failures = []
    if control_ber != 0.0 or not control_stats.get("success"):
        failures.append(
            f"control ber={control_ber} success="
            f"{control_stats.get('success')}"
        )
    if len(wrong_results) < gate["min_wrong_keys_tested"]:
        failures.append(
            f"only {len(wrong_results)} wrong keys tested"
        )
    if wrong_recoveries > gate["max_wrong_key_recoveries"]:
        failures.append(f"wrong-key recoveries={wrong_recoveries}")
    if partial_recoveries > gate["max_partial_access_recoveries"]:
        failures.append(f"partial-access recoveries={partial_recoveries}")
    status = "PASS" if not failures else "FAIL"

    artifact = {
        "experiment": EXPERIMENT,
        "title": "Threat-model boundary (W9.3)",
        "model_id": args.model,
        "family": context.family,
        "status": status,
        "gate": {
            **gate,
            "gate_source": (
                f"experiment_registry.THRESHOLDS['{EXPERIMENT}']"
            ),
            "failures": failures,
            "measured": {
                "control_ber": control_ber,
                "control_recovered": bool(control_stats.get("success")),
                "wrong_keys_tested": len(wrong_results),
                "wrong_key_recoveries": wrong_recoveries,
                "wrong_key_plaintext_emitted": plaintext_emitted,
                "partial_access_recoveries": partial_recoveries,
            },
        },
        "conditions": {
            "control": {
                "has": ["carrier map", "correct key", "all layers"],
                "ber": control_ber,
                "bit_errors": control_errors,
                "bits_compared": control_n,
                "recovered_matches": control_msg == MESSAGE,
            },
            "wrong_key": {
                "has": ["carrier map", "WRONG key", "all layers"],
                "keys_tested": len(wrong_results),
                "recoveries": wrong_recoveries,
                "plaintext_emitted": plaintext_emitted,
                "error_kinds": {
                    kind: sum(
                        r["error_kind"] == kind for r in wrong_results
                    )
                    for kind in sorted(
                        {r["error_kind"] for r in wrong_results}
                    )
                },
                "channel_ber_with_map_no_key": channel_ber_map_no_key,
            },
            "partial_access": {
                "has": [
                    "carrier map", "correct key", "subset of layers"
                ],
                "cells": partial_cells,
                "recoveries": partial_recoveries,
            },
            "public_rule_keyless": {
                "has": [
                    "released weights only (algorithm and payload "
                    "size are not secrets)"
                ],
                "exact_size": keyless_exact,
                "nominal_size": keyless_nominal,
                "message_recoverable": False,
                "message_protection_basis": (
                    "C2: 0 of "
                    f"{len(wrong_results)} keys that are not the "
                    "embedding key decrypt — GCM authentication fails "
                    "before any plaintext can be produced. An "
                    "attacker who reads the channel without the key "
                    "holds ciphertext only."
                ),
            },
        },
        "method": {
            "seed": SEED,
            "one_embed": (
                "Every condition reads the SAME production embed "
                "(exp23's paired design): conditions differ only in "
                "what the reader is allowed to hold."
            ),
            "partial_access_protocol": (
                "prefix = lowest layer ids first (contiguous stream "
                "prefix), scattered = seeded sample; BER on available "
                "bits compares against each layer's TRUE stream slice "
                "(stream order is sorted layers, base_embedder)."
            ),
            "public_rule_attacker": (
                "exp25.keyless_qaci_read — QACI re-run on the stego "
                "residuals (exp13's tier_kerckhoffs shape, sign read)."
            ),
            "pre_registered": (
                "Gates in THRESHOLDS['exp27'] fixed before this run; "
                "channel readability per condition is reported, not "
                "gated — the gate covers message confidentiality."
            ),
        },
        "notes": [
            "Boundary in one line: MESSAGE confidentiality holds at "
            "every access level without the key (C2/C3/C4); CHANNEL "
            "readability is a separate, measured axis and is NOT "
            "claimed here (exp13 owns the keyless-read FAIL, exp25 "
            "attributes it to selection policy).",
            "C3 is all-or-nothing for message recovery by "
            "construction: AES-GCM over a stream with missing layers "
            "cannot authenticate. The cell-level numbers say WHERE "
            "partial access stops paying (coverage and clean-bit "
            "reads), which is the boundary a reviewer can probe.",
            "exp13's phase-tier attacker remains out of scope (it "
            "needs no parameters and is measured on exp13's LWE "
            "embed); this experiment measures conditions around ONE "
            "production embed.",
        ],
        "reproducibility": context.reproducibility(),
    }

    target = (
        RESULTS_DIR
        / f"{EXPERIMENT}_threat_model_{_slug(args.model)}.json"
    )
    target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    log(f"wrote {target} — gate: {status}"
        + (f" — {failures}" if failures else ""))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
