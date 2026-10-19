"""
W5.2 — QAE encode + LWE read-out (exp21).

RESEARCH_PLAN §4 W5.2: *"QAE encode + LWE read-out. Quantization-aware
placement, parity decode. Plausible: both mechanisms avoid sign flips."*

The premise is half-false as written — exp17 established QAE-V1 DOES
force sign flips (`+max(|v|, margin)` / `-max(|v|, margin)`, sign-family
with a margin floor). So the question this experiment actually asks:
**can the LWE parity read-out decode a QAE-encoded stream at all?**

One embed, two read-outs, nothing else varying:

    matched    the same stego read through qae's own extractor
               (SignExtractor) — the in-run control: must be 0.0,
               or a failure below would say nothing about pairing.
    interop    the same stego read through LWEStrategy's parity
               read-out (`bit = floor(v / w) % 2`, w = 0.010 from
               DEFAULT_GRID_WIDTH, stego-std independent).

Prediction written before the run: qae's values are +/-max(|v|,
margin) with |v| mostly below one grid width (residual std ~0.002,
w = 0.010), so floor(v/w) is 0 for every positive value and -1 for
every negative one — parity mirrors sign below the width, making the
LWE read-out the exact COMPLEMENT of the qae stream: BER near 1.0,
not chance.

First-run result: 0.5433 — near chance, prediction MISSED. The
correction: QACI selects the TOP-magnitude tail, and the top-tail
carriers measured on this model are all above one grid width
(sampled top-tail min 0.013 > w=0.010, median 0.021), so no
sub-width complement regime exists. What does hold is the identity
parity(v) = sign(v) XOR cell-parity(|v|) for non-multiples of w:
the raw interop read-out is the sign bit XOR a per-carrier constant
derivable from the STEGO magnitudes alone. Read-out C applies that
public correction — if it returns exactly 0.0, the structure
(parity read-out is sign reading plus a public relabeling, i.e. the
two mechanisms are not independent channels) is verified, not
asserted.

Gate: THRESHOLDS['exp21'] — both readings at exp3's 0.0 (reused).
The verdict stays the RAW interop's: read-out C is a structural
finding, never a rescue of the plan's claim as written.

Usage:
    python -m src.experiments.exp21_qae_lwe_readout --model <id>
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from src.core.types import EmbeddingConfig  # noqa: E402
from src.embedding.intelligent_embedder import IntelligentEmbedder  # noqa: E402
from src.embedding.strategy_registry import (  # noqa: E402
    build as build_strategy,
)
from src.embedding.strategy_registry import extract_with  # noqa: E402
from src.experiments.experiment_registry import gate_for  # noqa: E402
from src.experiments.exp10_strategy_comparison import (  # noqa: E402
    MESSAGE,
    PAYLOAD_BITS,
)
from src.experiments.exp15_lwe_fidelity import _context_for, _slug  # noqa: E402
from src.experiments.paths import RESULTS_DIR  # noqa: E402
from src.experiments.residual_source import load_cached_residuals  # noqa: E402

EXPERIMENT = "exp21"

PLAN_QUOTE = (
    "W5.2: QAE encode + LWE read-out. Quantization-aware placement, "
    "parity decode. Plausible: both mechanisms avoid sign flips."
)

PREMISE_STATUS = (
    "half-false as written (exp17): QAE-V1 forces sign flips — "
    "+max(|v|, margin) / -max(|v|, margin), sign-family with a "
    "margin floor. The question is therefore whether parity "
    "read-out decodes a sign-family stream at all."
)


def log(message: str) -> None:
    """Flush every stage — a long MPS run must never look hung (§3.7)."""
    print(message, flush=True)


def ber_against(transmitted, recovered) -> Dict[str, Any]:
    compared = min(len(transmitted), len(recovered))
    errors = sum(
        1
        for a, b in zip(transmitted[:compared], recovered[:compared])
        if a != b
    )
    return {
        "ber": errors / compared if compared else None,
        "bits_compared": compared,
        "bit_errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B",
        help="model id whose residual cache is complete "
             "(default: Qwen2.5-3B, exp17's model)",
    )
    args = parser.parse_args()

    context = _context_for(args.model)
    log(f"Model: {args.model} "
        f"(family {context.family}, {context.expected_layers} layers)")
    log("Loading residuals from cache ...")
    residuals = load_cached_residuals(
        args.model, context.expected_layers
    )

    # --- one embed through the production path ----------------------
    config = EmbeddingConfig(
        total_payload_bits=PAYLOAD_BITS,
        embedding_strategy="qae",
        model_family=context.family,
        num_hidden_layers=context.expected_layers,
    )
    qae = build_strategy(config, "qae")
    log("[exp21] embed via production path (strategy=qae) ...")
    result = IntelligentEmbedder(config).embed(MESSAGE, residuals)
    transmitted = result.embedded_bits
    log(f"  embedded {result.bits_embedded}/{result.total_bits} bits, "
        f"transmitted {len(transmitted)}")

    # --- read 1: matched (in-run control) ---------------------------
    log("[exp21] read-out A: matched (qae's own extractor) ...")
    matched = extract_with(
        qae,
        result.embedded_residuals,
        result.carrier_indices,
        residuals_ref=None,
        strategy_name="qae",
    )
    control = ber_against(transmitted, matched)
    log(f"  control ber={control['ber']} "
        f"({control['bit_errors']}/{control['bits_compared']})")

    # --- read 2: interop (LWE parity read-out on the same stream) ---
    log("[exp21] read-out B: LWE parity read-out on the same stream ...")
    lwe_config = EmbeddingConfig(
        total_payload_bits=PAYLOAD_BITS,
        embedding_strategy="lwe",
        model_family=context.family,
        num_hidden_layers=context.expected_layers,
    )
    lwe = build_strategy(lwe_config, "lwe")
    interop = extract_with(
        lwe,
        result.embedded_residuals,
        result.carrier_indices,
        residuals_ref=None,
        strategy_name="lwe",
    )
    pair = ber_against(transmitted, interop)
    log(f"  interop ber={pair['ber']} "
        f"({pair['bit_errors']}/{pair['bits_compared']})")

    # --- read 3: the identity, applied publicly ---------------------
    # parity(v) = sign(v) XOR cell-parity(|v|) for non-multiples of
    # w, so the raw read-out XOR'd with floor(|stego|/w)%2 XOR 1
    # should return the transmitted bits EXACTLY — using only what
    # an extractor holds (stego + the public width).
    log("[exp21] read-out C: public cell-parity correction of B ...")
    import math

    corrected = []
    for layer_id in sorted(result.carrier_indices):
        flat = result.embedded_residuals[layer_id].flatten()
        for pos in result.carrier_indices[layer_id]:
            r = interop[len(corrected)]
            pf = math.floor(abs(flat[pos].item()) / 0.010) % 2
            corrected.append(r ^ pf ^ 1)
    fixed = ber_against(transmitted, corrected)
    log(f"  corrected ber={fixed['ber']} "
        f"({fixed['bit_errors']}/{fixed['bits_compared']})")

    # --- verdict ----------------------------------------------------
    gate = gate_for(EXPERIMENT)
    control_ok = bool(
        control["ber"] is not None
        and control["ber"] == gate["max_control_ber"]
    )
    interop_ok = bool(
        pair["ber"] is not None and pair["ber"] == gate["max_ber"]
    )

    artifact: Dict[str, Any] = {
        "experiment": EXPERIMENT,
        "title": "W5.2 — QAE encode + LWE read-out (interop)",
        "model_id": args.model,
        "family": context.family,
        "num_layers": context.expected_layers,
        "gate": {
            **gate,
            "gate_source": "experiment_registry.THRESHOLDS['exp21']",
        },
        "readouts": {
            "matched_control": {
                **control,
                "extractor": "SignExtractor (qae's own)",
                "meets_gate": control_ok,
            },
            "lwe_interop": {
                **pair,
                "extractor": (
                    "LWEStrategy.extract — floor(v/0.010)%2, "
                    "DEFAULT_GRID_WIDTH, stego statistics"
                ),
                "meets_gate": interop_ok,
            },
            "lwe_interop_corrected": {
                **fixed,
                "adapter": (
                    "raw read XOR floor(|stego|/0.010)%2 XOR 1 — "
                    "the parity(v) = sign(v) XOR cell-parity(|v|) "
                    "identity, using stego magnitudes only"
                ),
                "meets_gate": bool(
                    fixed["ber"] is not None
                    and fixed["ber"] == gate["max_ber"]
                ),
                "role": (
                    "structural finding, NOT a gate rescue: the "
                    "verdict below is the raw interop's"
                ),
            },
        },
        "verdict": "PASS" if (control_ok and interop_ok) else "FAIL",
        "premise": {
            "plan_quote": PLAN_QUOTE,
            "status": PREMISE_STATUS,
        },
        "method": {
            "embed": (
                "single production-path embed (IntelligentEmbedder, "
                "strategy=qae); both read-outs decode the SAME stego "
                "so the read-out pairing is the only variable"
            ),
            "payload_bits": PAYLOAD_BITS,
            "message_len": len(MESSAGE),
            "model_note": "exp17's model (same-model comparability)",
        },
        "notes": [
            "The matched control at 0.0 is what makes a non-zero "
            "interop BER attributable to the pairing rather than a "
            "broken embed (exp17 measured this path at 0.0 over "
            "48,256 bits).",
            "A FAIL here is the plan's plausibility claim failing, "
            "recorded with its number — exp13's pattern. Neither "
            "threshold is ever relaxed.",
            "Pre-registered prediction (complement, BER near 1.0) "
            "MISSED: measured near-chance because carriers are the "
            "top-magnitude tail, all above one grid width. The miss "
            "and its correction are recorded, not quietly rewritten.",
            "Read-out C returning 0.0 (if it does) means the parity "
            "channel is publicly re-encodable to the sign bit: the "
            "'hybrid' would be sign reading with a relabeling, not "
            "a second independent channel.",
        ],
    }

    target = RESULTS_DIR / f"{EXPERIMENT}_qae_lwe_{_slug(args.model)}.json"
    target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    log("=" * 70)
    log(f"control (matched): ber={control['ber']}")
    log(f"interop (lwe read): ber={pair['ber']}")
    log(f"gate: control <= {gate['max_control_ber']} -> {control_ok}, "
        f"interop <= {gate['max_ber']} -> {interop_ok}")
    log(f"verdict: {artifact['verdict']}")
    log(f"wrote {target}")

    del residuals
    torch.mps.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
