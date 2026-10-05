"""
W5.1 — the adaptive meta-strategy as designed (exp19).

RESEARCH_PLAN §3 W5.1: *"adaptive_strategy as designed. Noise-threshold
routing to LWE/neural/sign. Cheap. Tests someone else's design and gives
a baseline for anything better."*

AdaptiveStrategy estimates sigma from first-order residual differences
(median |delta| / sqrt(2), median across layers) and routes:

    sigma < 0.0005  ->  lwe      (ultra-low noise: fidelity)
    sigma < 0.003   ->  neural   (moderate: learned robustness)
    else            ->  sign     (high noise: robustness)

What this measures, per model:

    1. the design's actual routing decision on real cached residuals
       (sigma estimate and the branch it selects);
    2. a round trip through the ROUTED branch — embed via the
       production path (IntelligentEmbedder with the new registry
       entry), extract via AdaptiveRoutedExtractor, which fetches the
       selected branch's own companion decoder;
    3. if the route selects an unavailable branch (neural without a
       trained model raises EmbeddingError), the failure is recorded
       as the design's own, and the available branches (sign, lwe)
       are round-tripped as the design's fallback baseline.

A preliminary probe found the three first-pass models route to three
DIFFERENT branches (gemma-2-2b -> lwe, Qwen2.5-3B -> neural,
Llama-3.1-8B -> sign), so this is not a one-branch measurement.

Scope: the design's routing and round trip only. No detector axis
(exp18 owns detectability for the routed primitives), no perplexity.
The lwe branch builds LWEStrategy with a fresh random secret_key per
run — the round trip is self-consistent, but its encoding is not
bit-identical to the registry's zero-key lwe.

Gate: THRESHOLDS['exp19'].max_ber = 0.0 — exp3's number, reused.
Routing choice itself is measurement, not gate.

Usage:
    python -m src.experiments.exp19_adaptive_routing --model <id>
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

from src.core.exceptions import EmbeddingError  # noqa: E402
from src.core.types import EmbeddingConfig  # noqa: E402
from src.embedding.intelligent_embedder import IntelligentEmbedder  # noqa: E402
from src.embedding.strategy_registry import extract_with  # noqa: E402
from src.experiments.experiment_registry import gate_for  # noqa: E402
from src.experiments.exp15_lwe_fidelity import _context_for, _slug  # noqa: E402
from src.experiments.paths import RESULTS_DIR  # noqa: E402
from src.experiments.residual_source import load_cached_residuals  # noqa: E402

EXPERIMENT = "exp19"

PAYLOAD_BITS = 10_000
MESSAGE = "A" * 1_250
FALLBACK_BRANCHES = ("sign", "lwe")


def log(message: str) -> None:
    """Flush every stage — a long MPS run must never look hung (§3.7)."""
    print(message, flush=True)


def _round_trip(ie, result, strategy_name: str) -> Dict[str, Any]:
    """exp10's honest BER: compare against the transmitted bits."""
    recovered = extract_with(
        ie.strategy,
        result.embedded_residuals,
        result.carrier_indices,
        residuals_ref=None,
        strategy_name=strategy_name,
    )
    transmitted = result.embedded_bits
    compared = min(len(transmitted), len(recovered))
    errors = sum(
        1
        for a, b in zip(transmitted[:compared], recovered[:compared])
        if a != b
    )
    ber = errors / compared if compared else 1.0

    out: Dict[str, Any] = {
        "ber": ber,
        "bits_compared": compared,
        "bit_errors": errors,
        "pipeline_attempted": False,
        "pipeline_ok": None,
        "pipeline_note": "",
    }

    # DecryptPipeline is hardcoded to SignExtractor (exp10's recorded
    # wiring note): running it on a non-sign branch would misdecode and
    # prove nothing about the scheme, so it is attempted only where it
    # is the right decoder.
    if strategy_name in ("sign", "adaptive"):
        selected = getattr(ie.strategy, "selected_strategy", None)
        if strategy_name == "adaptive" and selected != "sign":
            out["pipeline_note"] = (
                "not attempted: routed branch is not sign; "
                "DecryptPipeline is sign-only (exp10's wiring note)"
            )
        else:
            from src.extraction.decrypt_pipeline import DecryptPipeline

            recovered_msg, stats = DecryptPipeline(key=result.key).run(
                result.embedded_residuals,
                result.carrier_indices,
            )
            out["pipeline_attempted"] = True
            out["pipeline_ok"] = bool(stats.get("success")) and (
                recovered_msg == MESSAGE
            )
    else:
        out["pipeline_note"] = (
            "not attempted: DecryptPipeline is sign-only "
            "(exp10's wiring note)"
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    context = _context_for(args.model)
    log(f"Model: {args.model} "
        f"(family {context.family}, {context.expected_layers} layers)")
    log("Loading residuals from cache ...")
    residuals = load_cached_residuals(
        args.model, context.expected_layers
    )

    def config_for(strategy: str) -> EmbeddingConfig:
        return EmbeddingConfig(
            total_payload_bits=PAYLOAD_BITS,
            embedding_strategy=strategy,
            model_family=context.family,
            num_hidden_layers=context.expected_layers,
        )

    gate = gate_for(EXPERIMENT)
    round_trips: Dict[str, Dict[str, Any]] = {}
    forced: list = []

    # --- 1. The design as designed: adaptive route -------------------
    log("[exp19] adaptive round trip ...")
    ie = IntelligentEmbedder(config_for("adaptive"))
    routing: Dict[str, Any] = {
        "estimated_sigma": None,
        "selected_branch": None,
        "thresholds": {
            "lwe": ie.strategy.NOISE_THRESHOLD_LWE,
            "neural": ie.strategy.NOISE_THRESHOLD_NEURAL,
        },
        "design_route_available": False,
        "failure": None,
    }
    try:
        result = ie.embed(MESSAGE, residuals)
        # IntelligentEmbedder's EmbedResult wrapper does not carry the
        # inner EmbeddingResult's metadata (where AdaptiveStrategy
        # parked estimated_sigma), so the estimate is recomputed from
        # the same residuals — estimate_noise is a deterministic
        # median statistic, so the value is identical to the one the
        # routing decision used.
        routing["estimated_sigma"] = ie.strategy.estimate_noise(residuals)
        routing["selected_branch"] = ie.strategy.selected_strategy
        routing["design_route_available"] = True
        round_trips[f"adaptive->{routing['selected_branch']}"] = _round_trip(
            ie, result, "adaptive"
        )
        log(f"  routed to {routing['selected_branch']} "
            f"(sigma={routing['estimated_sigma']}) -> round trip done")
    except EmbeddingError as exc:
        routing["estimated_sigma"] = ie.strategy.estimate_noise(residuals)
        routing["selected_branch"] = ie.strategy.selected_strategy
        routing["design_route_available"] = False
        routing["failure"] = f"{type(exc).__name__}: {exc}"
        log(f"  ROUTING FAILURE: sigma={routing['estimated_sigma']} "
            f"-> {routing['selected_branch']} -> {exc}")

        # The design's own available branches, as its fallback baseline.
        for branch in FALLBACK_BRANCHES:
            log(f"  forced fallback '{branch}' ...")
            forced_ie = IntelligentEmbedder(config_for(branch))
            res = forced_ie.embed(MESSAGE, residuals)
            round_trips[f"forced->{branch}"] = _round_trip(
                forced_ie, res, branch
            )
            forced.append(branch)
            log(f"    forced {branch}: "
                f"ber={round_trips[f'forced->{branch}']['ber']}")

    # --- 2. Verdict --------------------------------------------------
    measured = [t["ber"] for t in round_trips.values()]
    passed = bool(measured) and all(
        b <= gate["max_ber"] for b in measured
    )

    artifact: Dict[str, Any] = {
        "experiment": EXPERIMENT,
        "title": "W5.1 — adaptive meta-strategy as designed",
        "model_id": args.model,
        "family": context.family,
        "num_layers": context.expected_layers,
        "gate": {
            "max_ber": gate["max_ber"],
            "measured_round_trips": measured,
            "status": "PASS" if passed else "FAIL",
            "gate_source": "experiment_registry.THRESHOLDS['exp19']",
        },
        "routing": routing,
        "round_trips": round_trips,
        "forced_fallbacks": forced,
        "method": {
            "estimator": (
                "AdaptiveStrategy.estimate_noise: median |delta| of "
                "adjacent residuals / sqrt(2), median across layers"
            ),
            "routing_rule": (
                "sigma < 0.0005 -> lwe; sigma < 0.003 -> neural; "
                "else sign (AdaptiveStrategy class constants)"
            ),
            "embed_path": (
                "production: IntelligentEmbedder with the new "
                "registry entry 'adaptive'"
            ),
            "extract_path": (
                "extract_with(strategy_name=...) -> "
                "AdaptiveRoutedExtractor -> the selected branch's "
                "own companion decoder, cover withheld (residuals_ref "
                "= None)"
            ),
            "payload_bits": PAYLOAD_BITS,
            "message_len": len(MESSAGE),
            "sigma_recompute": (
                "estimated_sigma is recomputed via estimate_noise after "
                "embed: IntelligentEmbedder's EmbedResult wrapper drops "
                "the inner result's metadata (deterministic median "
                "statistic, so the value equals the routing input)"
            ),
            "lwe_key": (
                "fresh os.urandom(32) per run (the design's own "
                "constructor); self-consistent round trip, not "
                "bit-identical to registry lwe"
            ),
        },
        "notes": [
            "Routing choice is measurement, not gate — the gate "
            "covers only round trips that run.",
            "A routing failure (neural branch without a trained "
            "model) is the design's own outcome, recorded, never "
            "patched: the design says train a model and pass "
            "neural_model_path.",
            "No detector/perplexity axis here — exp18 owns "
            "detectability of the routed primitives.",
        ],
    }

    target = RESULTS_DIR / f"{EXPERIMENT}_adaptive_{_slug(args.model)}.json"
    target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    log("=" * 70)
    log(f"sigma={routing['estimated_sigma']} -> "
        f"{routing['selected_branch']} "
        f"(available={routing['design_route_available']})")
    for name, t in round_trips.items():
        log(f"{name:28s} ber={t['ber']}")
    log(f"gate (<= {gate['max_ber']}): {artifact['gate']['status']}")
    log(f"wrote {target}")

    del residuals
    torch.mps.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
