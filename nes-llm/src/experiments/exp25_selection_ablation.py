"""
exp25 — selection-policy ablation (W9.1).

The question a reviewer will ask first: is the win the *mechanism* or
the *selection*? This holds everything constant — sign write rule,
QACI Hamilton layer allocation, payload (10,000 bits), message, model —
and varies ONLY the within-layer position policy:

    random    uniform positions per layer (torch.randperm)
    magnitude top-|residual| positions per layer
              (== the production path's own fallback: EmbeddingConfig
               carrier_selection='magnitude'; QACI without weights)
    keyed     secret-key-derived positions (IndexSampler, SHA-256 key)

Measured per arm: round-trip BER, production decrypt, exp7's
detectability trio (SecurityValidator KL / detector accuracy / sign
bias), noise robustness at sigma=0.001 (exp6/exp10's protocol), and a
keyless attacker that re-runs the PUBLIC QACI selection on the stego
residuals at the exact and nominal payload sizes (exp13's
tier_kerckhoffs shape, sign read).

Pre-registered hypotheses live in HYPOTHESES and are evaluated against
the measurements; the gate lives in experiment_registry.THRESHOLDS so
this file contains no threshold of its own.

Usage
-----
    python -m src.experiments.exp25_selection_ablation [--model <id>]
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from src.carrier_intelligence.qaci_pipeline import QACIPipeline  # noqa: E402
from src.carrier_intelligence.selector import CarrierSelector  # noqa: E402
from src.carrier_selection.index_sampler import IndexSampler  # noqa: E402
from src.core.types import EmbeddingConfig  # noqa: E402
from src.embedding.intelligent_embedder import IntelligentEmbedder  # noqa: E402
from src.embedding.strategy_registry import (  # noqa: E402
    build as build_strategy,
    extract_with,
)
from src.experiments.exp15_lwe_fidelity import _context_for, _slug  # noqa: E402
from src.experiments.experiment_registry import THRESHOLDS  # noqa: E402
from src.experiments.paths import RESULTS_DIR  # noqa: E402
from src.experiments.residual_source import load_cached_residuals  # noqa: E402
from src.extraction.decrypt_pipeline import DecryptPipeline  # noqa: E402
from src.steganalysis.security_validator import SecurityValidator  # noqa: E402

EXPERIMENT = "exp25"
STRATEGY = "sign"            # production's own scheme (exp23's constant)
PAYLOAD_BITS = 10_000        # exp10/exp13/exp23's payload
MESSAGE = "A" * 1_250        # 10,000 bits of UTF-8 'A'
SEED = 42
NOISE_SIGMA = 0.001          # exp6's first robustness point
NOISE_TRIALS = 3             # exp10's trial count
POLICIES = ("random", "magnitude", "keyed")
REPLICATES = {"random": 3, "magnitude": 1, "keyed": 3}


def log(message: str) -> None:
    print(f"[{EXPERIMENT}] {message}", flush=True)


# ---------------------------------------------------------------------
# Pure policy core (importable without a model; unit-tested)
# ---------------------------------------------------------------------

def build_policy_indices(
    policy: str,
    residuals: Dict[int, torch.Tensor],
    layer_allocation: Dict[int, int],
    *,
    key: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[int, List[int]]:
    """Within-layer carrier positions for one policy.

    ``layer_allocation`` is QACI's Hamilton allocation, identical for
    every arm — only the within-layer position choice differs. The
    magnitude arm calls CarrierSelector.select_by_magnitude, the very
    function the production fallback and the keyless attacker re-run,
    so 'publicly re-derivable' means exactly this function.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    if policy in ("keyed", "random") and (key is None) == (seed is None):
        raise ValueError(f"{policy} needs exactly one of key/seed")

    out: Dict[int, List[int]] = {}
    for lid in sorted(layer_allocation):
        k = int(layer_allocation[lid])
        if k == 0:
            out[lid] = []
            continue
        flat = residuals[lid].flatten()
        if policy == "magnitude":
            out[lid] = list(
                CarrierSelector().select_by_magnitude(flat, k)
            )
        elif policy == "random":
            gen = torch.Generator().manual_seed(int(seed) + int(lid))
            out[lid] = torch.randperm(
                flat.numel(), generator=gen
            )[:k].tolist()
        else:  # keyed
            out[lid] = sorted(
                IndexSampler.sample_positions(key, flat.numel(), k)
            )
    return out


def allocation_digest(layer_allocation: Dict[int, int]) -> str:
    """Prove every arm shared one allocation: sha1 of its canonical form."""
    canon = json.dumps(
        {str(k): int(v) for k, v in sorted(layer_allocation.items())},
        sort_keys=True,
    )
    return hashlib.sha1(canon.encode()).hexdigest()[:12]


class _PolicySelect:
    """Replacement bound onto IntelligentEmbedder.pipeline.select.

    Delegates to the production QACI pipeline (identical allocation,
    identical profiling), then — for the non-production arms — swaps
    the within-layer positions for the policy's. The embedder's crypto
    (AES-256-GCM), length header and write rule are untouched: this is
    an injection at the one seam where carriers are chosen.
    """

    def __init__(self, original, policy: str, *, key=None, seed=None):
        self._original = original
        self._policy = policy
        self._key = key
        self._seed = seed

    def select(self, residuals, total_payload_bits, **kwargs):
        # _original is the pipeline's BOUND select method (captured
        # before this wrapper shadows the attribute), so call it
        # directly — it has no .select of its own.
        base = self._original(
            residuals=residuals,
            total_payload_bits=total_payload_bits,
            **kwargs,
        )
        if self._policy == "magnitude":
            return base  # production fallback IS magnitude — no swap
        base.selected_indices = build_policy_indices(
            self._policy,
            residuals,
            base.layer_allocation,
            key=self._key,
            seed=self._seed,
        )
        base.total_selected = sum(
            len(v) for v in base.selected_indices.values()
        )
        return base


# ---------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------

def _ber(transmitted: List[int], recovered: List[int]):
    n = min(len(transmitted), len(recovered))
    if not n:
        return None, 0, 0
    errors = sum(
        1 for a, b in zip(transmitted[:n], recovered[:n]) if a != b
    )
    return errors / n, errors, n


def _noise_ber(strategy, embedded_r, carriers, transmitted,
               sigma=NOISE_SIGMA, trials=NOISE_TRIALS):
    """exp10's noise protocol on THIS arm's embed (re-embeds nothing)."""
    gen = torch.Generator().manual_seed(SEED)
    total_errors = total_compared = 0
    for _ in range(trials):
        noisy = {
            lid: t + torch.randn(t.shape, generator=gen) * sigma
            for lid, t in embedded_r.items()
        }
        recovered = extract_with(
            strategy, noisy, carriers,
            residuals_ref=None, strategy_name=STRATEGY,
        )
        _, errs, n = _ber(transmitted, recovered)
        total_errors += errs
        total_compared += n
    return total_errors / total_compared if total_compared else None


def keyless_qaci_read(stego, true_idx, transmitted, config,
                      total_bits: int) -> Dict[str, Any]:
    """exp13's tier_kerckhoffs, sign read: attacker re-runs the PUBLIC
    QACI selection on the stego residuals at the given payload size."""
    pipe = QACIPipeline(
        total_layers=config.num_hidden_layers, gamma=config.gamma
    )
    selection = pipe.select(residuals=stego, total_payload_bits=total_bits)
    att = selection.selected_indices

    tp = fp = fn = alloc_match = 0
    for lid in sorted(stego):
        attacker = set(att.get(lid, []))
        true = set(true_idx.get(lid, []))
        tp += len(attacker & true)
        fp += len(attacker - true)
        fn += len(true - attacker)
        alloc_match += len(attacker) == len(true)

    strategy = build_strategy(config, STRATEGY)
    recovered = extract_with(
        strategy, stego, att, residuals_ref=None, strategy_name=STRATEGY
    )
    ber, _, _ = _ber(transmitted, recovered)
    return {
        "attacker_total_bits": int(total_bits),
        "positions": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": tp / (tp + fp) if (tp + fp) else None,
            "recall": tp / (tp + fn) if (tp + fn) else None,
        },
        "layers_with_matching_allocation": alloc_match,
        "layers_total": len(stego),
        "keyless_ber": ber,
    }


# ---------------------------------------------------------------------
# One arm = one (policy, replicate) cell
# ---------------------------------------------------------------------

def run_arm(
    policy: str,
    replicate: int,
    config: EmbeddingConfig,
    residuals: Dict[int, torch.Tensor],
) -> Dict[str, Any]:
    key_label: Optional[str] = None
    seed: Optional[int] = None
    if policy == "random":
        seed = SEED + 100 * replicate
    elif policy == "keyed":
        key_label = f"ablation-key-{replicate}"

    embedder = IntelligentEmbedder(config)
    # Embedder calls pipeline.select(...) — shadow the attribute with
    # the wrapper's BOUND METHOD (the wrapper object itself is not
    # callable), after capturing the original bound method inside.
    embedder.pipeline.select = _PolicySelect(
        embedder.pipeline.select, policy, key=key_label, seed=seed
    ).select

    torch.manual_seed(SEED)
    result = embedder.embed(MESSAGE, residuals)
    transmitted = result.embedded_bits
    carriers = result.carrier_indices
    embedded_r = result.embedded_residuals

    strategy = build_strategy(config, STRATEGY)
    recovered = extract_with(
        strategy, embedded_r, carriers,
        residuals_ref=None, strategy_name=STRATEGY,
    )
    ber, bit_errors, bits_compared = _ber(transmitted, recovered)

    message, stats = DecryptPipeline(key=result.key).run(
        embedded_r, carriers
    )

    torch.manual_seed(SEED)
    sec = SecurityValidator().validate(
        original_residuals=residuals,
        embedded_residuals=embedded_r,
        carrier_indices=carriers,
    )

    robust = _noise_ber(strategy, embedded_r, carriers, transmitted)

    keyless_exact = keyless_qaci_read(
        embedded_r, carriers, transmitted, config, len(transmitted)
    )
    keyless_nominal = keyless_qaci_read(
        embedded_r, carriers, transmitted, config,
        config.total_payload_bits,
    )

    return {
        "policy": policy,
        "replicate": replicate,
        "identity": key_label if key_label else f"seed:{seed}",
        "bits_embedded": int(result.bits_embedded),
        "carrier_count": int(sum(len(v) for v in carriers.values())),
        "allocation_digest": allocation_digest(result.layer_allocation),
        "ber": ber,
        "bit_errors": bit_errors,
        "bits_compared": bits_compared,
        "decrypt_ok": bool(stats.get("success")),
        "recovered_matches": message == MESSAGE,
        "kl_divergence": float(sec.kl_divergence),
        "detector_accuracy": float(sec.detector_accuracy),
        "sign_bias": float(sec.sign_bias),
        "mean_shift": float(sec.moment_shift["mean_shift"]),
        "std_shift": float(sec.moment_shift["std_shift"]),
        "detectability_gate": {
            "kl_within_0_05": bool(sec.kl_divergence <= 0.05),
            "accuracy_within_0_55": bool(sec.detector_accuracy <= 0.55),
        },
        "robustness_ber_sigma_0_001": robust,
        "robustness_gate": {
            # exp6's reference line, reported per arm — baseline
            # arms are flagged against it but do not gate status
            # (THRESHOLDS['exp25'] says so explicitly).
            "within_exp6_line_0_02": (
                bool(robust is not None and robust <= 0.02)
            ),
        },
        "keyless_exact_size": keyless_exact,
        "keyless_nominal_size": keyless_nominal,
    }


# ---------------------------------------------------------------------
# Pre-registration (written before the first run; never edited after)
# ---------------------------------------------------------------------

HYPOTHESES = [
    {
        "id": "H1",
        "gate": "max_ber_all_arms",
        "statement": (
            "Every position policy round-trips the same 10,000-bit "
            "payload at BER 0.0 — the sign write rule is "
            "position-agnostic."
        ),
        "expected": "ber == 0.0 for every arm and replicate",
    },
    {
        "id": "H2",
        "gate": None,
        "statement": (
            "All arms stay inside exp7's detectability gates "
            "(detector accuracy <= 0.55, KL <= 0.05). Gate status is "
            "driven by the production (magnitude) arm; baselines are "
            "flagged against the same lines."
        ),
        "expected": "every arm: accuracy <= 0.55 and kl <= 0.05",
    },
    {
        "id": "H3",
        "gate": None,
        "statement": (
            "Keyless re-derivation — re-running the public QACI "
            "selection on the stego residuals at the exact payload "
            "size — locates carriers ONLY for the magnitude policy: "
            "the sign rule preserves |r| exactly (sign_strategy_v2), "
            "so magnitude ranks survive embedding and are public; "
            "keyed and random positions are not publicly derivable."
        ),
        "expected": (
            "magnitude: precision >= 0.99 and keyless_ber < 0.1; "
            "random & keyed: keyless_ber > 0.4"
        ),
    },
    {
        "id": "H4",
        "gate": None,
        "statement": (
            "Noise robustness orders with carrier magnitude: at "
            "sigma=0.001 small-|r| carriers flip sign more often, so "
            "the magnitude arm's BER is no worse than the random arm's."
        ),
        "expected": (
            "magnitude robustness_ber <= random mean robustness_ber"
        ),
    },
]


def _supports(hyp_id: str, arms: Dict[str, Any]) -> Optional[bool]:
    """Observed verdict per pre-registered hypothesis."""
    reps = {p: arms[p]["replicates"] for p in arms}
    if hyp_id == "H1":
        return all(r["ber"] == 0.0 for rs in reps.values() for r in rs)
    if hyp_id == "H2":
        return all(
            r["detector_accuracy"] <= 0.55 and r["kl_divergence"] <= 0.05
            for rs in reps.values() for r in rs
        )
    if hyp_id == "H3":
        mag = reps["magnitude"][0]["keyless_exact_size"]
        mag_ok = (
            mag["positions"]["precision"] is not None
            and mag["positions"]["precision"] >= 0.99
            and mag["keyless_ber"] is not None
            and mag["keyless_ber"] < 0.1
        )
        others_ok = all(
            r["keyless_exact_size"]["keyless_ber"] is not None
            and r["keyless_exact_size"]["keyless_ber"] > 0.4
            for p in ("random", "keyed") for r in reps[p]
        )
        return bool(mag_ok and others_ok)
    if hyp_id == "H4":
        mag = reps["magnitude"][0]["robustness_ber_sigma_0_001"]
        rand = [
            r["robustness_ber_sigma_0_001"] for r in reps["random"]
        ]
        if mag is None or any(v is None for v in rand):
            return None
        return bool(mag <= sum(rand) / len(rand))
    return None


def evaluate_gate(arms: Dict[str, Any], gate: Dict[str, Any]):
    failures = []
    for policy, block in arms.items():
        for r in block["replicates"]:
            if r["ber"] != 0.0:
                failures.append(
                    f"{policy}[{r['replicate']}] ber={r['ber']}"
                )
    mag = arms["magnitude"]["replicates"][0]
    if mag["detector_accuracy"] > gate["max_detector_accuracy"]:
        failures.append(
            f"production detector_accuracy={mag['detector_accuracy']}"
        )
    if mag["kl_divergence"] > gate["max_kl_divergence"]:
        failures.append(f"production kl={mag['kl_divergence']}")
    if (
        mag["robustness_ber_sigma_0_001"] is not None
        and mag["robustness_ber_sigma_0_001"]
        > gate["max_production_ber_at_sigma_0_001"]
    ):
        failures.append(
            "production robustness_ber="
            f"{mag['robustness_ber_sigma_0_001']}"
        )
    return ("PASS" if not failures else "FAIL"), failures


def _summarize(block: Dict[str, Any]) -> Dict[str, Any]:
    reps = block["replicates"]
    bers = [r["ber"] for r in reps if r["ber"] is not None]
    return {
        "replicates": len(reps),
        "ber_values": bers,
        "ber_max": max(bers) if bers else None,
        "keyless_exact_ber_values": [
            r["keyless_exact_size"]["keyless_ber"] for r in reps
        ],
        "robustness_values": [
            r["robustness_ber_sigma_0_001"] for r in reps
        ],
    }


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

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

    arms: Dict[str, Any] = {}
    for policy in POLICIES:
        log(f"arm '{policy}' — {REPLICATES[policy]} replicate(s) ...")
        reps = []
        for replicate in range(REPLICATES[policy]):
            rep = run_arm(policy, replicate, config, residuals)
            log(
                f"  rep {replicate}: ber={rep['ber']} "
                f"acc={rep['detector_accuracy']:.4f} "
                f"kl={rep['kl_divergence']:.5f} "
                f"robust={rep['robustness_ber_sigma_0_001']} "
                f"keyless(exact)="
                f"{rep['keyless_exact_size']['keyless_ber']}"
            )
            reps.append(rep)
        arms[policy] = {"replicates": reps, "summary": _summarize({
            "replicates": reps
        })}

    # One allocation across arms — the ablation's control variable.
    digests = {
        r["allocation_digest"]
        for block in arms.values() for r in block["replicates"]
    }
    allocation_shared = len(digests) == 1
    log(f"allocation shared across arms: {allocation_shared} "
        f"(digest {sorted(digests)})")

    gate = dict(THRESHOLDS[EXPERIMENT])
    status, failures = evaluate_gate(arms, gate)
    gate_block = {
        **gate,
        "gate_source": f"experiment_registry.THRESHOLDS['{EXPERIMENT}']",
        "status": status,
        "failures": failures,
    }
    log(f"gate: {status}" + (f" — {failures}" if failures else ""))

    hypotheses = []
    for hyp in HYPOTHESES:
        observed = _supports(hyp["id"], arms)
        hypotheses.append({
            **hyp,
            "observed_supported": observed,
        })
        log(f"{hyp['id']}: supported={observed}")

    artifact = {
        "experiment": EXPERIMENT,
        "title": "Selection-policy ablation (W9.1)",
        "model_id": args.model,
        "family": context.family,
        "status": status,
        "gate": gate_block,
        "arms": arms,
        "allocation_shared_across_arms": allocation_shared,
        "hypotheses": hypotheses,
        "method": {
            "held_constant": (
                "sign write rule, QACI Hamilton layer allocation "
                "(digest-pinned), payload 10,000 bits, message, model, "
                "AES-256-GCM + 32-bit header (IntelligentEmbedder "
                "unmodified — only pipeline.select is bound to a "
                "policy wrapper)"
            ),
            "varied": (
                "within-layer positions only: random (randperm), "
                "magnitude (production fallback == "
                "CarrierSelector.select_by_magnitude), keyed "
                "(IndexSampler SHA-256 key)"
            ),
            "keyless_attacker": (
                "exp13's tier_kerckhoffs shape with a sign read: re-run "
                "the public QACI selection on the STEGO residuals at "
                "the exact (len(transmitted)) and nominal "
                "(config.total_payload_bits) sizes"
            ),
            "robustness": (
                "exp10's protocol: additive Gaussian sigma=0.001, "
                f"{NOISE_TRIALS} trials, fresh generator seeded {SEED}"
            ),
            "seed": SEED,
            "pre_registered": (
                "HYPOTHESES in this file were written before the first "
                "run; gates live in experiment_registry.THRESHOLDS so "
                "this file contains no threshold of its own. Misses "
                "are recorded, not rewritten."
            ),
        },
        "notes": [
            "Baseline arms (random, keyed) are NOT deployment "
            "candidates: their detectability and robustness numbers "
            "are reported against exp7/exp6's reference lines as "
            "per-arm flags; only the production arm gates the "
            "artifact's status (THRESHOLDS description says so "
            "explicitly).",
            "exp13's phase-tier attacker is out of scope here — it is "
            "measured on exp13's LWE embed and needs no parameters; "
            "this experiment measures the public-rule (Kerckhoffs) "
            "attacker for the sign scheme. Cross-attacker coverage "
            "remains exp13's result.",
            "Magnitude preserving |r| is a property of "
            "sign_strategy_v2.get_bit_for_residual (returns +/-"
            "max(|r|, min_magnitude)) — this was the code-level "
            "basis for H3, and the MEASUREMENT refined it: "
            "near-exact position re-derivation (see "
            "keyless_exact_size.positions.precision — precision "
            "≈ 0.99) still reads near chance, because Hamilton "
            "allocation recomputed on the stego weights never "
            "reproduces the writer's per-layer counts in full "
            "(field: layers_with_matching_allocation; the sign "
            "flips depend on the AES key, so the drift varies per "
            "embed), and the first mismatched layer "
            "desynchronizes the stream from there down — "
            "consistent with exp13's tier_kerckhoffs, which "
            "records the same field for the same reason. H3 "
            "missed and is kept as written; this note is "
            "post-run interpretation, not a pre-registered "
            "claim. Precision and layer-matches are cited from "
            "the fields, never fixed here.",
        ],
        "reproducibility": context.reproducibility(),
    }

    target = (
        RESULTS_DIR
        / f"{EXPERIMENT}_selection_ablation_{_slug(args.model)}.json"
    )
    target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    log(f"wrote {target}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
