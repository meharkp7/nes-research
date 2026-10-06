"""
W6 — model surgery survival (exp23).

RESEARCH_PLAN §4 W6: *"Does the embedding survive contact with the
rest of the ML lifecycle? A carrier that survives one cycle but dies
on first fine-tune is not viable. Each is: embed → surgery →
extract → BER. Expect degradation; the question is whether it is
total or graceful."*

One production-path embed (sign, exp3's payload), then the SAME
stego weights run through nine cells — a weight-path control plus
eight surgeries — so the surgery is the only variable (exp21's
paired design, in weight space).

Pipeline (the weights, exactly as the codebase defines them):

    r_ref  = extract_residuals(nf4, fp16)      # R = W_FP16 - dequant(W_NF4)
    W_stego[l] = dequant(nf4)[l] + embedded_r[l]
             (= apply_residuals_to_model's own formula; built from
              fp16[l] - r_ref[l] + embedded_r[l] per layer)
    cell:   W' = surgery(W_stego);  r' = embedded_r + (W' - W_stego)
    extract: strategy extractor over r' at the embed's carriers

Because r' is the embedded residual plus the surgery's WEIGHT-SPACE
delta, the control (no surgery) is embedded_r exactly — a non-zero
control BER means plumbing, not physics, and the gate says so.

Cells (all fp32 — the pipeline's own arithmetic; an fp16 checkout
would round sub-residual displacements away):

    control      no surgery (validity check, must be 0.0)
    lora_*       LoRA-shaped low-rank updates at two stated RMS
                 scales (0.1%, 1% of RMS(W)) — the MERGE step of a
                 LoRA is a linear add; training is out of scope
                 (peft absent, recorded under not_run)
    prune_*      magnitude pruning, per layer: smallest-|W|
                 fraction zeroed (10%, 30%)
    nf4_requant  W' = dequant(quantize_nf4(W_stego), blocksize 64)
                 — the NF4 leg of W6.3; the GPTQ/AWQ legs cannot run
                 (no gptqmodel/auto_gptq; transformers 5.16.1 removed
                 AwqQuantizer) — recorded under not_run, not patched
    merge_*      task-vector merge with Qwen2.5-3B-Instruct (same
                 shapes, cached): W' = W_stego + t·(W_inst - W_stego),
                 at t = 0.01 / 0.05 / 0.5

W6.2 (1–10k fine-tuning steps) does not run: `peft`, `trl` and
`gptqmodel` are absent from this environment (probed at runtime and
recorded in the artifact's `not_run`), installs are not a research
action, and an honest full-parameter run exceeds this machine. The
existing sigma-noise axes (exp6/10/18/22) remain the only measured
proxy for dense drift; fine-tune drift stays NOT_RUN and named.

Pre-registered expectations, recorded before the run (misses are
recorded, never rewritten): control 0.0 (structural); lora cells
expected to survive or degrade gracefully (carriers are top-|r|
positions, margins ≫ the stated RMS scales); prune outcome depends
on the |W|-vs-|r| overlap and is genuinely open; nf4_requant
plausibly heavy loss (deltas smaller than NF4 bucket spacing never
reach the bytes) but measured, not assumed; merge expected to die
as t grows. Gate: THRESHOLDS['exp23'] — exp3's 0.0 twice, both
reused. Per-cell verdicts (exp18's rule); the BER column IS the
total-vs-graceful reading.

Usage:
    python -m src.experiments.exp23_model_surgery --model <id>
"""

import argparse
import gc
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from src.core.types import EmbeddingConfig  # noqa: E402
from src.embedding.intelligent_embedder import IntelligentEmbedder  # noqa: E402
from src.embedding.strategy_registry import (  # noqa: E402
    build as build_strategy,
    extract_with,
)
from src.experiments.experiment_registry import gate_for  # noqa: E402
from src.experiments.exp15_lwe_fidelity import _context_for, _slug  # noqa: E402
from src.experiments.paths import RESULTS_DIR  # noqa: E402
from src.experiments.residual_source import load_cached_residuals  # noqa: E402

EXPERIMENT = "exp23"
STRATEGY = "sign"          # production's own scheme (DecryptPipeline)
PAYLOAD_BITS = 10_000      # exp10's / exp3's payload
MESSAGE = "A" * 1_250
SEED = 42
MERGE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
PRUNE_FRACTIONS = (0.10, 0.30)
LORA_RATIOS = (0.001, 0.01)   # target RMS(delta)/RMS(W)
LORA_RANK = 8
MERGE_T = (0.01, 0.05, 0.5)

# The environment probe, run at artifact time — blockers recorded as
# facts from THIS interpreter, not as prose.
DEPENDENCIES = ("peft", "trl", "gptqmodel", "auto_gptq", "bitsandbytes")


def log(message: str) -> None:
    """Flush every stage — a long run must never look hung (§3.7)."""
    print(message, flush=True)


# ------------------------------------------------------------------
# Surgery primitives (pure; unit-tested without a model)
# ------------------------------------------------------------------

def magnitude_prune(w: torch.Tensor, fraction: float) -> torch.Tensor:
    """Zero the smallest-|w| `fraction` of entries, in place on a copy.

    Per-layer magnitude pruning — the standard variant: victims are
    chosen by |weight| within each tensor.
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"prune fraction must be in [0, 1), got {fraction}")
    out = w.clone()
    flat = out.abs().flatten()
    k = int(fraction * flat.numel())
    if k:
        cutoff = torch.kthvalue(flat, k).values
        out[out.abs() <= cutoff] = 0.0
    return out


def low_rank_delta(
    w: torch.Tensor,
    ratio: float,
    rank: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """A LoRA-shaped delta: B @ A scaled so RMS(delta) = ratio * RMS(w).

    The merge step of a LoRA adapter is exactly this linear add; the
    training that produced B, A is out of scope (peft absent — see
    not_run). Seeded, so a cell reproduces.
    """
    rows, cols = w.shape
    a = torch.randn(rank, cols, generator=generator, dtype=torch.float32)
    b = torch.randn(rows, rank, generator=generator, dtype=torch.float32)
    delta = b @ a
    target = ratio * w.float().pow(2).mean().sqrt()
    delta = delta * (target / delta.float().pow(2).mean().sqrt())
    return delta.to(w.dtype)


def merge_delta(
    w_stego: torch.Tensor,
    w_other: torch.Tensor,
    t: float,
) -> torch.Tensor:
    """Task-vector merge delta: returns t * (W_other - W_stego), the
    delta to ADD so W' = W_stego + delta.

    The base-side weight of the task vector IS W_stego (it stands in
    for the base plus the payload); t=0 is the no-op, t=1 lands on
    W_other exactly. The caller applies it (measure_cell computes
    r' = embedded_r + delta).
    """
    if w_other.shape != w_stego.shape:
        raise ValueError(
            f"merge shapes differ: {tuple(w_other.shape)} vs "
            f"{tuple(w_stego.shape)} — never reshape a mismatched matrix"
        )
    return t * (w_other.float() - w_stego.float()).to(w_stego.dtype)


def nf4_requant(w: torch.Tensor) -> torch.Tensor:
    """The W6.3 NF4 leg: dequant(quantize_nf4(w)) — group 64, the
    shipped format's blocksize, via bitsandbytes' own kernels."""
    from bitsandbytes.functional import dequantize_4bit, quantize_4bit

    q, state = quantize_4bit(
        w.detach().to(torch.float16).cpu(), quant_type="nf4"
    )
    back = dequantize_4bit(q, state, quant_type="nf4")
    return back.to(torch.float32).to(w.device)


# ------------------------------------------------------------------
# Cells
# ------------------------------------------------------------------

def _rms(t: torch.Tensor) -> float:
    return float(t.float().pow(2).mean().sqrt())


def measure_cell(
    name: str,
    deltas: Dict[int, torch.Tensor],
    embedded_residuals: Dict[int, torch.Tensor],
    carrier_indices: Dict[int, List[int]],
    embedded_bits: List[int],
    strategy,
    w_stego: Dict[int, torch.Tensor],
) -> Dict[str, Any]:
    """One surgery: r' = embedded_r + delta, extract, measure.

    Magnitude stats are recorded beside the BER so the number can be
    explained, not just cited.
    """
    r_prime: Dict[int, torch.Tensor] = {}
    d2 = w2 = 0.0
    dcar2 = 0.0
    ncar = 0
    touched = 0
    for lid, emb in embedded_residuals.items():
        delta = deltas.get(lid)
        if delta is None:
            r_prime[lid] = emb
            continue
        r_prime[lid] = emb + delta
        d2 += float(delta.float().pow(2).sum())
        w2 += float(w_stego[lid].float().pow(2).sum())
        idx = carrier_indices.get(lid, [])
        if idx:
            dv = delta.float().flatten()[idx]
            dcar2 += float(dv.pow(2).sum())
            ncar += len(idx)
            touched += int((dv != 0).sum())

    recovered = extract_with(
        strategy, r_prime, carrier_indices, residuals_ref=None,
        strategy_name=STRATEGY,
    )
    compared = min(len(embedded_bits), len(recovered))
    errors = sum(
        1
        for a, b in zip(embedded_bits[:compared], recovered[:compared])
        if a != b
    )
    return {
        "surgery": name,
        "ber": errors / compared if compared else None,
        "bit_errors": errors,
        "bits_compared": compared,
        "rms_delta_over_rms_w": (d2 / w2) ** 0.5 if w2 else None,
        "rms_delta_at_carriers": (dcar2 / ncar) ** 0.5 if ncar else None,
        "carriers_displaced": touched,
        "carriers_total": ncar,
    }


def not_run_findings() -> List[Dict[str, str]]:
    """What cannot run here, with the probe that proves it."""
    availability = {
        dep: importlib.util.find_spec(dep) is not None
        for dep in DEPENDENCIES
    }
    findings = []
    if not availability["peft"] or not availability["trl"]:
        findings.append({
            "item": "W6.2 — 1-10k fine-tuning steps",
            "status": "BLOCKED (recorded, not patched)",
            "reason": (
                f"peft={availability['peft']}, trl={availability['trl']} "
                "in this environment; installs are not a research action, "
                "and an honest full-parameter run exceeds this machine "
                "(26 GB). wikitext is cached but no trainer exists to "
                "run it. Fine-tune drift stays NOT_RUN and named; the "
                "sigma-noise axes (exp6/10/18/22) remain the only "
                "measured proxy for dense drift."
            ),
        })
    if not availability["gptqmodel"] and not availability["auto_gptq"]:
        findings.append({
            "item": "W6.3 — GPTQ leg (NF4 -> GPTQ -> back)",
            "status": "BLOCKED (recorded, not patched)",
            "reason": (
                f"gptqmodel={availability['gptqmodel']}, "
                f"auto_gptq={availability['auto_gptq']}; transformers' "
                "GPTQConfig imports but has no quantization backend "
                "without one of them. The NF4 leg runs; GPTQ does not."
            ),
        })
    findings.append({
        "item": "W6.3 — AWQ leg",
        "status": "BLOCKED (recorded, not patched)",
        "reason": (
            "transformers 5.16.1 raises ImportError for AwqQuantizer "
            "(probed at artifact time); autoawq/awq absent. Consistent "
            "with the standing AWQ dequantizer gate."
        ),
    })
    if not availability["bitsandbytes"]:
        findings.append({
            "item": "W6.3 — NF4 leg",
            "status": "BLOCKED",
            "reason": "bitsandbytes absent — nf4_requant cannot run",
        })
    return findings


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
    log(f"Model: {args.model} "
        f"(family {context.family}, {context.expected_layers} layers)")

    log("[exp23] loading cached residuals ...")
    residuals = load_cached_residuals(args.model, context.expected_layers)

    # One embed, every cell reads from it (exp21's paired design).
    config = EmbeddingConfig(
        total_payload_bits=PAYLOAD_BITS,
        embedding_strategy=STRATEGY,
        model_family=context.family,
        num_hidden_layers=(
            context.actual_layers or context.expected_layers
        ),
    )
    log(f"[exp23] embed via production path (strategy={STRATEGY}) ...")
    embed_result = IntelligentEmbedder(config).embed(MESSAGE, residuals)
    embedded_r = embed_result.embedded_residuals
    carriers = embed_result.carrier_indices
    transmitted = embed_result.embedded_bits
    log(f"  embedded {len(transmitted)} bits "
        f"(reported {embed_result.bits_embedded})")

    strategy = build_strategy(config, STRATEGY)

    log("[exp23] direct control (exp10's path, no weights) ...")
    direct = extract_with(
        strategy, embedded_r, carriers, residuals_ref=None,
        strategy_name=STRATEGY,
    )
    cmp_d = min(len(transmitted), len(direct))
    direct_ber = (
        sum(a != b for a, b in zip(transmitted[:cmp_d], direct[:cmp_d]))
        / cmp_d if cmp_d else None
    )
    log(f"  direct control ber={direct_ber}")

    # The model pair: dequant reference + fp16 side, from the code's
    # own residual definition.
    log("[exp23] loading model pair (nf4 + fp16) ...")
    nf4_model, fp16_model, _tok = context.ensure_models()
    from src.model.model_loader import extract_residuals

    log("[exp23] computing R = W_FP16 - dequant(W_NF4) from the pair ...")
    r_ref = extract_residuals(nf4_model, fp16_model, context.family)

    cache_check = max(
        float(
            (r_ref[lid].detach().float().cpu()
             - residuals[lid].detach().float().cpu()).abs().max()
        )
        for lid in r_ref
        if lid in residuals
    )
    log(f"  max|pair residual - cached residual| = {cache_check:.3e}")

    # W_stego = dequant(nf4) + embedded_residual, layer by layer —
    # apply_residuals_to_model's formula, built without mutating the
    # models. dequant(nf4) = fp16 - r_ref by the definition above.
    # Everything lands on CPU: cached residuals are CPU tensors and
    # surgery/extract arithmetic runs there.
    log("[exp23] building W_stego per layer ...")
    w_stego: Dict[int, torch.Tensor] = {}
    for lid, fp16_w in _down_projs(fp16_model, context.family).items():
        w_stego[lid] = (
            fp16_w.detach().float().cpu()
            - r_ref[lid].float().cpu()
            + embedded_r[lid].float()
        )
    # Weight-path control: r' = W_stego - dequant equals embedded_r
    # by construction (cache_vs_pair above proves the views match),
    # so the control cell runs zero deltas — validating the cell
    # arithmetic itself, not the identity.
    control_delta = {
        lid: torch.zeros_like(w) for lid, w in w_stego.items()
    }
    del r_ref, residuals
    gc.collect()

    cells: List[Dict[str, Any]] = []
    gate = gate_for(EXPERIMENT)

    log("[exp23] cell control (through the weights) ...")
    cells.append(measure_cell(
        "control", control_delta, embedded_r, carriers, transmitted,
        strategy, w_stego,
    ))
    log(f"  ber={cells[-1]['ber']}")

    # --- W6.1: LoRA-shaped low-rank updates ------------------------
    for ratio in LORA_RATIOS:
        name = f"lora_{ratio:g}"
        log(f"[exp23] cell {name} (rank {LORA_RANK}, "
            f"RMS target {ratio:g}) ...")
        deltas: Dict[int, torch.Tensor] = {}
        for lid, w in w_stego.items():
            gen = torch.Generator().manual_seed(SEED + lid)
            deltas[lid] = low_rank_delta(w, ratio, LORA_RANK, gen)
        cells.append(measure_cell(
            name, deltas, embedded_r, carriers, transmitted,
            strategy, w_stego,
        ))
        log(f"  ber={cells[-1]['ber']} "
            f"rms_d/rms_w={cells[-1]['rms_delta_over_rms_w']:.2e}")
        del deltas
        gc.collect()

    # --- W6.4: magnitude pruning -----------------------------------
    for fraction in PRUNE_FRACTIONS:
        name = f"prune_{int(fraction * 100)}"
        log(f"[exp23] cell {name} ...")
        deltas = {
            lid: magnitude_prune(w, fraction) - w
            for lid, w in w_stego.items()
        }
        cells.append(measure_cell(
            name, deltas, embedded_r, carriers, transmitted,
            strategy, w_stego,
        ))
        log(f"  ber={cells[-1]['ber']} "
            f"carriers_displaced={cells[-1]['carriers_displaced']}/"
            f"{cells[-1]['carriers_total']}")
        del deltas
        gc.collect()

    # --- W6.3: NF4 leg ---------------------------------------------
    log("[exp23] cell nf4_requant ...")
    deltas = {
        lid: nf4_requant(w) - w for lid, w in w_stego.items()
    }
    cells.append(measure_cell(
        "nf4_requant", deltas, embedded_r, carriers, transmitted,
        strategy, w_stego,
    ))
    log(f"  ber={cells[-1]['ber']}")
    del deltas
    gc.collect()

    # --- W6.5: task-vector merge with the same-shape checkpoint ----
    log(f"[exp23] loading merge partner {MERGE_MODEL_ID} ...")
    other = _load_merge_partner(MERGE_MODEL_ID, context.family)
    for t in MERGE_T:
        name = f"merge_{t:g}"
        log(f"[exp23] cell {name} ...")
        deltas = {
            lid: merge_delta(w_stego[lid], other[lid], t)
            for lid in w_stego
        }
        cells.append(measure_cell(
            name, deltas, embedded_r, carriers, transmitted,
            strategy, w_stego,
        ))
        log(f"  ber={cells[-1]['ber']}")
        del deltas
        gc.collect()
    del other
    gc.collect()

    for cell in cells:
        cell["meets_gate"] = cell["ber"] == gate["max_ber"]
    control_ok = cells[0]["ber"] == gate["max_control_ber"] == direct_ber

    artifact: Dict[str, Any] = {
        "experiment": EXPERIMENT,
        "title": "W6 — model surgery survival (first pass, one model)",
        "model_id": args.model,
        "family": context.family,
        "num_layers": context.expected_layers,
        "gate": {
            **gate,
            "gate_source": "experiment_registry.THRESHOLDS['exp23']",
        },
        "control": {
            "direct_ber": direct_ber,
            "weight_path_ber": cells[0]["ber"],
            "cache_vs_pair_max_abs": cache_check,
            "valid": control_ok,
        },
        "cells": cells,
        "not_run": not_run_findings(),
        "method": {
            "pipeline": (
                "W_stego = dequant(nf4) + embedded_residual per layer "
                "(apply_residuals_to_model's formula); cell residual = "
                "embedded_residual + (W' - W_stego); extraction via the "
                "sign strategy's own extractor (production scheme)"
            ),
            "residual_definition": "R = W_FP16 - dequantize(W_NF4)",
            "dtype": (
                "float32 throughout — the pipeline's own arithmetic "
                "(extract_residuals dequantizes to fp32); fp16 would "
                "round sub-residual displacements away"
            ),
            "strategy": STRATEGY,
            "payload_bits": PAYLOAD_BITS,
            "one_embed_for_all_cells": True,
            "surgery_scope": (
                "mlp.down_proj only — the payload's scope; other "
                "modules cannot touch the payload by construction"
            ),
            "prune": "per-layer magnitude pruning, smallest-|W| fraction",
            "lora": (
                f"LoRA-shaped B@A, rank {LORA_RANK}, RMS scaled to "
                f"{LORA_RATIOS} of RMS(W); training out of scope "
                "(peft absent — see not_run), merge step is linear"
            ),
            "nf4_requant": (
                "bitsandbytes quantize_4bit/dequantize_4bit, nf4, "
                "default blocksize 64 (the shipped group size)"
            ),
            "merge": (
                f"task-vector: W' = W_stego + t*(W_instruct - W_stego), "
                f"t in {MERGE_T}; partner {MERGE_MODEL_ID} (same shapes) "
                "— the only cached same-architecture second checkpoint"
            ),
            "seed": SEED,
            "pre_registered": (
                "control 0.0; lora survive-or-graceful; prune open "
                "(|W|-vs-|r| overlap unknown); nf4_requant plausibly "
                "heavy loss (in-bucket deltas never reach the bytes); "
                "merge dies as t grows. Misses are recorded, not "
                "rewritten."
            ),
        },
        "notes": [
            "Per-cell verdicts, exp18's rule: 'survives' = BER 0.0 "
            "against exp3's number; any non-zero BER is a cell failing "
            "that axis, and the BER column is the plan's "
            "total-vs-graceful reading. No gate is relaxed for "
            "expected degradation.",
            "The weight-path control must equal the direct control "
            "(both exp3's 0.0): it proves the weights carry exactly "
            "the embedded residual before any surgery touches them.",
            "cache_vs_pair_max_abs records whether the cached "
            "residuals equal a fresh R from the model pair; a large "
            "value would mean embed operated on a different residual "
            "view than the weights are built from — reported, never "
            "assumed away.",
        ],
        "reproducibility": context.reproducibility(),
    }

    target = RESULTS_DIR / f"{EXPERIMENT}_model_surgery_{_slug(args.model)}.json"
    target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    log("=" * 70)
    log(f"control: direct={direct_ber} weight_path={cells[0]['ber']} "
        f"cache_vs_pair={cache_check:.3e}")
    for cell in cells:
        log(f"  {cell['surgery']:<14} ber={cell['ber']} "
            f"errs={cell['bit_errors']}/{cell['bits_compared']} "
            f"gate={cell['meets_gate']}")
    log(f"not_run: {len(artifact['not_run'])} blockers recorded")
    log(f"wrote {target}")

    del w_stego, embedded_r, carriers, transmitted
    gc.collect()
    torch.mps.empty_cache()
    return 0


def _down_projs(model, family: str) -> Dict[int, torch.Tensor]:
    """{layer_id: down_proj.weight} for the payload's module."""
    from src.model.registry import get_layer_module, get_num_layers

    out = {}
    for i in range(get_num_layers(model)):
        out[i] = get_layer_module(model, family, i, "mlp").down_proj.weight
    return out


def _load_merge_partner(model_id: str, family: str) -> Dict[int, torch.Tensor]:
    """fp16 down_projs of the merge partner, then the model is freed."""
    from transformers import AutoModelForCausalLM

    other = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map={"": "cpu"}
    )
    weights = {
        lid: t.detach().cpu().float()
        for lid, t in _down_projs(other, family).items()
    }
    del other
    gc.collect()
    return weights


if __name__ == "__main__":
    raise SystemExit(main())
