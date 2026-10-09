#!/usr/bin/env python3
"""Real BitsAndBytes-NF4 tensor pilot for Candidate A (QSE) and Candidate B (DCE).

This is deliberately a tensor-level experiment, not a full checkpoint/artifact
sender-receiver protocol. It uses a real FP16 model tensor and the installed
bitsandbytes NF4 quantizer/codebook, records all settings, and never overwrites
an existing report. Results must not be described as model-level stealth or
end-to-end robustness evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
from pathlib import Path
import random
import sys
from datetime import datetime, timezone

import torch
from bitsandbytes.functional import quantize_4bit, dequantize_4bit
from transformers import AutoModelForCausalLM


def bits_for(seed: int, n: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.getrandbits(1) for _ in range(n)]


def positions_for(key: bytes, n_values: int, n_bits: int, domain: bytes) -> list[int]:
    if n_bits > n_values:
        raise ValueError(f"payload ({n_bits}) exceeds tensor values ({n_values})")
    out: list[int] = []
    seen: set[int] = set()
    counter = 0
    while len(out) < n_bits:
        block = hmac.new(key, domain + counter.to_bytes(8, "big"), hashlib.sha256).digest()
        counter += 1
        for off in range(0, len(block), 4):
            pos = int.from_bytes(block[off:off + 4], "big") % n_values
            if pos not in seen:
                seen.add(pos)
                out.append(pos)
                if len(out) == n_bits:
                    break
    return out


def ber(expected: list[int], observed: list[int]) -> dict:
    errors = sum(a != b for a, b in zip(expected, observed))
    return {"bit_errors": errors, "bits_compared": len(expected),
            "ber": errors / len(expected) if expected else 0.0,
            "exact_recovery": errors == 0}


def tv_from_counts(a: dict[int, int], b: dict[int, int], n: int) -> float:
    if not n:
        return 0.0
    support = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in support) / n


def unpack_codes(packed: torch.Tensor, n_values: int) -> list[int]:
    # Match the repository's established NF4 SafeTensors nibble convention:
    # even carrier index -> high nibble; odd index -> low nibble.
    raw = packed.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()
    return [((raw[i // 2] >> 4) & 15) if i % 2 == 0 else (raw[i // 2] & 15)
            for i in range(n_values)]


def pack_codes(codes: list[int], device: torch.device) -> torch.Tensor:
    if len(codes) % 2:
        raise ValueError("NF4 packed code count must be even")
    raw = bytearray(len(codes) // 2)
    for i in range(0, len(codes), 2):
        raw[i // 2] = ((codes[i] & 15) << 4) | (codes[i + 1] & 15)
    return torch.tensor(list(raw), dtype=torch.uint8, device=device)


def hist(values: list[int]) -> dict[int, int]:
    result: dict[int, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def run(args: argparse.Namespace) -> dict:
    source = Path(args.model).expanduser().resolve()
    report_path = Path(args.output).expanduser().resolve()
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report_path}")
    if not source.exists():
        raise FileNotFoundError(f"Model path does not exist: {source}")

    print(f"Loading FP16/BF16 source model on CPU: {source}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(source), torch_dtype=torch.float16, device_map="cpu",
        local_files_only=args.local_files_only, trust_remote_code=False,
    )
    state = model.state_dict()
    if args.tensor not in state:
        examples = [name for name in state if name.endswith(args.tensor)]
        raise KeyError(f"Tensor {args.tensor!r} not found. Suffix matches: {examples[:10]}")
    original = state[args.tensor].detach().float().cpu().contiguous().flatten()
    del state, model
    if args.max_values and original.numel() > args.max_values:
        original = original[:args.max_values].contiguous()
    if original.numel() % 2:
        original = original[:-1].contiguous()

    qweight, qstate = quantize_4bit(original, quant_type="nf4", blocksize=args.blocksize)
    base = dequantize_4bit(qweight, quant_state=qstate).float().cpu().flatten()
    if base.numel() != original.numel():
        raise RuntimeError(f"NF4 round-trip shape mismatch: {base.numel()} vs {original.numel()}")
    n_bits = min(args.payload_bits, original.numel())
    key = bytes.fromhex(args.test_key_hex)
    bits = bits_for(args.seed, n_bits)
    positions = positions_for(key, original.numel(), n_bits, b"NES-real-NF4-candidate-pilot-v1")
    ref_hash = hashlib.sha256(original.numpy().tobytes()).hexdigest()
    report = {
        "schema": "nes.real_nf4_candidate_pilot.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(source), "tensor": args.tensor,
        "tensor_values_tested": int(original.numel()), "payload_bits": n_bits,
        "seed": args.seed, "blocksize": args.blocksize,
        "quantizer": "bitsandbytes.functional.quantize_4bit(quant_type='nf4')",
        "source_tensor_sha256": ref_hash,
        "receiver_contract": "candidate-specific tensor pilot; keyed carrier positions; not a full artifact protocol",
        "candidates": {},
    }

    # Candidate A / QSE: embed relative to an actual NF4-derived reference
    # residual, then run the modified reconstructed weights through real NF4.
    residual = original - base
    qref, sref = quantize_4bit(original, quant_type="nf4", blocksize=args.blocksize)
    reference = original - dequantize_4bit(qref, quant_state=sref).float().cpu().flatten()
    delta = max(float(reference.std().item()) * args.qse_margin, 1e-8)
    embedded_residual = residual.clone()
    for bit, pos in zip(bits, positions):
        embedded_residual[pos] = reference[pos] + delta if bit else reference[pos] - delta
    qse_weight = base + embedded_residual
    qse_q, qse_state = quantize_4bit(qse_weight, quant_type="nf4", blocksize=args.blocksize)
    qse_dequant = dequantize_4bit(qse_q, quant_state=qse_state).float().cpu().flatten()
    qse_observed_residual = qse_dequant - base
    qse_observed = [int(qse_observed_residual[p] >= reference[p]) for p in positions]
    qse_err = ber(bits, qse_observed)
    report["candidates"]["A_QSE"] = {
        **qse_err, "margin_scale_of_reference_std": args.qse_margin,
        "absolute_margin": delta,
        "weight_rmse_vs_original_fp16": float(torch.mean((qse_dequant-original)**2).sqrt().item()),
        "weight_rmse_vs_clean_nf4": float(torch.mean((qse_dequant-base)**2).sqrt().item()),
        "receiver_side_information": "original FP16 tensor and clean NF4 reconstruction/reference residual required in this pilot",
        "interpretation": "real NF4 quantizer round trip; not yet artifact-only receiver recovery",
    }

    # Candidate B / DCE: optimize real NF4 codebook indices at keyed carriers.
    # The codebook and per-block absmax come from the actual bitsandbytes state.
    original_codes = unpack_codes(qweight, original.numel())
    codebook = qstate.code.detach().float().cpu().flatten().tolist()
    absmax = qstate.absmax.detach().float().cpu().flatten().tolist()
    blocksize = int(qstate.blocksize)
    if len(codebook) != 16:
        raise RuntimeError(f"Expected 16-entry NF4 codebook, got {len(codebook)}")
    if math.ceil(original.numel() / blocksize) > len(absmax):
        raise RuntimeError("NF4 quantization-state absmax does not cover tensor")
    target_hist = hist([original_codes[p] for p in positions])
    baseline_codes = original_codes.copy()
    dce_codes = original_codes.copy()
    # Baseline: closest dequantized NF4 level whose index parity carries the bit.
    feasible_by_pos: list[list[tuple[int, float]]] = []
    for bit, pos in zip(bits, positions):
        scale = absmax[pos // blocksize]
        target_value = float(base[pos].item())
        feasible = [(c, (codebook[c] * scale - target_value) ** 2)
                    for c in range(16) if (c & 1) == bit]
        feasible_by_pos.append(feasible)
        baseline_codes[pos] = min(feasible, key=lambda item: (item[1], item[0]))[0]

    # DCE: greedily trade local weight distortion against global carrier-code
    # histogram matching. lambda is fixed before each run and reported.
    counts: dict[int, int] = {}
    target_n = max(1, len(positions))
    for code, count in target_hist.items():
        counts[code] = 0
    for pos_i, (bit, pos, feasible) in enumerate(zip(bits, positions, feasible_by_pos)):
        best_code = None
        best_score = None
        scale = absmax[pos // blocksize]
        old_value = float(base[pos].item())
        for code, distortion in feasible:
            projected = counts.copy()
            projected[code] = projected.get(code, 0) + 1
            support = set(target_hist) | set(projected)
            before = sum((counts.get(k, 0) - target_hist.get(k, 0) * (pos_i / target_n)) ** 2 for k in support)
            after = sum((projected.get(k, 0) - target_hist.get(k, 0) * ((pos_i + 1) / target_n)) ** 2 for k in support)
            score = distortion / (float(torch.mean(base.square()).item()) + 1e-12) + args.dce_lambda * (after - before)
            tie = (score, distortion, code)
            if best_score is None or tie < best_score:
                best_score, best_code = tie, code
        assert best_code is not None
        dce_codes[pos] = best_code
        counts[best_code] = counts.get(best_code, 0) + 1

    device = qweight.device
    bnbaseline_q = pack_codes(baseline_codes, device)
    dce_q = pack_codes(dce_codes, device)
    # Validate packed code layout by checking untouched code round-trip against
    # original codes, then let bitsandbytes perform the actual dequantization.
    baseline_dequant = dequantize_4bit(bnbaseline_q, quant_state=qstate).float().cpu().flatten()
    dce_dequant = dequantize_4bit(dce_q, quant_state=qstate).float().cpu().flatten()
    b_codes_after = unpack_codes(bnbaseline_q, original.numel())
    d_codes_after = unpack_codes(dce_q, original.numel())
    baseline_observed = [b_codes_after[p] & 1 for p in positions]
    dce_observed = [d_codes_after[p] & 1 for p in positions]
    b_err, d_err = ber(bits, baseline_observed), ber(bits, dce_observed)
    target_codes = [original_codes[p] for p in positions]
    baseline_selected = [b_codes_after[p] for p in positions]
    dce_selected = [d_codes_after[p] for p in positions]
    report["candidates"]["B_DCE"] = {
        **d_err,
        "baseline_nearest_feasible": {
            **b_err,
            "weight_rmse_vs_original_fp16": float(torch.mean((baseline_dequant-original)**2).sqrt().item()),
            "weight_rmse_vs_clean_nf4": float(torch.mean((baseline_dequant-base)**2).sqrt().item()),
            "carrier_code_histogram_tv_vs_original": tv_from_counts(hist(target_codes), hist(baseline_selected), n_bits),
        },
        "weight_rmse_vs_original_fp16": float(torch.mean((dce_dequant-original)**2).sqrt().item()),
        "weight_rmse_vs_clean_nf4": float(torch.mean((dce_dequant-base)**2).sqrt().item()),
        "carrier_code_histogram_tv_vs_original": tv_from_counts(hist(target_codes), hist(dce_selected), n_bits),
        "dce_lambda": args.dce_lambda,
        "receiver_side_information": "NF4 packed code indices and keyed carrier positions; no FP16 cover needed to decode parity",
        "interpretation": "real NF4 codebook/packed-code tensor pilot; no save/reload or model-utility evaluation in this stage",
    }
    report["checks"] = {
        "clean_nf4_tensor_shape_matches": base.numel() == original.numel(),
        "candidate_b_packed_code_roundtrip_matches_selected_codes": all(
            d_codes_after[p] == dce_codes[p] for p in positions
        ),
        "candidate_b_receiver_decodes_payload": d_err["bit_errors"] == 0,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report: {report_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local FP16/BF16 HF model directory or cached model ID")
    parser.add_argument("--tensor", default="model.layers.0.self_attn.q_proj.weight")
    parser.add_argument("--output", required=True, help="New JSON path; existing files are never overwritten")
    parser.add_argument("--payload-bits", type=int, default=10000)
    parser.add_argument("--max-values", type=int, default=0, help="0 means use full tensor")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--blocksize", type=int, default=64)
    parser.add_argument("--qse-margin", type=float, default=0.25)
    parser.add_argument("--dce-lambda", type=float, default=0.01)
    parser.add_argument("--test-key-hex", default="00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.payload_bits < 1 or args.blocksize < 1 or args.qse_margin < 0 or args.dce_lambda < 0:
        parser.error("payload-bits/blocksize must be positive; qse-margin and dce-lambda must be non-negative")
    try:
        run(args)
    except Exception as exc:
        print(f"REAL_NF4_CANDIDATE_PILOT_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
