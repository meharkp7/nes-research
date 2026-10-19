#!/usr/bin/env python3
"""Qwen-first multi-string pilot for the five existing residual-stream methods.

This writes a *residual-tensor experiment artifact*, not a Hugging Face model
checkpoint. It is a stepping stone for validating the shared corpus format,
method-matched extraction, selected-layer allocation and fresh-process reload.
QSE/DCE use the separate packed-NF4 lane and are not silently routed here.

Example:
  python scripts/seven_method_residual_pilot.py embed \
    --model Qwen/Qwen2.5-3B --method lwe_grid_parity \
    --message "first string" --message "नमस्ते 🌍" \
    --output ../artifacts/lwe-pilot.pt --corpus-out ../artifacts/lwe-pilot.expected.bin

  python scripts/seven_method_residual_pilot.py extract \
    --artifact ../artifacts/lwe-pilot.pt \
    --expected-corpus ../artifacts/lwe-pilot.expected.bin
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.types import EmbeddingConfig  # noqa: E402
from src.carrier_intelligence.qaci_pipeline import QACIPipeline  # noqa: E402
from src.embedding.strategy_registry import build, extract_with  # noqa: E402
from src.experiments.model_context import make_context  # noqa: E402
from src.experiments.residual_source import _cache_for  # noqa: E402
from src.experiments.seven_method_protocol import (  # noqa: E402
    MessageRecord, bits_to_bytes, bytes_to_bits, corpus_summary, decode_corpus,
    encode_corpus, load_jsonl, normalize_records,
)

RESIDUAL_METHODS = {
    "sign": "sign",
    "magnitude_aware": "magnitude_aware",
    "qae": "qae",
    "lwe_grid_parity": "lwe",
    "split_sign_parity": "split",
}


def _records(args: argparse.Namespace) -> list[MessageRecord]:
    records: list[MessageRecord] = []
    if args.messages_file:
        records.extend(load_jsonl(args.messages_file))
    records.extend(
        MessageRecord(f"cli-{index + 1:04d}", text)
        for index, text in enumerate(args.message or [])
    )
    rows = normalize_records(records)
    if not rows:
        raise ValueError("Provide at least one --message or --messages-file record.")
    return rows


def _parse_layers(value: str, expected_layers: int) -> list[int]:
    try:
        layers = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--layers must be a comma-separated list of integers") from exc
    if not layers:
        raise ValueError("at least one layer must be selected")
    if len(set(layers)) != len(layers):
        raise ValueError("--layers contains duplicate layer IDs")
    if any(layer < 0 or layer >= expected_layers for layer in layers):
        raise ValueError(f"layer IDs must be in [0, {expected_layers - 1}]")
    return layers


def _config_dict(config: EmbeddingConfig) -> dict[str, Any]:
    return {
        "total_payload_bits": config.total_payload_bits,
        "embedding_strategy": config.embedding_strategy,
        "carrier_selection": config.carrier_selection,
        "use_qaci": config.use_qaci,
        "model_family": config.model_family,
        "num_hidden_layers": config.num_hidden_layers,
        "alpha": config.alpha,
        "noise_level": config.noise_level,
        "min_magnitude": config.min_magnitude,
        "percentile_threshold": config.percentile_threshold,
        "gamma": config.gamma,
        "split_fraction": config.split_fraction,
        "lwe_width_rule": config.lwe_width_rule,
    }


def _config_from_dict(raw: dict[str, Any]) -> EmbeddingConfig:
    return EmbeddingConfig(**raw)


def _load_selected_residuals(model_id: str, expected_layers: int, layers: list[int]):
    cache = _cache_for(model_id)
    if not cache.is_complete(expected_layers):
        raise RuntimeError(
            f"Residual cache incomplete for {model_id}; expected {expected_layers} layers "
            f"under {cache.model_cache_dir}"
        )
    residuals = {}
    for layer_id in layers:
        row = torch.load(cache._layer_path(layer_id), map_location="cpu", weights_only=True)
        residuals[layer_id] = row["residual"].detach().cpu().contiguous()
        del row
    return residuals, str(cache.model_cache_dir)


def _select_layer_ids_from_profiles(profiles: list[dict[str, Any]]) -> list[int]:
    """Select a data-dependent quality cohort using the observed median score.

    Raw parameter count is not a useful stopping condition here: a single
    transformer residual layer can have far more parameters than the payload,
    causing the old greedy loop to select exactly one layer in practice.
    Keep the above-median quality cohort instead, then let QACI allocate bits
    across that cohort. No layer IDs or cohort size are fixed in advance.
    """
    if not profiles:
        raise ValueError("cannot select layers from an empty profile list")
    scores = sorted(float(profile["quality_score"]) for profile in profiles)
    middle = len(scores) // 2
    threshold = (
        scores[middle]
        if len(scores) % 2
        else (scores[middle - 1] + scores[middle]) / 2.0
    )
    selected = [
        int(profile["layer_id"])
        for profile in profiles
        if float(profile["quality_score"]) >= threshold
    ]
    if not selected:
        # Defensive fallback for unusual non-finite/custom profile values.
        best = max(profiles, key=lambda p: float(p["quality_score"]))
        selected = [int(best["layer_id"])]
    return sorted(set(selected))


def _select_layers_automatically(model_id: str, expected_layers: int, total_payload_bits: int, gamma: float):
    """Profile every cached layer and select the observed above-median QACI cohort."""
    from src.carrier_intelligence.layer_profiler import LayerProfiler
    cache = _cache_for(model_id)
    if not cache.is_complete(expected_layers):
        raise RuntimeError(f"Residual cache incomplete for {model_id}; expected {expected_layers} layers under {cache.model_cache_dir}")
    profiler = LayerProfiler()
    profiles = []
    for layer_id in range(expected_layers):
        row = torch.load(cache._layer_path(layer_id), map_location="cpu", weights_only=True)
        residual = row["residual"].detach().cpu().contiguous()
        profiles.append(profiler.profile(residual, layer_id, "residual", expected_layers))
        del residual, row
    selected = _select_layer_ids_from_profiles(profiles)
    selected_capacity = sum(
        int(profile["num_params"])
        for profile in profiles
        if int(profile["layer_id"]) in set(selected)
    )
    if selected_capacity < total_payload_bits:
        raise ValueError(
            f"Payload needs {total_payload_bits} bits; selected quality cohort "
            f"provides only {selected_capacity} parameters"
        )
    profile_map = {int(p["layer_id"]): p for p in profiles}
    return selected, profile_map, str(cache.model_cache_dir)

def run_embed(args: argparse.Namespace) -> dict:
    method = args.method
    registry_name = RESIDUAL_METHODS[method]
    context = make_context(args.model)
    rows = _records(args)
    corpus = encode_corpus(rows)
    bits = bytes_to_bits(corpus)
    if args.layers.strip().lower() == "auto":
        layers, layer_profile_map, cache_dir = _select_layers_automatically(
            args.model, context.expected_layers, len(bits), args.gamma
        )
    else:
        layers = _parse_layers(args.layers, context.expected_layers)
        layer_profile_map, cache_dir = {}, ""
    output = args.output.expanduser().resolve()
    corpus_out = args.corpus_out.expanduser().resolve()
    if output == corpus_out:
        raise ValueError("artifact and corpus output paths must differ")
    if output.exists() or corpus_out.exists():
        raise FileExistsError("Refusing to overwrite existing artifact or corpus output")
    residuals, loaded_cache_dir = _load_selected_residuals(args.model, context.expected_layers, layers)
    cache_dir = loaded_cache_dir or cache_dir

    config = EmbeddingConfig(
        total_payload_bits=len(bits),
        embedding_strategy=registry_name,
        model_family=context.family,
        num_hidden_layers=context.expected_layers,
        gamma=args.gamma,
        min_magnitude=args.min_magnitude,
        split_fraction=args.split_fraction,
        lwe_width_rule=args.lwe_width_rule,
    )
    selection = QACIPipeline(
        total_layers=context.expected_layers, gamma=args.gamma
    ).select(residuals, total_payload_bits=len(bits), use_position_bias=False)
    strategy = build(config, registry_name)
    result = strategy.embed(residuals, bits, selection.selected_indices)
    if not result.success or result.bits_embedded != len(bits):
        raise RuntimeError(
            f"incomplete embedding: requested {len(bits)} bits, "
            f"embedded {result.bits_embedded}; increase selected layers/capacity"
        )
    used = sum(len(indices) for indices in result.carrier_indices.values())
    if used != len(bits):
        raise RuntimeError(f"carrier count {used} does not equal payload bits {len(bits)}")

    report = {
        "schema": "nes.seven_method_residual_artifact.v1",
        "artifact_kind": "residual_tensor_bundle_not_model_checkpoint",
        "method": method,
        "registry_method": registry_name,
        "source_model_id": args.model,
        "model_family": context.family,
        "expected_model_layers": context.expected_layers,
        "selected_layers": layers,
        "layer_selection": {"mode": "qaci_profile_then_capacity_gated_quality_ranking" if args.layers.strip().lower() == "auto" else "explicit", "profiles": {str(k): v for k, v in layer_profile_map.items()}, "selection_note": "All layers profiled one at a time; selected the observed above-median quality cohort using measured QACI quality without positional bias. This avoids treating raw parameter count as a reason to stop after one layer; cohort size is data-dependent, not fixed." if args.layers.strip().lower() == "auto" else "Explicit layer IDs supplied."},
        "cache_dir": cache_dir,
        "config": _config_dict(config),
        "corpus": corpus_summary(rows),
        "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
        "payload_bits": len(bits),
        "carrier_count": used,
        "carrier_allocation_by_layer": {
            str(layer): len(indices) for layer, indices in result.carrier_indices.items()
        },
        "receiver_contract": (
            "extract from saved residual tensors and carrier metadata; no original residual cover"
        ),
        "security_note": (
            "research pilot only; default registry key/config is not a cryptographic security claim"
        ),
    }
    artifact = {
        "schema": report["schema"],
        "method": method,
        "registry_method": registry_name,
        "config": report["config"],
        "metadata": report,
        "embedded_residuals": result.embedded_weights,
        "carrier_indices": result.carrier_indices,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    corpus_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with corpus_out.open("xb") as handle:
            handle.write(corpus)
        with output.open("xb") as handle:
            torch.save(artifact, handle)
    except Exception:
        output.unlink(missing_ok=True)
        corpus_out.unlink(missing_ok=True)
        raise
    print(json.dumps({
        "status": "embedded",
        "method": method,
        "model": args.model,
        "layers": layers,
        "messages": len(rows),
        "payload_bits": len(bits),
        "artifact": str(output),
        "expected_corpus": str(corpus_out),
        "artifact_kind": report["artifact_kind"],
    }, indent=2))
    return report


def run_extract(args: argparse.Namespace) -> dict:
    artifact_path = args.artifact.expanduser().resolve()
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if artifact.get("schema") != "nes.seven_method_residual_artifact.v1":
        raise ValueError("unsupported residual artifact schema")
    registry_name = artifact["registry_method"]
    config = _config_from_dict(artifact["config"])
    strategy = build(config, registry_name)
    recovered_bits = extract_with(
        strategy,
        artifact["embedded_residuals"],
        artifact["carrier_indices"],
        residuals_ref=None,
        strategy_name=registry_name,
    )
    if len(recovered_bits) != config.total_payload_bits:
        raise ValueError(
            f"decoder returned {len(recovered_bits)} bits, expected {config.total_payload_bits}"
        )
    recovered_corpus = bits_to_bytes(recovered_bits)
    metadata = artifact["metadata"]
    recovered_sha = hashlib.sha256(recovered_corpus).hexdigest()
    internal_digest_match = recovered_sha == metadata["corpus_sha256"]
    try:
        rows = decode_corpus(recovered_corpus)
        decode_error = None
        recovered_messages = [
            {"id": row.id, "utf8_bytes": len(row.text.encode("utf-8"))}
            for row in rows
        ]
    except (ValueError, UnicodeDecodeError) as exc:
        rows = []
        decode_error = f"{type(exc).__name__}: {exc}"
        recovered_messages = []
    report: dict[str, Any] = {
        "schema": "nes.seven_method_residual_extract.v1",
        "artifact": str(artifact_path),
        "method": artifact["method"],
        "registry_method": registry_name,
        "source_model_id": metadata["source_model_id"],
        "selected_layers": metadata["selected_layers"],
        "message_count": len(rows) if decode_error is None else None,
        "recovered_messages": recovered_messages,
        "decode_error": decode_error,
        "recovered_corpus_sha256": recovered_sha,
        "embedded_manifest_digest_match": internal_digest_match,
        "artifact_kind": metadata["artifact_kind"],
        "message_texts_written_to_report": False,
    }
    if args.expected_corpus:
        expected_path = args.expected_corpus.expanduser().resolve()
        expected = expected_path.read_bytes()
        expected_bits = bytes_to_bits(expected)
        common = min(len(expected_bits), len(recovered_bits))
        errors = sum(a != b for a, b in zip(expected_bits[:common], recovered_bits[:common]))
        errors += abs(len(expected_bits) - len(recovered_bits))
        denominator = max(len(expected_bits), len(recovered_bits), 1)
        report.update({
            "expected_corpus": str(expected_path),
            "expected_bits": len(expected_bits),
            "recovered_bits": len(recovered_bits),
            "bit_errors": errors,
            "ber": errors / denominator,
            "exact_match": expected == recovered_corpus,
        })
    if args.recovered_corpus_out:
        recovered_path = args.recovered_corpus_out.expanduser().resolve()
        if recovered_path.exists():
            raise FileExistsError(f"Refusing to overwrite {recovered_path}")
        recovered_path.parent.mkdir(parents=True, exist_ok=True)
        with recovered_path.open("xb") as handle:
            handle.write(recovered_corpus)
        report["recovered_corpus_out"] = str(recovered_path)
    if args.report:
        report_path = args.report.expanduser().resolve()
        if report_path.exists():
            raise FileExistsError(f"Refusing to overwrite {report_path}")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not internal_digest_match or report.get("exact_match") is False:
        raise SystemExit(2)
    return report


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    embed = sub.add_parser("embed", help="embed a framed multi-string corpus in selected residual layers")
    embed.add_argument("--model", default="Qwen/Qwen2.5-3B")
    embed.add_argument("--method", choices=RESIDUAL_METHODS, required=True)
    embed.add_argument("--message", action="append", default=[])
    embed.add_argument("--messages-file", type=Path)
    embed.add_argument("--layers", default="auto", help="auto profiles all cached layers and selects by QACI quality/capacity; explicit comma-separated IDs are for controlled ablations")
    embed.add_argument("--output", type=Path, required=True)
    embed.add_argument("--corpus-out", type=Path, required=True)
    embed.add_argument("--gamma", type=float, default=2.5)
    embed.add_argument("--min-magnitude", type=float, default=0.001)
    embed.add_argument("--split-fraction", type=float, default=0.5)
    embed.add_argument("--lwe-width-rule", choices=("global", "per_layer", "layer_rank"), default="global")
    extract = sub.add_parser("extract", help="reload an artifact and recover/verify its framed corpus")
    extract.add_argument("--artifact", type=Path, required=True)
    extract.add_argument("--expected-corpus", type=Path)
    extract.add_argument("--recovered-corpus-out", type=Path)
    extract.add_argument("--report", type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "embed":
        run_embed(args)
        return 0

    try:
        run_extract(args)
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        # Keep extraction failures machine-readable across the subprocess
        # boundary. Previously exceptions before report construction (e.g.
        # artifact reload, strategy construction, or decoder errors) produced
        # no JSON report, leaving the matrix with BER=None and little context.
        failure = {
            "schema": "nes.seven_method_residual_extract.v1",
            "status": "EXTRACT_EXCEPTION",
            "artifact": str(args.artifact.expanduser().resolve()),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "exact_match": None,
            "ber": None,
        }
        if args.report:
            report_path = args.report.expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            if not report_path.exists():
                report_path.write_text(
                    json.dumps(failure, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
        print(json.dumps(failure, indent=2, ensure_ascii=False), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
