"""
NES Command Line Interface.

Usage:
    # Embed a message into a model
    python -m src.cli embed \
        --message "secret text" \
        --model   meta-llama/Llama-3-8B \
        --output  embedded_model/ \
        --keyfile keys/run1.json

    # Extract a message from an embedded model
    python -m src.cli extract \
        --model   embedded_model/ \
        --keyfile keys/run1.json \
        --keyid   <key_id>

    # Run benchmark gates
    python -m src.cli benchmark --layers 8 --size 10000 --bits 5000

    # Find optimal parameters
    python -m src.cli tune --layers 8 --size 10000

    # W8 — delta-only distribution: embed → ship a PATCH, not a model
    python -m src.cli delta-export \
        --model Qwen/Qwen2.5-3B \
        --message "payload" \
        --delta-out out/patch.nesdelta \
        --key-out out/patch.key
    python -m src.cli delta-inspect --delta out/patch.nesdelta
    python -m src.cli delta-extract \
        --model Qwen/Qwen2.5-3B \
        --delta out/patch.nesdelta \
        --keyfile out/patch.key
"""

import argparse
import json
import sys
import os


def cmd_embed(args):
    """Embed a message into a model's residuals (synthetic demo if no GPU)."""
    import torch
    from src.embedding.intelligent_embedder import IntelligentEmbedder
    from src.crypto.key_manager             import KeyManager
    from src.core.types                     import EmbeddingConfig

    print(f"\n[NES] Embedding message ({len(args.message)} chars)...")

    # Synthetic residuals for demo — replace with real model residuals
    n_layers  = args.layers
    layer_size = args.size
    residuals = {i: torch.randn(layer_size) * 0.05 for i in range(n_layers)}

    config   = EmbeddingConfig(
        total_payload_bits=args.bits,
        embedding_strategy="sign",
    )
    embedder = IntelligentEmbedder(config)
    result   = embedder.embed(args.message, residuals)

    if not result.success:
        print("[NES] ❌ Embedding failed.")
        sys.exit(1)

    # Save key
    km  = KeyManager()
    kid = km.add_key(result.key, model_id=args.model)
    os.makedirs(os.path.dirname(args.keyfile) if os.path.dirname(args.keyfile) else ".", exist_ok=True)
    km.save(args.keyfile)

    print(f"[NES] ✅ Embedding complete.")
    print(f"       Bits embedded : {result.bits_embedded:,}")
    print(f"       Layers used   : {sum(1 for b in result.layer_allocation.values() if b > 0)}")
    print(f"       Key ID        : {kid}")
    print(f"       Key saved to  : {args.keyfile}")

    # Save carrier map alongside key
    carrier_file = args.keyfile.replace(".json", "_carriers.json")
    with open(carrier_file, "w") as f:
        json.dump({
            str(lid): indices
            for lid, indices in result.carrier_indices.items()
        }, f)
    print(f"       Carrier map   : {carrier_file}")


def cmd_extract(args):
    """Extract a message from embedded residuals."""
    import torch
    from src.extraction.decrypt_pipeline import DecryptPipeline
    from src.crypto.key_manager          import KeyManager

    print(f"\n[NES] Extracting message...")

    km  = KeyManager()
    km.load(args.keyfile)
    key = km.get_key(args.keyid)

    # Load carrier map
    carrier_file = args.keyfile.replace(".json", "_carriers.json")
    with open(carrier_file, "r") as f:
        raw = json.load(f)
    carrier_indices = {int(lid): indices for lid, indices in raw.items()}

    # Synthetic residuals — replace with real embedded model residuals
    n   = max(carrier_indices.keys()) + 1
    sz  = max(max(v) for v in carrier_indices.values() if v) + 1
    residuals = {}
    for lid, indices in carrier_indices.items():
        t = torch.zeros(sz)
        # Demo: set positive for all (will decode as all-1s — real use needs actual residuals)
        residuals[lid] = t

    pipeline = DecryptPipeline(key=key)
    message, stats = pipeline.run(residuals, carrier_indices)

    if stats.get("success"):
        print(f"[NES] ✅ Extraction complete.")
        print(f"       Message : {message}")
    else:
        print(f"[NES] ❌ Extraction failed: {stats.get('error')}")
        sys.exit(1)


def cmd_benchmark(args):
    """Run all NES quality gates."""
    from src.evaluation.nes_benchmark import NESBenchmark

    bench  = NESBenchmark(
        n_layers=    args.layers,
        layer_size=  args.size,
        payload_bits=args.bits,
        verbose=     True,
    )
    result = bench.run_all()
    sys.exit(0 if result["all_pass"] else 1)


def cmd_tune(args):
    """Find optimal parameters for given residual shape."""
    import torch
    from src.evaluation.optimal_config_finder import OptimalConfigFinder

    residuals = {i: torch.randn(args.size) * 0.05 for i in range(args.layers)}
    finder    = OptimalConfigFinder(verbose=True)
    config, params = finder.find(residuals)

    out = "configs/optimal.json"
    os.makedirs("configs", exist_ok=True)
    config.to_json(out)
    print(f"\n[NES] Config saved to {out}")


# ======================================================================
# W8 — delta-only distribution (W8.1 recipient tool, W8.2 metadata)
# ======================================================================

def _resolve_family(model_id, family_arg):
    if family_arg:
        return family_arg
    from src.experiments.experiment_registry import family_of

    try:
        return family_of(model_id)
    except Exception:
        sys.exit(
            f"[NES] ❌ unknown model {model_id!r}: pass --family "
            "(it is not in the experiment registry)"
        )


def _load_clean_residuals(model_id, family):
    """The base model: R_clean = W_FP16 - dequant(W_NF4), same pair
    every experiment reads residuals from."""

    from src.model.model_loader import (
        DEVICE,
        extract_residuals,
        load_model_pair,
    )

    print(
        f"[NES] loading base pair {model_id} (family={family})...",
        file=sys.stderr,
    )
    nf4_model, fp16_model, _tokenizer = load_model_pair(
        model_id, device=DEVICE
    )
    return extract_residuals(nf4_model, fp16_model, family)


def cmd_delta_export(args):
    """Sender: embed a message, write the delta + its key file."""
    from src.core.types import EmbeddingConfig
    from src.delta import build_delta, save_delta, verify_delta
    from src.embedding.intelligent_embedder import IntelligentEmbedder

    if args.message_file:
        with open(args.message_file, "r", encoding="utf-8") as f:
            message = f.read()
    else:
        message = args.message
    if not message:
        sys.exit("[NES] ❌ empty message")

    family = _resolve_family(args.model, args.family)
    residuals = _load_clean_residuals(args.model, family)

    config = EmbeddingConfig(
        total_payload_bits=args.bits,
        embedding_strategy="sign",
        model_family=family,
        num_hidden_layers=len(residuals),
    )
    result = IntelligentEmbedder(config).embed(message, residuals)

    payload = build_delta(
        residuals, result, model_id=args.model, family=family
    )
    save_delta(payload, args.delta_out)
    report = verify_delta(payload)

    with open(args.key_out, "w") as f:
        f.write(result.key.hex() + "\n")

    print(f"[NES] ✅ delta written: {args.delta_out}")
    print(f"       carriers      : {report['carrier_count']:,} "
          f"({report['changed_count']:,} values changed)")
    print(f"       payload       : {report['payload_bits']:,} bits")
    print(f"       sha256        : {report['sha256'][:16]}…")
    print(f"       key           : {args.key_out} "
          "(share OUT OF BAND — the delta is not secret, the key is)")


def cmd_delta_inspect(args):
    """Verify a delta's integrity metadata and print the audit report.

    Runs no models: an auditor can check the file before paying for
    the base model."""

    from src.delta import load_delta

    try:
        payload = load_delta(args.delta)
    except Exception as exc:  # corruption, truncation, foreign file
        print(f"[NES] ❌ integrity check FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

    meta = payload["metadata"]
    print(f"[NES] ✅ integrity OK: {args.delta}")
    print(f"       format        : {meta['format']} v{meta['version']}")
    print(f"       model         : {meta['model_id']} ({meta['family']})")
    print(f"       strategy      : {meta['strategy']}  "
          f"key_id={meta['key_id']}")
    print(f"       carriers      : {meta['carrier_count']:,} across "
          f"{meta['layer_count']} layers")
    print(f"       changed values: {meta['changed_count']:,}")
    print(f"       payload length: {meta['payload_bits']:,} bits "
          f"(+ {meta['header_bits']}-bit header)")
    print(f"       sha256        : {meta['sha256']}")
    print(f"       created       : {meta['created_utc']}")


def cmd_delta_extract(args):
    """Recipient: base model + delta + key → payload (W8.1)."""
    from src.delta import (
        load_delta,
        recover_payload,
    )

    # --- key: hex on the command line or hex in a file ---------------
    if args.key and args.keyfile:
        sys.exit("[NES] ❌ pass one of --key or --keyfile, not both")
    if args.key:
        raw = args.key
    elif args.keyfile:
        with open(args.keyfile, "r") as f:
            raw = f.read().strip()
    else:
        sys.exit("[NES] ❌ one of --key (64 hex chars) or --keyfile is required")
    try:
        key = bytes.fromhex(raw)
    except ValueError:
        sys.exit("[NES] ❌ key must be hex (64 characters)")
    if len(key) != 32:
        sys.exit(f"[NES] ❌ key must be 32 bytes, got {len(key)}")

    try:
        payload = load_delta(args.delta)
    except Exception as exc:
        print(f"[NES] ❌ integrity check FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

    family = _resolve_family(args.model, args.family)
    residuals = _load_clean_residuals(args.model, family)

    try:
        message, report = recover_payload(residuals, payload, key)
    except Exception as exc:
        print(f"[NES] ❌ recovery failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Report to stderr, message to stdout: `nes delta-extract … > msg`
    # gives the payload alone.
    print(
        f"[NES] ✅ recovered: integrity OK "
        f"({report['carrier_count']:,} carriers, "
        f"{report['payload_bits_from_header']:,}-bit payload, "
        f"sha256 {report['sha256'][:16]}…)",
        file=sys.stderr,
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(message)
        print(f"[NES] message written to {args.out}", file=sys.stderr)
    else:
        print(message)


def main():
    parser = argparse.ArgumentParser(
        prog="nes",
        description="Neural-Entropic Steganography CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- embed ---
    p_embed = sub.add_parser("embed", help="Embed a message into a model")
    p_embed.add_argument("--message", required=True)
    p_embed.add_argument("--model",   default="llama-3-8b")
    p_embed.add_argument("--output",  default="embedded_model/")
    p_embed.add_argument("--keyfile", default="keys/nes_keys.json")
    p_embed.add_argument("--layers",  type=int, default=32)
    p_embed.add_argument("--size",    type=int, default=10000)
    p_embed.add_argument("--bits",    type=int, default=50000)

    # --- extract ---
    p_ext = sub.add_parser("extract", help="Extract a message from an embedded model")
    p_ext.add_argument("--model",   default="embedded_model/")
    p_ext.add_argument("--keyfile", required=True)
    p_ext.add_argument("--keyid",   required=True)

    # --- benchmark ---
    p_bench = sub.add_parser("benchmark", help="Run all quality gates")
    p_bench.add_argument("--layers", type=int, default=8)
    p_bench.add_argument("--size",   type=int, default=10000)
    p_bench.add_argument("--bits",   type=int, default=5000)

    # --- tune ---
    p_tune = sub.add_parser("tune", help="Find optimal parameters")
    p_tune.add_argument("--layers", type=int, default=8)
    p_tune.add_argument("--size",   type=int, default=10000)

    # --- W8: delta-only distribution ---
    p_dx = sub.add_parser(
        "delta-export",
        help="Sender: embed a message and write delta + key file",
    )
    p_dx.add_argument("--model", required=True,
                      help="HF model id of the base model pair")
    p_dx.add_argument("--family",
                      help="architecture family (default: registry lookup)")
    p_dx.add_argument("--message", help="message text")
    p_dx.add_argument("--message-file", help="read the message from a file")
    p_dx.add_argument("--bits", type=int, default=50_000,
                      help="payload budget passed to the embedder")
    p_dx.add_argument("--delta-out", required=True,
                      help="output delta path (.nesdelta)")
    p_dx.add_argument("--key-out", required=True,
                      help="where to write the hex AES key")

    p_di = sub.add_parser(
        "delta-inspect",
        help="W8.2: verify a delta's integrity metadata (no models)",
    )
    p_di.add_argument("--delta", required=True)

    p_de = sub.add_parser(
        "delta-extract",
        help="W8.1 recipient: base model + delta + key → payload",
    )
    p_de.add_argument("--model", required=True)
    p_de.add_argument("--family")
    p_de.add_argument("--delta", required=True)
    p_de.add_argument("--key", help="64 hex chars (32-byte AES key)")
    p_de.add_argument("--keyfile", help="file containing the hex key")
    p_de.add_argument("--out", help="write the message to this file")

    args = parser.parse_args()
    {
        "embed":        cmd_embed,
        "extract":      cmd_extract,
        "benchmark":    cmd_benchmark,
        "tune":         cmd_tune,
        "delta-export": cmd_delta_export,
        "delta-inspect": cmd_delta_inspect,
        "delta-extract": cmd_delta_extract,
    }[args.command](args)


if __name__ == "__main__":
    main()