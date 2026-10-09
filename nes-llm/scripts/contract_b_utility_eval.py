#!/usr/bin/env python3
"""Exploratory original-vs-stego perplexity check for Contract B.

This uses a fixed, locally defined diagnostic text suite, not a benchmark dataset.
Run only after B1.4 sender/receiver recovery succeeds. Quantized-model evaluation
runs sequentially to avoid keeping two 3B models in memory at once.

Example from nes-llm:
../.venv/bin/python scripts/contract_b_utility_eval.py \
  --original ../cache/contract_b_nf4_probe_retry \
  --stego ../cache/contract_b_nf4_b14_10k \
  --device cpu --max-tokens 256

The metric is exploratory utility evidence only; do not generalize it to all tasks.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

DIAGNOSTIC_TEXTS = [
    "A neural language model estimates the probability of a token given the preceding context. Evaluation should use the same token sequence for every model variant.",
    "Quantization reduces the number of bits used to represent model weights. The resulting approximation can reduce memory use, but may also change numerical behavior.",
    "A reproducible experiment records its configuration, input data, software versions, output artifacts, and failures. A successful run should be independently verifiable.",
    "The city library opened a new reading room with desks, reference books, and quiet spaces. Visitors can reserve a seat and borrow books using their library cards.",
    "A software test checks a specific property under a defined set of assumptions. Passing one test does not prove that a system is secure or correct in every setting.",
    "The weather station measures temperature, humidity, and wind speed at regular intervals. Researchers compare these observations with forecasts to estimate prediction error.",
    "A model checkpoint contains learned parameters and configuration metadata. A modified checkpoint should be compared with its original version using identical inputs and evaluation settings.",
    "In a controlled study, researchers change one factor at a time and keep the evaluation procedure fixed. Results should include uncertainty and limitations.",
]


def evaluate_checkpoint(checkpoint: Path, tokenizer, device: str, max_tokens: int) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        device_map={"": device},
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    token_nll = 0.0
    token_count = 0
    per_text = []
    try:
        with torch.inference_mode():
            for idx, text in enumerate(DIAGNOSTIC_TEXTS):
                # Use the same explicit encode path as the preflight, then build
                # the batch tensor directly. This avoids tokenizer-call wrapper shape
                # differences across Transformers versions.
                token_ids = tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_tokens,
                )
                input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
                if input_ids.shape[1] < 2:
                    raise ValueError(f"Diagnostic text {idx} tokenized to fewer than 2 tokens")
                output = model(input_ids=input_ids, labels=input_ids)
                loss = float(output.loss.detach().cpu().item())
                if not math.isfinite(loss):
                    raise ValueError(f"Non-finite loss on diagnostic text {idx}: {loss}")
                n_tokens = int(input_ids.shape[1] - 1)
                token_nll += loss * n_tokens
                token_count += n_tokens
                per_text.append({
                    "text_index": idx,
                    "input_tokens": int(input_ids.shape[1]),
                    "scored_tokens": n_tokens,
                    "mean_nll": loss,
                    "perplexity": math.exp(loss),
                })
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    mean_nll = token_nll / token_count
    return {
        "checkpoint": str(checkpoint),
        "token_count": token_count,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "per_text": per_text,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--stego", required=True)
    parser.add_argument(
        "--tokenizer-path",
        default="",
        help="Local model/tokenizer directory or cached model ID. Defaults to --original.",
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.max_tokens < 8:
        parser.error("--max-tokens must be at least 8")
    original = Path(args.original).expanduser().resolve()
    stego = Path(args.stego).expanduser().resolve()
    for path in (original, stego):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if original == stego:
        parser.error("--original and --stego must be different checkpoint directories")

    import torch
    from transformers import AutoTokenizer
    tokenizer_source = args.tokenizer_path.strip() or str(original)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, trust_remote_code=True, local_files_only=True
    )
    # Preflight the entire diagnostic corpus before loading/evaluating both checkpoints.
    token_lengths = []
    for idx, text in enumerate(DIAGNOSTIC_TEXTS):
        ids = tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=args.max_tokens,
        )
        token_lengths.append(len(ids))
        if len(ids) < 2:
            raw_ids = tokenizer.encode(text, add_special_tokens=False)
            raise ValueError(
                "Tokenizer preflight failed before model evaluation: "
                f"text_index={idx}, token_length_with_special_tokens={len(ids)}, "
                f"token_length_without_special_tokens={len(raw_ids)}, "
                f"tokenizer_source={tokenizer_source!r}, "
                f"tokenizer_class={type(tokenizer).__name__}, "
                f"vocab_size={getattr(tokenizer, 'vocab_size', None)}, "
                f"model_max_length={getattr(tokenizer, 'model_max_length', None)}, "
                f"sample={text[:100]!r}. "
                "Check that --tokenizer-path points to the matching Qwen tokenizer files."
            )
    original_result = evaluate_checkpoint(original, tokenizer, args.device, args.max_tokens)
    stego_result = evaluate_checkpoint(stego, tokenizer, args.device, args.max_tokens)
    delta = (stego_result["perplexity"] / original_result["perplexity"] - 1.0) * 100.0
    report = {
        "stage": "Contract-B-utility-diagnostic",
        "status": "MEASURED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "max_tokens_per_text": args.max_tokens,
        "tokenizer_source": tokenizer_source,
        "tokenizer_class": type(tokenizer).__name__,
        "token_lengths": token_lengths,
        "diagnostic_text_count": len(DIAGNOSTIC_TEXTS),
        "diagnostic_corpus_sha256": hashlib.sha256("\n".join(DIAGNOSTIC_TEXTS).encode()).hexdigest(),
        "original": original_result,
        "stego": stego_result,
        "perplexity_delta_percent": delta,
        "mean_nll_delta": stego_result["mean_nll"] - original_result["mean_nll"],
        "torch_version": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "interpretation": (
            "Small fixed diagnostic suite only. This is not a representative benchmark, "
            "does not establish task utility or statistical significance, and should not be "
            "reported as general model-fidelity evidence without a larger held-out corpus."
        ),
    }
    output = Path(args.output).expanduser().resolve() if args.output else stego.parent / "contract_b_utility_diagnostic.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[Contract B utility] Original PPL: {original_result['perplexity']:.6f}")
    print(f"[Contract B utility] Stego PPL:    {stego_result['perplexity']:.6f}")
    print(f"[Contract B utility] Delta:         {delta:+.6f}%")
    print(f"[Contract B utility] Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
