"""Paired utility evaluation on 32 fixed prompts held out from the prior 8-text diagnostic.

The suite is project-curated, not an external benchmark. The script evaluates original
and stego checkpoints sequentially, reports paired per-prompt NLL differences and a
deterministic prompt-level bootstrap interval, and applies a predeclared 2% PPL
non-inferiority threshold. This remains exploratory model-fidelity evidence, not task utility.

Run from nes-llm/ after confirming artifact recovery and local tokenizer availability:
../.venv/bin/python scripts/contract_b_utility_eval_heldout.py \
  --original ../cache/contract_b_nf4_probe_retry \
  --stego ../cache/contract_b_nf4_b14_10k \
  --tokenizer-path Qwen/Qwen2.5-3B --device cpu --max-tokens 256 \
  --output ../cache/contract_b_utility_heldout_20261010.json

The output path must be new. The 2% threshold is a project-defined criterion, not a
universal standard; freeze it before inspecting results.
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

HELDOUT_TEXTS = [
    "Explain why a reproducible machine-learning experiment should record random seeds, software versions, and dataset hashes.",
    "Describe the difference between a model's training loss and its performance on unseen examples.",
    "A database transaction should either complete all required updates or leave the database unchanged. Explain why this property matters.",
    "Write a short explanation of how public-key cryptography differs from symmetric-key cryptography.",
    "An engineer observes that a service becomes slower as request volume increases. List measurements that could help locate the bottleneck.",
    "Summarize the trade-offs between storing data in a relational database and a document database.",
    "Explain why correlation between two variables does not, by itself, establish that one causes the other.",
    "A city is considering replacing some car lanes with protected bicycle lanes. Identify two potential benefits and two implementation concerns.",
    "Describe a simple method for checking whether a CSV file contains missing values, duplicate rows, and unexpected column types.",
    "Explain the role of a validation set when selecting hyperparameters for a machine-learning model.",
    "A secure password reset flow should avoid revealing whether an email address belongs to an account. Explain the reasoning.",
    "Compare unit tests, integration tests, and end-to-end tests using a software service as an example.",
    "A team reports an improvement after changing five parts of an algorithm simultaneously. Explain why it is difficult to identify the cause.",
    "Describe how confidence intervals help communicate uncertainty in an experimental estimate.",
    "Explain why a model can have high overall accuracy while performing poorly on a rare but important class.",
    "A logistics planner must deliver packages to multiple destinations while minimizing travel distance. Name two constraints that may affect the route.",
    "Describe the purpose of version control branches during collaborative software development.",
    "Explain why a checksum can detect accidental data corruption but does not automatically provide authenticity.",
    "A sensor produces occasional extreme readings. Describe a careful process for investigating the readings before removing them.",
    "Compare precision and recall for a classifier that identifies fraudulent transactions.",
    "Explain why an API should validate incoming data even when the client interface already validates it.",
    "A researcher wants to compare two language-model checkpoints. Explain why the tokenizer and evaluation text should be held constant.",
    "Describe one advantage and one limitation of using synthetic data to test a machine-learning pipeline.",
    "A team is deploying a model to a device with limited memory and compute. Name practical optimizations and their possible costs.",
    "Explain why a benchmark should specify its evaluation split and prevent training data from leaking into the test set.",
    "Describe how caching can improve application latency and one problem that stale cached values may cause.",
    "A hospital scheduling system must account for staff availability, room capacity, and urgent cases. Explain why these constraints interact.",
    "Explain the difference between encryption and hashing, and give one appropriate use for each.",
    "A developer receives an intermittent failure that cannot be reproduced locally. List useful information to collect from the failing environment.",
    "Describe why a model's average score may hide meaningful variation across demographic groups or task categories.",
    "Explain how an ablation study can test whether a component contributes to a system's measured performance.",
    "A monitoring system reports an increase in false alarms after a threshold change. Describe how to evaluate whether the change was beneficial.",
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
            for idx, text in enumerate(HELDOUT_TEXTS):
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


def bootstrap_prompt_delta_ci(original_result: dict, stego_result: dict, *, seed: int = 20261010, n_boot: int = 10000) -> dict:
    """Percent change in PPL from equally weighted per-prompt mean NLL deltas."""
    import random
    orig = {row["text_index"]: row["mean_nll"] for row in original_result["per_text"]}
    steg = {row["text_index"]: row["mean_nll"] for row in stego_result["per_text"]}
    if orig.keys() != steg.keys() or len(orig) < 2:
        raise ValueError("Paired prompt indices are missing or mismatched")
    paired = [steg[i] - orig[i] for i in sorted(orig)]
    rng = random.Random(seed)
    estimates = []
    n = len(paired)
    for _ in range(n_boot):
        sample = [paired[rng.randrange(n)] for _ in range(n)]
        estimates.append((math.exp(statistics.mean(sample)) - 1.0) * 100.0)
    estimates.sort()
    lo = estimates[int(0.025 * n_boot)]
    hi = estimates[min(n_boot - 1, int(0.975 * n_boot))]
    point = (math.exp(statistics.mean(paired)) - 1.0) * 100.0
    return {
        "method": "paired nonparametric bootstrap resampling prompts with replacement",
        "seed": seed,
        "replicates": n_boot,
        "prompt_count": n,
        "mean_prompt_level_ppl_delta_percent": point,
        "ci_95_percent": [lo, hi],
        "unit_of_resampling": "prompt; not token",
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
    parser.add_argument("--acceptance-threshold-percent", type=float, default=2.0, help="Predeclared maximum allowed upper 95% CI for prompt-level relative PPL change; default 2.0 percent.")
    args = parser.parse_args()
    if args.max_tokens < 8:
        parser.error("--max-tokens must be at least 8")
    if args.acceptance_threshold_percent <= 0:
        parser.error("--acceptance-threshold-percent must be positive")
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
    for idx, text in enumerate(HELDOUT_TEXTS):
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
    bootstrap = bootstrap_prompt_delta_ci(original_result, stego_result)
    upper_ci = bootstrap["ci_95_percent"][1]
    accepted = upper_ci <= args.acceptance_threshold_percent
    report = {
        "stage": "Contract-B-heldout-utility-evaluation",
        "acceptance_threshold_percent": args.acceptance_threshold_percent,
        "acceptance_rule": "PASS only if upper endpoint of paired prompt-level 95% bootstrap CI for relative PPL change is <= the predeclared threshold.",
        "acceptance_status": "PASS" if accepted else "FAIL_OR_INCONCLUSIVE",
        "paired_prompt_bootstrap": bootstrap,
        "status": "MEASURED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "max_tokens_per_text": args.max_tokens,
        "tokenizer_source": tokenizer_source,
        "tokenizer_class": type(tokenizer).__name__,
        "token_lengths": token_lengths,
        "heldout_prompt_count": len(HELDOUT_TEXTS),
        "corpus_design": "32 fixed project-curated prompts not present in the original 8-text diagnostic; not an external benchmark and not evidence of task utility.",
        "diagnostic_corpus_sha256": hashlib.sha256("\n".join(HELDOUT_TEXTS).encode()).hexdigest(),
        "original": original_result,
        "stego": stego_result,
        "perplexity_delta_percent": delta,
        "mean_nll_delta": stego_result["mean_nll"] - original_result["mean_nll"],
        "torch_version": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "interpretation": (
            "Fixed project-curated prompt suite held out from the earlier 8-text diagnostic. "
            "The prompt-level bootstrap quantifies variation across these prompts only; it does "
            "not establish external benchmark performance or task utility. The 2% threshold is "
            "project-defined. Interpret aggregate token-weighted PPL change alongside the "
            "prompt-level interval and report both, regardless of acceptance status."
        ),
    }
    output = Path(args.output).expanduser().resolve() if args.output else stego.parent / "contract_b_utility_heldout.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[Contract B utility] Original PPL: {original_result['perplexity']:.6f}")
    print(f"[Contract B utility] Stego PPL:    {stego_result['perplexity']:.6f}")
    print(f"[Contract B utility] Delta:         {delta:+.6f}%")
    print(f"[Contract B utility] Prompt-level PPL delta: {bootstrap['mean_prompt_level_ppl_delta_percent']:+.6f}%")
    print(f"[Contract B utility] Paired prompt bootstrap 95% CI: [{bootstrap['ci_95_percent'][0]:+.6f}%, {bootstrap['ci_95_percent'][1]:+.6f}%]")
    print(f"[Contract B utility] Threshold: <= {args.acceptance_threshold_percent:.3f}% upper CI; status: {report['acceptance_status']}")
    print(f"[Contract B utility] Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
