"""
Real perplexity experiment — WikiText-2 PPL before and after embedding.

This produces the actual perplexity numbers for Table 3 in the paper.
Requires a model + tokenizer (not synthetic residuals).
"""

import math
from typing import Dict, List, Optional, Tuple
import torch


class PerplexityExperiment:
    """
    Measures WikiText-2 perplexity before and after embedding.

    Usage:
        exp = PerplexityExperiment(model, tokenizer)
        result = exp.run(embed_fn, n_samples=500)
        print(result.report())
    """

    def __init__(
        self,
        model,
        tokenizer,
        device:     str = "cpu",
        max_length: int = 512,
        batch_size: int = 4,
    ):
        self.model      = model
        self.tokenizer  = tokenizer
        self.device     = device
        self.max_length = max_length
        self.batch_size = batch_size

    def load_wikitext2(self, n_samples: int = 500) -> List[str]:
        """Load WikiText-2 test split."""
        try:
            from datasets import load_dataset
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            texts   = [
                x["text"] for x in dataset
                if len(x["text"].strip()) > 100
            ][:n_samples]
            print(f"[PPLExp] Loaded {len(texts)} WikiText-2 samples")
            return texts
        except Exception as e:
            print(f"[PPLExp] Could not load WikiText-2: {e}")
            print("[PPLExp] Using synthetic text samples instead")
            return [
                "The quick brown fox jumps over the lazy dog. " * 10
            ] * min(n_samples, 50)

    def compute_perplexity(self, texts: List[str]) -> float:
        self.model.eval()
        total_loss   = 0.0
        total_tokens = 0

        # Detect actual model device
        try:
            model_device = next(self.model.parameters()).device
        except Exception:
            model_device = self.device

        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i: i + self.batch_size]
                try:
                    inputs    = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_length,
                        padding=True,
                    )
                    # Move to actual model device (not hardcoded self.device)
                    input_ids = inputs["input_ids"].to(model_device)
                    labels    = input_ids.clone()
                    if "attention_mask" in inputs:
                        labels[inputs["attention_mask"].to(model_device) == 0] = -100

                    outputs  = self.model(input_ids=input_ids, labels=labels)
                    n_tokens = (labels != -100).sum().item()
                    if n_tokens > 0:
                        total_loss   += outputs.loss.item() * n_tokens
                        total_tokens += n_tokens
                except Exception as e:
                    print(f"[PPLExp] Batch {i} failed: {e}")
                    continue

        if total_tokens == 0:
            return float('inf')
        return math.exp(total_loss / total_tokens)

    def run(
        self,
        embed_fn,
        n_samples: int = 500,
        label:     str = "experiment",
    ) -> "PPLResult":
        """
        Run full PPL experiment.

        Args:
            embed_fn: Callable that modifies model weights in-place.
                      Signature: embed_fn(model) → None
            n_samples: Number of WikiText-2 samples to evaluate on.
            label: Experiment label for reporting.

        Returns:
            PPLResult with before/after PPL and degradation.
        """
        texts = self.load_wikitext2(n_samples)

        # Baseline PPL (before embedding)
        print(f"[PPLExp] Computing baseline PPL...")
        ppl_baseline = self.compute_perplexity(texts)
        print(f"[PPLExp] Baseline PPL: {ppl_baseline:.4f}")

        # Apply embedding
        print(f"[PPLExp] Applying embedding...")
        embed_fn(self.model)

        # Embedded PPL (after embedding)
        print(f"[PPLExp] Computing embedded PPL...")
        ppl_embedded = self.compute_perplexity(texts)
        print(f"[PPLExp] Embedded PPL: {ppl_embedded:.4f}")

        degradation = (ppl_embedded - ppl_baseline) / max(ppl_baseline, 1e-8)
        print(f"[PPLExp] Degradation: {degradation*100:.4f}%")

        return PPLResult(
            label=        label,
            ppl_baseline= ppl_baseline,
            ppl_embedded= ppl_embedded,
            degradation=  degradation,
            n_samples=    len(texts),
        )


class PPLResult:
    """Result of a perplexity experiment."""

    def __init__(
        self,
        label:        str,
        ppl_baseline: float,
        ppl_embedded: float,
        degradation:  float,
        n_samples:    int,
    ):
        self.label        = label
        self.ppl_baseline = ppl_baseline
        self.ppl_embedded = ppl_embedded
        self.degradation  = degradation
        self.n_samples    = n_samples

    @property
    def passed(self) -> bool:
        return self.degradation <= 0.02   # < 2%

    def report(self) -> str:
        return "\n".join([
            f"PPL Experiment: {self.label}",
            f"  Samples evaluated : {self.n_samples}",
            f"  Baseline PPL      : {self.ppl_baseline:.4f}",
            f"  Embedded PPL      : {self.ppl_embedded:.4f}",
            f"  Degradation       : {self.degradation*100:.4f}%",
            f"  Status            : {'PASS' if self.passed else 'FAIL'}",
        ])