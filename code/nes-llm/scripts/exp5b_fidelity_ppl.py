"""
Experiment 5B — Real Perplexity Fidelity Validation

Measures actual WikiText-2 perplexity degradation after NES embedding.

Target:
    PPL degradation < 2%
"""

import os
import traceback
# Must be set before importing torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from datasets import load_dataset
# from src.model.model_loader import apply_residuals_to_model
from src.model.model_loader import (
    load_model_pair,
    extract_residuals,
)

from src.evaluation.exp5_model_builder import (
    build_embedded_eval_model,
)

from src.evaluation.fidelity_validator import FidelityValidator
from src.embedding.intelligent_embedder import IntelligentEmbedder
from src.core.types import EmbeddingConfig


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# ============================================================
# MODELS
# ============================================================

MODELS = [
    # ("meta-llama/Llama-3.1-8B", "llama"),
    # ("mistralai/Mistral-7B-v0.3", "mistral"),
    # ("google/gemma-2-9b", "gemma"),
    # ("Qwen/Qwen2.5-7B", "qwen"),
    ("Qwen/Qwen2.5-3B", "qwen"),
    # ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llama"),
    # ("microsoft/Phi-3-mini-4k-instruct", "phi3"),
]


# ============================================================
# LOAD WIKITEXT-2
# ============================================================

print("\nLoading WikiText-2 validation set...")

dataset = load_dataset(
    "wikitext",
    "wikitext-2-raw-v1",
    split="validation",
)

# Guide requirement:
# Keep texts longer than 50 characters and use first 200.
texts = [
    text
    for text in dataset["text"]
    if len(text.strip()) > 50
][:200]

print(f"Evaluation samples: {len(texts)}")


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

PAYLOAD_BITS = 50_000

MESSAGE = "A" * 6_000

validator = FidelityValidator(
    max_ppl_degradation=0.02
)


# ============================================================
# RUN EXPERIMENT
# ============================================================

for model_id, family in MODELS:

    print("\n" + "=" * 70)
    print(f"MODEL: {model_id}")
    print("=" * 70)

    nf4 = None
    fp16 = None
    tok = None
    embedded_model = None

    try:

        # --------------------------------------------------------
        # 1. Load NF4 + FP16 model pair
        # --------------------------------------------------------
        # Time Complexity: O(model size)
        nf4, fp16, tok = load_model_pair(
            model_id,
            device=DEVICE,
        )

        # --------------------------------------------------------
        # 2. Detect actual architecture depth
        # --------------------------------------------------------
        # Time Complexity: O(1)
        actual_layers = len(nf4.model.layers)

        print(f"Detected layers: {actual_layers}")

        # --------------------------------------------------------
        # 3. Extract quantization residuals
        # --------------------------------------------------------
        # Time Complexity: O(L * W)
        residuals = extract_residuals(
            nf4,
            fp16,
            family,
        )

        print(f"Residual tensors: {len(residuals)}")

        # --------------------------------------------------------
        # 4. Baseline perplexity
        # --------------------------------------------------------
        # Time Complexity:
        # O(N * forward_pass)
        # N = number of WikiText samples
        print("\nCalculating baseline PPL...")

        ppl_base = validator.validate_perplexity(
            nf4,
            tok,
            texts,
        )

        print(f"Baseline PPL: {ppl_base:.4f}")
            # --------------------------------------------------------
        # 4B. Reconstruction control test
        # --------------------------------------------------------
        # Reconstruct using ORIGINAL residuals.
        # No embedding is applied here.
        #
        # W_reconstructed = W_NF4 + R_original
        #
        # This tells us whether model reconstruction itself
        # changes perplexity.

        print("\nBuilding reconstruction control model...")

        reconstructed_model = build_embedded_eval_model(
            nf4,
            fp16,
            residuals,
            family,
        )

        print("\nCalculating reconstructed-model PPL...")

        ppl_reconstructed = validator.validate_perplexity(
            reconstructed_model,
            tok,
            texts,
        )

        control_diff = (
            (ppl_reconstructed - ppl_base)
            / ppl_base
        ) * 100

        print(f"Reconstructed PPL: {ppl_reconstructed:.4f}")
        print(f"Control difference: {control_diff:.3f}%")
       # --------------------------------------------------------
        # 4C. Reload fresh model pair
        # --------------------------------------------------------

        del reconstructed_model
        del fp16
        del nf4

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        print("\nReloading fresh NF4 + FP16 pair for embedding...")

        nf4, fp16, tok = load_model_pair(
            model_id,
            device=DEVICE,
        )
        print("\nEmbedding payload...")
        # --------------------------------------------------------
        # 5. Configure NES embedding
        # --------------------------------------------------------
        # Time Complexity: O(1)
        config = EmbeddingConfig(
            total_payload_bits=PAYLOAD_BITS,
            model_family=family,
            num_hidden_layers=actual_layers,
        )

        embedder = IntelligentEmbedder(config)

        # --------------------------------------------------------
        # 6. Embed 50,000-bit payload
        # --------------------------------------------------------
        # Time Complexity: O(total residual elements)
        print("\nEmbedding payload...")

        embed_result = embedder.embed(
            MESSAGE,
            residuals,
        )
        # --------------------------------------------------------
        # 6B. Measure embedding distortion
        # --------------------------------------------------------
        # Time Complexity: O(total residual elements)
        total_changed = 0
        max_change = 0.0
        sum_change = 0.0

        for layer_id in residuals:

            original = residuals[layer_id]
            embedded = embed_result.embedded_residuals[layer_id]

            delta = (embedded - original).abs()

            total_changed += (delta > 0).sum().item()
            max_change = max(max_change, delta.max().item())
            sum_change += delta.sum().item()

        print("\nEmbedding distortion:")
        print(f"  Changed values : {total_changed:,}")
        print(f"  Max |delta|    : {max_change:.8f}")
        print(
            f"  Mean |delta|   : "
            f"{sum_change / max(total_changed, 1):.8f}"
        )
        # --------------------------------------------------------
        # 7. Build Exp5 embedded evaluation model
        # --------------------------------------------------------
        #
        # IMPORTANT:
        # We do NOT modify the NF4 model.
        #
        # For each affected layer:
        #
        #   W_embedded = W_NF4 + R_embedded
        #
        # The resulting FP16 weights are placed into the
        # separate FP16 evaluation model.
        #
        # Time Complexity: O(L * W)
        #
        print("\nBuilding embedded evaluation model...")

        embedded_model = build_embedded_eval_model(
            nf4,
            fp16,
            embed_result.embedded_residuals,
            family,
        )

        # --------------------------------------------------------
        # 8. Embedded-model perplexity
        # --------------------------------------------------------
        # Time Complexity:
        # O(N * forward_pass)
        print("\nCalculating embedded-model PPL...")

        ppl_embed = validator.validate_perplexity(
            embedded_model,
            tok,
            texts,
        )

        print(f"Embedded PPL: {ppl_embed:.4f}")

        # --------------------------------------------------------
        # 9. Compare PPL
        # --------------------------------------------------------
        # Time Complexity: O(1)
        result = validator.compare_perplexity(
            ppl_base,
            ppl_embed,
        )

        # --------------------------------------------------------
        # 10. Report
        # --------------------------------------------------------
        print("\n" + "-" * 70)
        print(f"Model              : {model_id}")
        print(f"Payload            : {PAYLOAD_BITS:,} bits")
        print(f"Baseline PPL       : {ppl_base:.4f}")
        print(f"Embedded PPL       : {ppl_embed:.4f}")
        print(
            f"PPL degradation   : "
            f"{result.ppl_degradation * 100:.3f}%"
        )
        print(f"Verdict            : {result.status}")
        print("-" * 70)

    except Exception as exc:

        print(
            f"\nERROR: {type(exc).__name__}: {exc}"
        )

        traceback.print_exc()

    finally:

        # --------------------------------------------------------
        # 11. Release model memory
        # --------------------------------------------------------
        del embedded_model
        del nf4
        del fp16
        del tok

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        print("\nMemory cleared.")