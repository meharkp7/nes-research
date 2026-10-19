"""
Experiment 5A — Tensor-Level Fidelity Validation

Measures the distortion introduced by NES embedding using
quantization residual tensors.

This is a fast tensor-level proxy.
Real WikiText-2 perplexity evaluation is performed in Exp5B.
"""

import os
import torch

from src.model.model_loader import load_model_pair, extract_residuals
from src.evaluation.fidelity_validator import FidelityValidator
from src.embedding.intelligent_embedder import IntelligentEmbedder
from src.core.types import EmbeddingConfig


# ============================================================
# DEVICE
# ============================================================

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# ============================================================
# MODELS
# ============================================================

MODELS = [
    ("meta-llama/Llama-3.1-8B", "llama"),
    ("mistralai/Mistral-7B-v0.3", "mistral"),
    ("google/gemma-2-9b", "gemma"),
    ("Qwen/Qwen2.5-7B", "qwen"),
    ("Qwen/Qwen2.5-3B", "qwen"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llama"),
    ("microsoft/Phi-3-mini-4k-instruct", "phi3"),
]


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

PAYLOAD_BITS = 50_000

# 6,000 ASCII characters = 48,000 plaintext bits.
# The embedder handles the actual payload encoding.
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

    nf4_model = None
    fp16_model = None

    try:

        # --------------------------------------------------------
        # 1. Load NF4 + FP16 pair
        # --------------------------------------------------------
        # Time Complexity: O(model size)
        nf4_model, fp16_model, _ = load_model_pair(
            model_id,
            device=DEVICE
        )

        # --------------------------------------------------------
        # 2. Detect actual number of layers
        # --------------------------------------------------------
        # Time Complexity: O(1)
        actual_layers = len(nf4_model.model.layers)

        print(f"Detected layers: {actual_layers}")

        # --------------------------------------------------------
        # 3. Extract quantization residuals
        # --------------------------------------------------------
        # Time Complexity: O(L * W)
        # L = number of layers
        # W = parameters in each selected down_proj matrix
        residuals = extract_residuals(
            nf4_model=nf4_model,
            fp16_model=fp16_model,
            family=family,
        )

        total_capacity = sum(
            tensor.numel()
            for tensor in residuals.values()
        )

        print(f"Residual tensors: {len(residuals)}")
        print(f"Total residual capacity: {total_capacity:,}")

        # --------------------------------------------------------
        # 4. Create embedding configuration
        # --------------------------------------------------------
        # Time Complexity: O(1)
        config = EmbeddingConfig(
            total_payload_bits=PAYLOAD_BITS,
            model_family=family,
            num_hidden_layers=actual_layers,
        )

        # --------------------------------------------------------
        # 5. Embed payload
        # --------------------------------------------------------
        # Time Complexity: O(total residual elements)
        embedder = IntelligentEmbedder(config)

        embed_result = embedder.embed(
            MESSAGE,
            residuals,
        )

        # --------------------------------------------------------
        # 6. Validate tensor fidelity
        # --------------------------------------------------------
        # Time Complexity: O(total residual elements)
        result = validator.validate_tensors(
            original_residuals=residuals,
            embedded_residuals=embed_result.embedded_residuals,
        )

        # --------------------------------------------------------
        # 7. Print results
        # --------------------------------------------------------
        print("\n" + result.report())

        print(f"\n  Payload bits    : {PAYLOAD_BITS:,}")
        print(f"  Carrier tensors : {len(embed_result.embedded_residuals)}")

    except Exception as exc:

        print(
            f"\nERROR: {type(exc).__name__}: {exc}"
        )

    finally:

        # --------------------------------------------------------
        # 8. Release model memory
        # --------------------------------------------------------
        del nf4_model
        del fp16_model

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        print("\nMemory cleared.")