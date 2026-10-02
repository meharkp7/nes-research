"""
Residual source for the experiment suite.

Residual extraction is the single most expensive shared prerequisite:
every experiment from Exp2 upward needs ``{layer_id: residual_tensor}``,
and recomputing NF4 dequantization per experiment would multiply a
multi-minute cost by the number of experiments.

This module resolves residuals from the validated on-disk cache when it
is complete, and falls back to live extraction otherwise. It is additive:
it does not modify ``model_loader`` or ``loader``, and it preserves the
validated residual definition ``R = W_FP16 - dequantize(W_NF4)``.
"""

from typing import Dict, Optional, Tuple

import torch

from src.experiments.paths import REPO_ROOT
from src.model.cache_manager import ModelTensorCache
from src.model.model_loader import extract_residuals, load_model_pair

# Absolute, resolved from the repo root rather than the current working
# directory. A relative "cache/models" silently misses whenever the suite
# is launched from nes-llm/ instead of nes-research/, which costs a full
# NF4 dequantization pass over every layer for no reason — and the cache
# on disk is over 100 GB.
CACHE_ROOT = REPO_ROOT / "cache" / "models"


def _cache_for(model_id: str) -> ModelTensorCache:
    return ModelTensorCache(
        model_id=model_id,
        cache_root=str(CACHE_ROOT),
        quantization_type="nf4",
        use_double_quant=True,
        compute_dtype="float16",
    )


def cache_status(model_id: str, num_layers: int) -> dict:
    """Report whether the residual cache can serve this model."""
    try:
        cache = _cache_for(model_id)
        complete = cache.is_complete(num_layers)
        return {
            "cache_root": CACHE_ROOT,
            "cache_dir": str(cache.model_cache_dir),
            "complete": complete,
            "cached_layers": cache.cached_layers(),
            "expected_layers": num_layers,
        }
    except Exception as exc:
        return {
            "cache_root": CACHE_ROOT,
            "complete": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def load_cached_residuals(
    model_id: str,
    num_layers: int,
) -> Dict[int, torch.Tensor]:
    """Load only the residual tensor for each cached layer.

    ``ModelTensorCache.load_layer`` returns residual + FP16 + NF4 weights,
    which is three times the memory actually needed here. For a 36-layer
    3B model that is the difference between ~3 GB and ~10 GB resident, so
    this reads the layer file directly and drops the unused tensors.

    Tensors stay on CPU: residual preprocessing deliberately does not move
    large dequantized tensors to MPS (§26).
    """
    cache = _cache_for(model_id)

    if not cache.is_complete(num_layers):
        raise RuntimeError(
            f"Residual cache incomplete for {model_id}: "
            f"{cache.model_cache_dir}"
        )

    residuals: Dict[int, torch.Tensor] = {}

    for layer_id in range(num_layers):
        data = torch.load(
            cache._layer_path(layer_id),
            map_location="cpu",
            weights_only=True,
        )
        residuals[layer_id] = data["residual"]
        # Drop the FP16 and NF4 copies immediately; they are only needed
        # by the Exp5 model builder, which builds from live models.
        del data

    return residuals


def get_residuals(
    model_id: str,
    family: str,
    num_layers: int,
    use_cache: bool = True,
    models: Optional[Tuple] = None,
) -> Tuple[Dict[int, torch.Tensor], dict]:
    """Return ``(residuals, provenance)`` for one model.

    ``provenance`` records whether residuals came from cache or live
    extraction, so every artifact states where its inputs came from.

    When ``models`` is supplied, live extraction reuses that already-loaded
    ``(nf4, fp16, tokenizer)`` pair so the caller does not pay a second
    model load.
    """
    provenance = {
        "model_id": model_id,
        "family": family,
        "expected_layers": num_layers,
        "residual_definition": "R = W_FP16 - dequantize(W_NF4)",
    }

    if use_cache:
        status = cache_status(model_id, num_layers)
        provenance["cache"] = status

        if status.get("complete"):
            print(f"  [residuals] cache hit: {model_id} ({num_layers} layers)")
            residuals = load_cached_residuals(model_id, num_layers)
            provenance["source"] = "cache"
            provenance["layers"] = len(residuals)
            return residuals, provenance

        print(f"  [residuals] cache incomplete, extracting live: {model_id}")

    if models is None:
        models = load_model_pair(model_id)

    nf4_model, fp16_model, _tokenizer = models

    actual_layers = len(nf4_model.model.layers)
    provenance["actual_layers"] = actual_layers
    provenance["expected_layers"] = num_layers
    provenance["layer_count_matches_expected"] = (
        actual_layers == num_layers
    )

    residuals = extract_residuals(
        nf4_model=nf4_model,
        fp16_model=fp16_model,
        family=family,
    )

    provenance["source"] = "live_extraction"
    provenance["layers"] = len(residuals)

    return residuals, provenance