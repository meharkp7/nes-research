"""
Precompute and cache model weights + NF4 residuals.

The expensive operations:

    1. Load FP16 model
    2. Load NF4 model
    3. Dequantize NF4 weights
    4. Compute FP16 - NF4 residuals

are performed once and cached layer-by-layer.

Later experiments can load the cached tensors directly.

This is the only cache-BUILD path in the repo. It runs through
``src.model.model_loader.extract_residuals`` — the same function the
experiment pipeline reads residuals with — writing each layer through
``cache_manager.ModelTensorCache``. The build and the read therefore
cannot drift: there is one residual definition.

Usage:
    python3 -m scripts.cache_model
    python3 -m scripts.cache_model --model Qwen/Qwen2.5-3B --force
    python3 -m scripts.cache_model --verify-against /path/to/cache/models
"""

import argparse
import gc
import os
from pathlib import Path

import torch

from src.experiments.paths import REPO_ROOT
from src.model.cache_manager import ModelTensorCache
from src.model.model_loader import (
    extract_residuals,
    load_model_pair,
)


# ==============================================================
# CONFIGURATION
# ==============================================================

MODEL_ID = "Qwen/Qwen2.5-3B"
FAMILY = "qwen"

CACHE_ROOT = REPO_ROOT / "cache" / "models"


# ==============================================================
# DEVICE
# ==============================================================

# The models run on MPS.
#
# BitsAndBytes dequantization runs wherever the NF4 weights live;
# save_layer() moves each finished tensor to CPU before writing,
# so the cache itself is always CPU-resident.

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# ==============================================================
# VERIFY: rebuild-compare one cache against an existing one
# ==============================================================

TENSOR_NAMES = ("residual", "fp16_weight", "nf4_dequantized")


def verify_against(
    cache: ModelTensorCache,
    other_root: Path,
    num_layers: int,
) -> int:
    """Compare this cache layer-by-layer against one in ``other_root``.

    Returns the number of mismatched (layer, tensor) pairs. The point
    of the check: prove the ported build reproduces a cache built by
    the retired path, tensor for tensor, at delta 0.0.
    """
    other = ModelTensorCache(
        model_id=cache.model_id,
        cache_root=str(other_root),
        quantization_type=cache.quantization_type,
        use_double_quant=cache.use_double_quant,
        compute_dtype=cache.compute_dtype,
    )

    if not other.is_complete(num_layers):
        print(
            f"[verify] reference cache incomplete: "
            f"{len(other.summary()['cached_layers'])}/{num_layers}"
        )
        return 1

    bad = 0
    for layer_id in range(num_layers):
        built = cache.load_layer(layer_id)
        ref = other.load_layer(layer_id)
        for name, ta, tb in zip(TENSOR_NAMES, built, ref):
            if ta.shape != tb.shape:
                print(
                    f"[verify] layer {layer_id} {name}: shape "
                    f"{tuple(ta.shape)} != {tuple(tb.shape)}"
                )
                bad += 1
            elif not torch.equal(ta, tb):
                delta = float((ta - tb).abs().max())
                print(
                    f"[verify] layer {layer_id} {name}: delta={delta:g}"
                )
                bad += 1

    n = num_layers * len(TENSOR_NAMES)
    if bad == 0:
        print(
            f"[verify] OK — all {n} tensors across "
            f"{num_layers} layers identical at delta 0.0"
        )
    else:
        print(f"[verify] FAILED — {bad}/{n} tensors differ")
    return bad


# ==============================================================
# MAIN
# ==============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build the residual cache."
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--family", default=FAMILY)
    parser.add_argument("--cache-root", default=str(CACHE_ROOT))
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute layers already in the cache",
    )
    parser.add_argument(
        "--verify-against",
        default=None,
        help=(
            "after building, compare every tensor against the same "
            "model's cache under this cache root"
        ),
    )
    args = parser.parse_args()

    cache = ModelTensorCache(
        model_id=args.model,
        cache_root=args.cache_root,
    )

    print(f"[cache_model] model:      {args.model}")
    print(f"[cache_model] family:     {args.family}")
    print(f"[cache_model] cache root: {args.cache_root}")
    print(f"[cache_model] device:     {DEVICE}")

    nf4_model, fp16_model, _tokenizer = load_model_pair(
        args.model,
        device=DEVICE,
    )

    residuals = extract_residuals(
        nf4_model,
        fp16_model,
        args.family,
        cache=cache,
        force_recompute=args.force,
    )

    expected = len(residuals)
    del residuals
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    got = len(cache.cached_layers())
    print(
        f"[cache_model] {got}/{expected} layers "
        f"{'OK' if got == expected else 'MISMATCH'}"
    )

    if args.verify_against:
        bad = verify_against(cache, Path(args.verify_against), expected)
        return 1 if bad else 0

    return 0 if got == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
