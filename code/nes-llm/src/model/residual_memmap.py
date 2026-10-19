"""
Memory-mapped residuals.

The laptop has 26 GB of RAM and an MPS ceiling around 30 GB. A model's
residuals are the largest single allocation the suite makes -- 3.2 GB for
Qwen2.5-3B, 8.6 GB for Gemma-2-9B in float32 -- and they are only read,
never written, outside the carrier positions that get embedded.

float16 is not an option. About 2.4% of residual values per layer are
subnormal in float16 and flush to zero. That leaves `mag_mean` unchanged
so exp2's metric is unaffected, but it would zero 2.4% of the weight
matrix in any reconstruction, which is catastrophic for perplexity.

So this keeps float32 exactly and instead stops holding all layers
resident at once. Residual layers are materialised as raw .npy files once
per model, then opened with ``mmap_mode='r'``. The OS pages them in on
demand and evicts them under pressure, so resident memory tracks the
layers actually being touched rather than the model's total size.

Disk cost is one extra copy of the residuals (~35 GB across the six
cached models). Process-local caching is deliberately avoided: a dict of
every layer would reintroduce exactly the pressure this avoids.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from src.model.cache_manager import ModelTensorCache
from src.experiments.paths import REPO_ROOT

MMAP_ROOT = REPO_ROOT / "cache" / "mmap"


def _cache_for(model_id: str) -> ModelTensorCache:
    return ModelTensorCache(
        model_id=model_id,
        cache_root=str(REPO_ROOT / "cache" / "models"),
        quantization_type="nf4",
        use_double_quant=True,
        compute_dtype="float16",
    )


def mmap_dir(model_id: str) -> Path:
    return MMAP_ROOT / ModelTensorCache._sanitize_model_id(model_id)


def materialize(
    model_id: str,
    num_layers: int,
    verbose: bool = True,
) -> Path:
    """Write residual-only .npy files from the existing .pt cache.

    One-time per model. Reads each .pt, keeps only the residual, and
    writes it as raw float32 so it can be mapped later.
    """
    cache = _cache_for(model_id)
    target = mmap_dir(model_id)
    target.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"  [mmap] materializing {model_id} -> {target}")

    for layer_id in range(num_layers):
        out = target / f"layer_{layer_id:04d}.npy"

        if out.exists():
            continue

        data = torch.load(
            cache._layer_path(layer_id),
            map_location="cpu",
            weights_only=True,
        )
        residual = data["residual"].numpy().astype(np.float32, copy=False)
        np.save(out, residual)
        del data, residual

    return target


def is_materialized(model_id: str, num_layers: int) -> bool:
    target = mmap_dir(model_id)

    if not target.exists():
        return False

    for layer_id in range(num_layers):
        if not (target / f"layer_{layer_id:04d}.npy").exists():
            return False

    return True


def load(
    model_id: str,
    num_layers: int,
    as_tensor: bool = True,
) -> Dict[int, torch.Tensor]:
    """Open residual layers as memory-mapped arrays.

    The arrays are wrapped, not copied. Residuals were saved from a
    contiguous float32 tensor, so the mapped array is contiguous and
    ``torch.from_numpy`` references it in place.

    The saving is that the OS, not Python, owns the pages. Clean mapped
    file pages are purgeable, so under memory pressure macOS reclaims
    them and re-reads on next touch. Holding a tensor in a Python dict
    pins its pages for the life of the process and cannot be reclaimed.
    That is the whole difference: not laziness, but reclaimability.
    """
    if not is_materialized(model_id, num_layers):
        materialize(model_id, num_layers)

    target = mmap_dir(model_id)
    layers: Dict[int, torch.Tensor] = {}

    for layer_id in range(num_layers):
        mapped = np.load(
            target / f"layer_{layer_id:04d}.npy", mmap_mode="r"
        )

        if not as_tensor:
            layers[layer_id] = mapped
            continue

        # torch.from_numpy on a read-only buffer warns that the storage
        # is non-writable. Nothing here writes to it: embedding clones
        # the carrier layer before modification.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*non-writable.*"
            )
            layers[layer_id] = torch.from_numpy(mapped)

    return layers


def materialize_all(
    models=(),
    verbose: bool = True,
) -> Dict[str, bool]:
    """Materialize several models, reporting which succeeded."""
    results: Dict[str, bool] = {}

    for model_id, num_layers in models:
        try:
            materialize(model_id, num_layers, verbose=verbose)
            results[model_id] = is_materialized(
                model_id, num_layers
            )
        except Exception as exc:
            if verbose:
                print(
                    f"  [mmap] FAILED {model_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
            results[model_id] = False

    return results