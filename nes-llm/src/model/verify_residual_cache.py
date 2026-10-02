"""
Verify a cached residual against live NF4 extraction.

Why this exists
---------------
Qwen2.5-7B's profiled residual magnitude is 0.009412, roughly 7.9x every
other model (0.0002-0.0019), and its spread *within* the model runs
0.0012-0.0130 while every other model's layers are tightly clustered.
That pattern does not look like a model property. Either the cache is
stale or was written from a different configuration, in which case every
Qwen2.5-7B number in the manifest is wrong.

I could not settle it earlier: two attempts to load NF4 + FP16 alongside a
7.6GB residual dictionary hit the MPS ceiling. Only the layers being
checked are held, so this compares a few layers at a time and releases
each before the next.

Usage
-----
    python -m src.model.verify_residual_cache Qwen/Qwen2.5-7B qwen 28 --layers 0 14 27
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from src.model.cache_manager import ModelTensorCache  # noqa: E402
from src.model.registry import get_layer_module  # noqa: E402
from src.experiments.paths import REPO_ROOT, RESULTS_DIR  # noqa: E402


def cache_for(model_id: str) -> ModelTensorCache:
    return ModelTensorCache(
        model_id=model_id,
        cache_root=str(REPO_ROOT / "cache" / "models"),
        quantization_type="nf4",
        use_double_quant=True,
        compute_dtype="float16",
    )


def verify(
    model_id: str,
    family: str,
    layers: List[int],
) -> Dict[str, Any]:
    """Compare cached residuals with freshly extracted ones, layer by layer."""
    import bitsandbytes.functional as bnb_func

    from src.model.model_loader import load_model_pair

    cache = cache_for(model_id)

    print(f"Loading {model_id}...", flush=True)
    nf4, fp16, _tokenizer = load_model_pair(model_id)
    print(
        f"  layers in model: {len(nf4.model.layers)}",
        flush=True,
    )

    rows: List[Dict[str, Any]] = []

    for layer_id in layers:
        quant_w = get_layer_module(
            nf4, family, layer_id, "mlp"
        ).down_proj.weight
        fp16_w = get_layer_module(
            fp16, family, layer_id, "mlp"
        ).down_proj.weight.to(quant_w.device)

        dequant = bnb_func.dequantize_4bit(
            getattr(quant_w, "data", quant_w),
            quant_w.quant_state,
        ).float().reshape(fp16_w.shape)

        live = (fp16_w.float() - dequant).flatten().cpu()

        cached = torch.load(
            cache._layer_path(layer_id),
            map_location="cpu",
            weights_only=True,
        )["residual"].flatten()

        exact = torch.equal(live, cached)
        max_abs = float((live - cached).abs().max().item())
        live_mag = float(live.abs().mean().item())
        cached_mag = float(cached.abs().mean().item())

        rows.append(
            {
                "layer_id": layer_id,
                "bit_exact": exact,
                "max_abs_diff": max_abs,
                "live_mag_mean": live_mag,
                "cached_mag_mean": cached_mag,
                "mag_ratio": (
                    cached_mag / live_mag if live_mag else None
                ),
                "shape_match": live.shape == cached.shape,
            }
        )

        print(
            f"  layer {layer_id:>3}: "
            f"exact={exact}  "
            f"live_mag={live_mag:.6f}  "
            f"cached_mag={cached_mag:.6f}  "
            f"max_diff={max_abs:.3e}",
            flush=True,
        )

        # Release before the next layer.
        del live, cached, dequant, quant_w, fp16_w
        gc.collect()

    del nf4, fp16, _tokenizer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    all_exact = all(r["bit_exact"] for r in rows)
    ratios = [
        r["mag_ratio"] for r in rows if r["mag_ratio"] is not None
    ]

    return {
        "model_id": model_id,
        "family": family,
        "layers_checked": layers,
        "rows": rows,
        "all_bit_exact": all_exact,
        "mag_ratio_min": min(ratios) if ratios else None,
        "mag_ratio_max": max(ratios) if ratios else None,
        "cache_is_trustworthy": all_exact,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a cached residual against live extraction"
    )
    parser.add_argument("model_id")
    parser.add_argument("family")
    parser.add_argument("num_layers", type=int)
    parser.add_argument(
        "--layers", nargs="*", type=int, default=None,
        help="Layers to check. Default: first, middle, last.",
    )
    args = parser.parse_args(argv)

    layers = args.layers or [
        0,
        args.num_layers // 2,
        args.num_layers - 1,
    ]

    result = verify(args.model_id, args.family, layers)

    print()
    print("=" * 60)
    print(f"bit-exact on all layers : {result['all_bit_exact']}")
    print(
        f"mag ratio (cached/live) : "
        f"{result['mag_ratio_min']:.6f} - {result['mag_ratio_max']:.6f}"
    )
    print(
        "VERDICT                 : "
        + ("cache matches live extraction"
           if result["all_bit_exact"]
           else "CACHE DOES NOT MATCH LIVE EXTRACTION")
    )
    print("=" * 60)

    slug = args.model_id.replace("/", "__").lower()
    out = RESULTS_DIR / f"verify_residual_cache_{slug}.json"
    out.write_text(
        json.dumps(
            {
                "experiment": "verify_residual_cache",
                "purpose": (
                    "Qwen2.5-7B residual magnitude is 7.9x every other "
                    "model with a 10x within-model spread, which does "
                    "not look like a model property."
                ),
                **result,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")

    return 0 if result["all_bit_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())