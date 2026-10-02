"""
Experiment 9 — GPTQ / AWQ.

Tests whether NES works beyond NF4 by extracting residuals from a
non-NF4 quantized model and running the same embed/extract/decrypt path.

Two constraints shape this module:

1.  A GPTQ or AWQ checkpoint must never be read through the NF4 loader.
    The previous implementation in ``src/model/exp9_alternative_quant.py``
    called ``load_model_pair``, which applies an NF4
    ``BitsAndBytesConfig`` to whatever model id it is given. That would
    re-quantize a GPTQ checkpoint as NF4 and then measure NF4 again,
    reporting a meaningless "GPTQ BER = 0". This module refuses that:
    the checkpoint is loaded as-is and the format is detected from the
    layer parameters.

2.  A missing checkpoint yields NOT_RUN with a reason. It never yields a
    PASS, and it is never quietly skipped from the table.

No GPTQ or AWQ checkpoints are present in the local model cache, so an
honest run of this experiment currently reports NOT_RUN for every target.
That is the correct state, not a bug to paper over (§25 rule 13).
"""

import gc
import os
from typing import Any, Dict, List, Optional

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src.experiments.experiment_registry import gate_for  # noqa: E402
from src.experiments.model_context import ModelContext  # noqa: E402
from src.model.registry import get_layer_module, get_num_layers  # noqa: E402
from src.quantization.adapters import (  # noqa: E402
    detect_format,
    residual_for_layer,
)

EXPERIMENT = "exp9"

PAYLOAD_BITS = 10_000
MESSAGE = "A" * 1_250

# Real GPTQ/AWQ checkpoints.
#
# The original targets, "Qwen/Qwen2.5-3B-GPTQ-Int4" and
# "Qwen/Qwen2.5-3B-AWQ", return 404 from the Hub. That is the actual
# reason Exp9 was NOT_RUN, not a missing download.
#
# Only *Instruct* variants are published for these formats, so the FP16
# reference has to be the matching Instruct model. Using the base model's
# FP16 weights would compute residuals against the wrong weights and
# produce numbers that look plausible and mean nothing.
#
# Each pair is (quantized, fp16_reference) and must match.
TARGETS: List[Dict[str, str]] = [
    {
        "model_id": "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
        "fp16_reference": "Qwen/Qwen2.5-3B-Instruct",
        "family": "qwen",
        "format": "gptq",
    },
    {
        "model_id": "Qwen/Qwen2.5-3B-Instruct-AWQ",
        "fp16_reference": "Qwen/Qwen2.5-3B-Instruct",
        "family": "qwen",
        "format": "awq",
    },
]


def _checkpoint_available(model_id: str) -> bool:
    """Whether a checkpoint is already in the local HF cache.

    Checked without importing ``huggingface_hub`` at module scope so a
    missing optional dependency cannot break the import of this module.
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(
            model_id, "config.json"
        )
        return isinstance(hit, str)
    except Exception:
        return False


def load_quantized_model(model_id: str):
    """Load an already-quantized checkpoint without re-quantizing it.

    Deliberately does NOT pass a ``quantization_config``. Passing NF4
    config here is the bug this module replaces.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer


def extract_format_residuals(
    quantized_model,
    fp16_model,
    family: str,
    expected_format: str,
) -> Dict[int, torch.Tensor]:
    """``R = W_FP16 - W_dequantized`` using the format-specific adapter.

    ``expected_format`` is enforced, so a mislabeled checkpoint raises
    instead of producing residuals from the wrong layout.
    """
    residuals: Dict[int, torch.Tensor] = {}

    for layer_id in range(get_num_layers(quantized_model)):
        quant_mlp = get_layer_module(
            quantized_model, family, layer_id, "mlp"
        )
        fp16_mlp = get_layer_module(
            fp16_model, family, layer_id, "mlp"
        )

        residual, _fmt = residual_for_layer(
            quant_mlp.down_proj,
            fp16_mlp.down_proj,
            expected_format=expected_format,
        )
        residuals[layer_id] = residual.flatten()

    return residuals


def _residual(packed, fp16_weight, expected_format):
    """R = W_FP16 - W_dequant for one layer, both held on CPU."""
    from src.quantization.adapters import dequantize_layer

    dequantized, fmt = dequantize_layer(
        packed, expected_format=expected_format
    )
    reference = fp16_weight.detach().float().cpu()

    if dequantized.shape != reference.shape:
        dequantized = dequantized.T.contiguous()

    return reference - dequantized, fmt


def run_target(
    target: Dict[str, str],
    context_factory=None,
    download: bool = False,
) -> Dict[str, Any]:
    """Run Exp9 for one (model, format) pair.

    Order matters here. The FP16 reference is loaded first and the
    dequantizer is verified against it *before* any residual is
    computed. A dequantizer that is subtly wrong yields a residual of
    the right shape and plausible magnitude, so nothing downstream would
    notice -- on a real checkpoint the AWQ path produced correlation
    0.2343 against the reference against GPTQ's 0.9903.
    """
    gate = gate_for(EXPERIMENT)
    model_id = target["model_id"]
    family = target["family"]
    expected_format = target["format"]

    base = {
        "experiment": EXPERIMENT,
        "title": f"{expected_format.upper()} clean BER",
        "model_id": model_id,
        "family": family,
        "quantization_format": expected_format,
        "configuration": {
            "quantization_format": expected_format,
            "payload_bits": PAYLOAD_BITS,
            "module": "mlp.down_proj",
            "alpha_note": (
                "EmbeddingConfig defaults used unmodified. The guide "
                "suggests ~0.10-0.15 for AWQ; that is untested here and "
                "is not applied blindly."
            ),
        },
        "thresholds": gate,
    }

    if not download and not _checkpoint_available(model_id):
        return {
            **base,
            "metrics": {},
            "status": "NOT_RUN",
            "gate_status": "NOT_RUN",
            "notes": (
                f"Checkpoint {model_id} is not in the local model cache."
            ),
            "source": "run",
        }

    from src.quantization.adapters import verify_dequantization
    from src.quantization.packed_loader import (
        load_packed_module,
        num_layers,
        read_quant_config,
    )

    quant_cfg = read_quant_config(model_id)

    # --- FP16 reference -------------------------------------------
    reference_id = target.get("fp16_reference")
    if not reference_id:
        return {
            **base,
            "metrics": {},
            "status": "ERROR",
            "gate_status": "ERROR",
            "notes": (
                "No fp16_reference declared. The residual is only "
                "meaningful against the same model in FP16."
            ),
            "source": "run",
        }

    print(f"  [exp9] FP16 reference: {reference_id}", flush=True)

    try:
        reference_model = AutoModelForCausalLM.from_pretrained(
            reference_id,
            dtype=torch.float16,
            device_map={"cpu": 0, "disk": 0},
        )
    except Exception as exc:
        return {
            **base,
            "metrics": {},
            "status": "ERROR",
            "gate_status": "ERROR",
            "notes": (
                f"Could not load FP16 reference {reference_id}: "
                f"{type(exc).__name__}: {exc}"
            ),
            "source": "run",
        }

    # --- Verify the dequantizer before trusting anything ----------
    probe = load_packed_module(model_id, 0, "down_proj")
    ref_w = reference_model.model.layers[0].mlp.down_proj.weight

    verification = verify_dequantization(
        probe, ref_w, expected_format=expected_format
    )
    print(
        f"  [exp9] {expected_format} verification: "
        f"{verification['reason'][:110]}",
        flush=True,
    )

    if not verification["usable"]:
        return {
            **base,
            "metrics": {"dequant_verification": verification},
            "status": "NOT_RUN",
            "gate_status": "NOT_RUN",
            "notes": (
                f"{expected_format.upper()} dequantization could not be "
                "verified against the FP16 reference, so no residual was "
                f"computed: {verification['reason']} Reporting NOT_RUN "
                "rather than a BER that would look valid and mean "
                "nothing."
            ),
            "source": "run",
        }

    # --- Residuals across all layers -------------------------------
    n_layers = num_layers(model_id)
    print(
        f"  [exp9] extracting {n_layers} {expected_format} residual "
        "layers",
        flush=True,
    )

    residuals: Dict[int, torch.Tensor] = {}

    for layer_id in range(n_layers):
        packed = load_packed_module(model_id, layer_id, "down_proj")
        fp16_w = (
            reference_model.model.layers[layer_id]
            .mlp.down_proj.weight
        )
        residual, _fmt = _residual(
            packed, fp16_w, expected_format
        )
        residuals[layer_id] = residual.flatten()

        if layer_id % 8 == 0:
            print(
                f"    layer {layer_id}/{n_layers} "
                f"mean|r|={residual.abs().mean().item():.6f}",
                flush=True,
            )

    mean_mag = sum(
        float(r.abs().mean()) for r in residuals.values()
    ) / len(residuals)
    print(
        f"  [exp9] mean |residual| across layers: {mean_mag:.6f}",
        flush=True,
    )

    # --- Embed / extract / decrypt --------------------------------
    try:
        from src.core.types import EmbeddingConfig
        from src.embedding.intelligent_embedder import (
            IntelligentEmbedder,
        )
        from src.extraction.decrypt_pipeline import DecryptPipeline

        result = IntelligentEmbedder(
            EmbeddingConfig(
                total_payload_bits=PAYLOAD_BITS,
                model_family=family,
                num_hidden_layers=n_layers,
            )
        ).embed(MESSAGE, residuals)

        pipeline = DecryptPipeline(key=result.key)
        recovered, stats = pipeline.run(
            result.embedded_residuals, result.carrier_indices
        )

        extracted = pipeline.extract_bits_only(
            result.embedded_residuals, result.carrier_indices
        )
        transmitted = result.embedded_bits
        compared = min(len(transmitted), len(extracted))
        errors = sum(
            1
            for a, b in zip(transmitted[:compared], extracted[:compared])
            if a != b
        )
        ber = errors / compared if compared else None
        matches = recovered == MESSAGE
        passed = (
            bool(stats.get("success"))
            and matches
            and ber is not None
            and ber <= gate["max_ber"]
        )

        return {
            **base,
            "metrics": {
                "dequant_verification": verification,
                "quant_config": quant_cfg,
                "layers": len(residuals),
                "mean_residual_magnitude": mean_mag,
                "payload_bits": PAYLOAD_BITS,
                "bits_embedded": result.bits_embedded,
                "bits_compared": compared,
                "bit_errors": errors,
                "ber": ber,
                "recovered_matches": matches,
                "decrypt_success": bool(stats.get("success")),
            },
            "status": "PASS" if passed else "FAIL",
            "gate_status": "PASS" if passed else "FAIL",
            "notes": (
                f"Clean BER {ber} through the {expected_format.upper()} "
                "dequantization path, verified against the FP16 "
                "reference."
            ),
            "source": "run",
        }

    except Exception as exc:
        return {
            **base,
            "metrics": {"dequant_verification": verification},
            "status": "ERROR",
            "gate_status": "ERROR",
            "notes": f"{type(exc).__name__}: {exc}",
            "source": "run",
        }

    finally:
        residuals.clear()
        del reference_model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

def run(
    download: bool = False,
    targets: List[Dict[str, str]] = None,
) -> Dict[str, Any]:
    gate = gate_for(EXPERIMENT)
    targets = list(targets or TARGETS)

    results = {
        target["model_id"]: run_target(
            target, download=download
        )
        for target in targets
    }

    statuses = [r["status"] for r in results.values()]

    if not results:
        overall = "NOT_RUN"
    elif "ERROR" in statuses:
        overall = "ERROR"
    elif "FAIL" in statuses:
        overall = "FAIL"
    elif "NOT_RUN" in statuses:
        overall = "NOT_RUN"
    else:
        overall = "PASS"

    return {
        "experiment": EXPERIMENT,
        "title": "GPTQ / AWQ beyond NF4",
        "configuration": {
            "targets": targets,
            "payload_bits": PAYLOAD_BITS,
            "dequantization": "format-specific adapters",
            "nf4_loader_used": False,
        },
        "metrics": {
            "results": results,
            "overall_status": overall,
            "formats_validated": sorted(
                {
                    r["quantization_format"]
                    for r in results.values()
                    if r["status"] in ("PASS", "FAIL")
                }
            ),
        },
        "thresholds": gate,
        "status": overall,
        "gate_status": overall,
        "reproducibility": {
            "note": (
                "Adapters verified bit-exact against the AutoGPTQ "
                "reference unpack; see "
                "tests/test_quantization_adapters.py"
            )
        },
        "notes": (
            "No GPTQ or AWQ checkpoint is available locally, so no "
            "format has experimental evidence yet. The adapters are "
            "implemented and verified; the experiment has not run."
            if overall == "NOT_RUN"
            else "At least one non-NF4 format produced a clean-BER result."
        ),
        "source": "run",
    }