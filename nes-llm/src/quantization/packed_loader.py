"""
Load GPTQ/AWQ checkpoints without a quantizer runtime.

``transformers`` cannot materialize either format here: GPTQ requires
``optimum`` and AWQ requires ``gptqmodel``, neither of which is
installed. Installing them would add heavy dependencies for a path whose
only job is to read four packed tensors.

The adapters in this package already work at the tensor level, so the
packed values are read straight out of the safetensors shards and wrapped
in the small interface the adapters expect. This also keeps the
dequantization logic under test in one place instead of delegating the
interesting part to a third-party runtime.

Real layout, for reference (Qwen2.5-3B-Instruct, group_size=128):

    GPTQ  qweight (out//8, in)   qzeros (groups, in//8)   scales (groups, in)
          g_idx   (out,)
    AWQ   qweight (out, in//8)   qzeros (groups, in//8)   scales (groups, in)
"""

import glob
import json
import os
from typing import Dict, Optional

import torch

HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")


def _snapshot_dir(model_id: str) -> str:
    base = os.path.join(
        HF_CACHE, "models--" + model_id.replace("/", "--"), "snapshots"
    )
    snapshots = sorted(glob.glob(os.path.join(base, "*")))
    if not snapshots:
        raise FileNotFoundError(
            f"No local snapshot for {model_id} under {base}"
        )
    return snapshots[-1]


def _weight_keys(model_id: str, module: str = "down_proj") -> Dict:
    """Collect every packed tensor for one module across all shards."""
    from safetensors import safe_open

    snapshot = _snapshot_dir(model_id)
    shards = sorted(glob.glob(os.path.join(snapshot, "*.safetensors")))

    if not shards:
        raise FileNotFoundError(
            f"No safetensors shards in {snapshot}"
        )

    wanted = (
        "qweight", "qzeros", "scales", "g_idx", "bias",
    )
    found: Dict[str, torch.Tensor] = {}

    for shard in shards:
        with safe_open(shard, framework="pt") as handle:
            for key in handle.keys():
                tail = key.split(".")[-1]
                if tail not in wanted:
                    continue
                # Match the same layer and module, so self_attn and mlp
                # weights are not mixed.
                if f"layers.{_layer_index(key)}.{module}." not in key:
                    continue
                found[tail] = handle.get_tensor(key)

    return found


def _layer_index(key: str) -> int:
    parts = key.split(".")
    if "layers" in parts:
        return parts[parts.index("layers") + 1]
    return "-1"


class PackedWeight:
    """Minimal stand-in for a quantized nn.Linear weight.

    Mirrors where each format actually keeps its tensors: GPTQ puts
    qzeros/scales/g_idx on the weight, AWQ puts qzeros/scales on the
    module. Exposing both on both objects would make the two formats
    indistinguishable to ``detect_format``.
    """

    def __init__(self, tensors, quant_method: str):
        if "qweight" not in tensors:
            raise KeyError("packed weight missing qweight")

        self.qweight = tensors["qweight"]

        if quant_method == "gptq":
            self.qzeros = tensors.get("qzeros")
            self.scales = tensors.get("scales")
            self.g_idx = tensors.get("g_idx")

        self.bias = tensors.get("bias")


class PackedModule:
    """Minimal stand-in for a quantized nn.Linear module."""

    def __init__(self, tensors: Dict[str, torch.Tensor], quant_method: str):
        self.weight = PackedWeight(tensors, quant_method)

        if quant_method == "awq":
            self.qzeros = tensors.get("qzeros")
            self.scales = tensors.get("scales")


def load_packed_module(
    model_id: str,
    layer_id: int,
    module: str = "down_proj",
) -> PackedModule:
    """Read one quantized layer's packed tensors from disk."""
    snapshot = _snapshot_dir(model_id)

    prefix = f"model.layers.{layer_id}.mlp.{module}."

    from safetensors import safe_open

    tensors: Dict[str, torch.Tensor] = {}

    for shard in sorted(
        glob.glob(os.path.join(snapshot, "*.safetensors"))
    ):
        with safe_open(shard, framework="pt") as handle:
            for key in handle.keys():
                if not key.startswith(prefix):
                    continue
                tensors[key[len(prefix):]] = handle.get_tensor(key)
        if "qweight" in tensors:
            break

    if not tensors:
        raise KeyError(
            f"No packed tensors for layer {layer_id} {module} in "
            f"{model_id}"
        )

    quant_method = read_quant_config(model_id).get(
        "quant_method", "gptq"
    )

    return PackedModule(tensors, quant_method)


def read_quant_config(model_id: str) -> Dict:
    """Quantization parameters as declared by the checkpoint."""
    snapshot = _snapshot_dir(model_id)
    path = os.path.join(snapshot, "config.json")

    if not os.path.exists(path):
        return {}

    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)

    return config.get("quantization_config", {}) or {}


def num_layers(model_id: str) -> int:
    snapshot = _snapshot_dir(model_id)
    path = os.path.join(snapshot, "config.json")

    with open(path, encoding="utf-8") as handle:
        return int(json.load(handle)["num_hidden_layers"])