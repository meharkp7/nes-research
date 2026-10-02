"""
Format-specific quantization adapters.

Each adapter answers one question: given a quantized linear layer, what
is its dequantized FP weight? Everything downstream — residual
extraction, QACI, embedding — is written against ``R = W_FP16 - W_deq``
and does not care which format produced ``W_deq``.

Keeping this format-specific matters. GPTQ packs weights along a
different axis than NF4, stores group-wise integer zeros and uses a
desc_act permutation, and AWQ stores group scales with a zero point. An
adapter that silently treats any of these as NF4 will either crash or,
worse, produce plausible-looking residuals that are meaningless.

Everything runs on CPU. GPTQ/AWQ dequantization builds a full dense FP
tensor, which is exactly the allocation that raises
``RuntimeError: Invalid buffer size`` on MPS (§26).
"""

from typing import Dict, Optional, Tuple

import torch

# ---------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------


def detect_format(module) -> Optional[str]:
    """Identify the quantization format of a linear layer.

    Returns ``"gptq"``, ``"awq"``, ``"nf4"`` or ``None`` for a plain
    FP layer. Detection is by the parameter names the format defines,
    never by the model id.
    """
    weight = getattr(module, "weight", None)

    if weight is None:
        return None

    # GPTQ: packed int32 weights plus group scales and act order.
    if hasattr(weight, "qweight") and hasattr(weight, "qzeros"):
        return "gptq"

    # bitsandbytes NF4 Params4bit.
    if hasattr(weight, "quant_state") or getattr(
        weight, "quant_type", None
    ) == "nf4":
        return "nf4"

    # AWQ: packed weights with a separate scales tensor carrying a
    # zero point, and qzeros.
    if hasattr(module, "scales") and hasattr(module, "qzeros"):
        return "awq"

    return None


def detect_model_format(model) -> Dict[str, str]:
    """Scan a model's linear layers and report the dominant format."""
    counts: Dict[str, int] = {}

    for module in model.modules():
        fmt = detect_format(module)
        if fmt is not None:
            counts[fmt] = counts.get(fmt, 0) + 1

    if not counts:
        return {"dominant": "none", "counts": {}, "mixed": False}

    dominant = max(counts, key=counts.get)
    total = sum(counts.values())

    return {
        "dominant": dominant,
        "counts": counts,
        "mixed": len(counts) > 1,
        "dominant_fraction": counts[dominant] / max(total, 1),
    }


# ---------------------------------------------------------------------
# GPTQ
# ---------------------------------------------------------------------


def dequantize_gptq_layer(module) -> torch.Tensor:
    """Dequantize a GPTQ-packed linear layer to FP32 on CPU.

    GPTQ stores, for an ``[in_features, out_features]`` weight:

        qweight   int32, ``[in_features // 32, out_features]``
                  holding ``32`` four-bit values per int32 word.
        qzeros    int32, same packing, group-wise zero points.
        scales    FP16, ``[num_groups, out_features]``
        g_idx     int32, ``[in_features]`` mapping each input channel
                  to its group. Equal to ``arange // group_size`` when
                  the checkpoint was quantized with ``desc_act=False``.

    The reference implementation is the AutoGPTQ pack/unpack loop; it is
    reimplemented here rather than imported because AutoGPTQ is not a
    dependency of this project and its helper is private.
    """
    weight = module.weight

    qweight = weight.qweight.detach().to("cpu")
    qzeros = weight.qzeros.detach().to("cpu")
    scales = weight.scales.detach().to("cpu").float()
    g_idx = weight.g_idx.detach().to("cpu")

    bits = 4
    pack_factor = 32 // bits  # 8 four-bit values per int32

    in_features = qweight.shape[0] * pack_factor
    out_features = qweight.shape[1]

    # --- Unpack the 4-bit weights and zero points -----------------
    #
    # Weights unpack along axis 1, zero points along the last axis;
    # see _unpack_4bit for why the two differ.
    weights = _unpack_4bit(
        qweight, in_features, out_features, axis=1
    )

    # The -1 bias on zero points is applied at pack time, so
    # dequantization adds 1 back (matching AutoGPTQ's `zeros + 1`).
    zeros = (
        _unpack_4bit(
            qzeros, scales.shape[0], out_features, axis=-1
        )
        + 1
    )

    # --- Scatter group scales to per-input-channel ----------------
    #
    # The reference reconstruction is exactly
    #     W[i, j] = scales[g_idx[i], j] * (w[i, j] - zeros[g_idx[i], j])
    #
    # so the group index comes from g_idx directly. With desc_act=False
    # that equals i // group_size, but for activation-order checkpoints
    # g_idx is a real permutation and must not be assumed positional.
    group_index = g_idx.long()

    if int(group_index.max().item()) >= scales.shape[0]:
        raise RuntimeError(
            f"g_idx references group {int(group_index.max().item())} "
            f"but only {scales.shape[0]} scale groups exist"
        )

    scale_by_channel = scales[group_index]
    zero_by_channel = zeros[group_index]

    return (weights - zero_by_channel) * scale_by_channel


def _unpack_4bit(
    packed: torch.Tensor,
    rows: int,
    columns: int,
    axis: int = 1,
) -> torch.Tensor:
    """Unpack int32-packed unsigned 4-bit values into ``[rows, columns]``.

    Verified against AutoGPTQ ``QuantLinear.forward``. The two packed
    tensors use different nibble axis orders:

        qweight  right_shift(unsqueeze(qweight, 1).expand(-1, 8, -1),
                             [0,4,...,28].unsqueeze(-1))
                 -> shape ``[words, 8, out]`` -> ``[in, out]``

        qzeros   right_shift(unsqueeze(qzeros, 2).expand(-1, -1, 8),
                             [0,4,...,28].unsqueeze(0))
                 -> shape ``[groups, out_words, 8]`` -> ``[groups, out]``

    So ``axis=1`` expands the nibble dimension after the packed rows
    (weights), and ``axis=-1`` expands it last (zeros). Using the wrong
    one silently transposes the weight matrix rather than erroring, which
    would yield plausible but meaningless residuals.

    Element ``i`` lives at nibble ``i % 8`` of int32 word ``i // 8``, with
    the first element in the least significant nibble.
    """
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)

    if axis == 1:
        expanded = packed.unsqueeze(1).expand(-1, 8, -1)
        unpacked = (
            expanded >> shifts.to(packed.dtype).reshape(1, 8, 1)
        ) & 0x0F
    elif axis == -1:
        expanded = packed.unsqueeze(2).expand(-1, -1, 8)
        unpacked = (
            expanded >> shifts.to(packed.dtype).reshape(1, 1, 8)
        ) & 0x0F
    else:
        raise ValueError(f"Unsupported nibble axis: {axis}")

    return unpacked.reshape(rows, columns).float()


# ---------------------------------------------------------------------
# AWQ
# ---------------------------------------------------------------------


def dequantize_awq_layer(module) -> torch.Tensor:
    """Dequantize an AWQ-packed linear layer to FP32 on CPU.

    AWQ stores group-wise FP scales and integer zero points alongside
    packed 4-bit weights, using the same int32 packing layout as GPTQ:

        qweight  ``[in_features // 8, out_features]``
        qzeros   ``[num_groups, out_features // 8]``
        scales   ``[num_groups, out_features]``

    Reconstructed as:

        W_deq[i, j] = (q[i, j] - z[i // group_size, j]) * s[i // group_size, j]

    Two deliberate differences from the GPTQ path above:

    -   AWQ has no activation-order permutation, so the group index is
        positional (``repeat_interleave``) rather than read from g_idx.
    -   AWQ does **not** bias zero points by -1 at pack time, so no
        ``+ 1`` is applied here. Applying GPTQ's correction would shift
        every dequantized weight by one scale unit.
    """
    weight = module.weight

    qweight = weight.qweight.detach().to("cpu")
    qzeros = module.qzeros.detach().to("cpu")
    scales = module.scales.detach().to("cpu").float()

    pack_factor = 32 // 4

    in_features = qweight.shape[0] * pack_factor
    out_features = qweight.shape[1]

    values = _unpack_4bit(qweight, in_features, out_features, axis=1)
    zeros = _unpack_4bit(qzeros, scales.shape[0], out_features, axis=-1)

    # scales.shape[0] is the NUMBER OF GROUPS, not the group width.
    # The group width is how many input channels each group covers.
    num_groups = scales.shape[0]

    if num_groups == 0 or in_features % num_groups != 0:
        raise RuntimeError(
            f"AWQ scale groups ({num_groups}) do not evenly divide "
            f"in_features ({in_features})"
        )

    group_width = in_features // num_groups

    scale_expanded = scales.repeat_interleave(group_width, dim=0)
    zero_expanded = zeros.repeat_interleave(group_width, dim=0)

    return (values - zero_expanded) * scale_expanded


# ---------------------------------------------------------------------
# NF4 (existing validated path, for dispatch completeness)
# ---------------------------------------------------------------------


def dequantize_nf4_layer(module) -> torch.Tensor:
    """Dequantize a bitsandbytes NF4 layer, keeping the CPU-safe path.

    Uses ``dequantize_4bit`` on CPU rather than ``Params4bit.dequantize``
    so the MPS failure mode from §26 cannot be reintroduced.
    """
    import bitsandbytes.functional as bnb_func

    weight = module.weight

    if hasattr(weight, "quant_state"):
        qstate = weight.quant_state
        if hasattr(qstate, "absmax") and qstate.absmax is not None:
            if qstate.absmax.device != torch.device("cpu"):
                import copy

                qstate = copy.copy(qstate)
                for attr in (
                    "absmax", "code", "offset",
                    "nested_absmax", "nested_quant_map",
                ):
                    value = getattr(qstate, attr, None)
                    if torch.is_tensor(value):
                        setattr(qstate, attr, value.detach().to("cpu"))

        return bnb_func.dequantize_4bit(
            getattr(weight, "data", weight).detach().to("cpu"),
            qstate,
        ).float()

    if hasattr(weight, "dequantize"):
        return weight.dequantize().float()

    return weight.float()


# ---------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------

DEQUANTIZERS = {
    "gptq": dequantize_gptq_layer,
    "awq": dequantize_awq_layer,
    "nf4": dequantize_nf4_layer,
}


def dequantize_layer(
    module,
    expected_format: Optional[str] = None,
) -> Tuple[torch.Tensor, str]:
    """Dequantize one linear layer, dispatching on its actual format.

    ``expected_format`` turns a silent mismatch into an error. That is the
    guard against the Exp9 failure mode where a GPTQ checkpoint is read
    through the NF4 path and yields meaningless residuals.

    Returns ``(weight, format_used)``.
    """
    fmt = detect_format(module)

    if fmt is None:
        return module.weight.detach().float().cpu(), "fp16"

    if expected_format and fmt != expected_format:
        raise RuntimeError(
            f"Expected a {expected_format} layer but detected {fmt}. "
            "Refusing to dequantize with the wrong format's layout; "
            "this would silently produce incorrect residuals."
        )

    dequantizer = DEQUANTIZERS.get(fmt)

    if dequantizer is None:
        raise RuntimeError(
            f"No dequantizer implemented for format {fmt!r}."
        )

    return dequantizer(module), fmt


def residual_for_layer(
    quantized_module,
    fp16_module,
    expected_format: Optional[str] = None,
) -> Tuple[torch.Tensor, str]:
    """Compute ``R = W_FP16 - W_dequantized`` for one linear layer.

    Returns ``(residual, format_used)``. Both operands are brought to CPU
    before the subtraction so no large dense tensor is allocated on MPS.
    """
    dequantized, fmt = dequantize_layer(
        quantized_module,
        expected_format=expected_format,
    )

    reference = fp16_module.weight.detach().float().cpu()

    if dequantized.shape != reference.shape:
        if dequantized.numel() == reference.numel():
            dequantized = dequantized.reshape(reference.shape)
        else:
            raise RuntimeError(
                f"Shape mismatch after {fmt} dequantization: "
                f"dequantized={tuple(dequantized.shape)} "
                f"reference={tuple(reference.shape)}"
            )

    return reference - dequantized, fmt