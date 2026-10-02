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

    AWQ and GPTQ both carry ``qweight``/``qzeros``/``scales``, so the
    only structural difference is *where* the zero points and scales
    live: GPTQ puts them on the weight alongside ``g_idx``, AWQ puts them
    on the module. ``g_idx`` is therefore the discriminator, and a bare
    pack of qweight+scales without it is ambiguous.
    """
    weight = getattr(module, "weight", None)

    if weight is None:
        return None

    has_pack = hasattr(weight, "qweight")

    if not has_pack:
        # bitsandbytes NF4 Params4bit.
        if hasattr(weight, "quant_state") or getattr(
            weight, "quant_type", None
        ) == "nf4":
            return "nf4"
        return None

    # GPTQ keeps qzeros/scales/g_idx on the weight.
    if hasattr(weight, "g_idx"):
        return "gptq"

    # AWQ keeps qzeros/scales on the module.
    if hasattr(module, "qzeros") and hasattr(module, "scales"):
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

    Verified against AutoGPTQ ``QuantLinear.forward``.

    Real checkpoint layout (Qwen2.5-3B-Instruct-GPTQ-Int4, group_size=128):

        qweight   (out_features // 8, in_features)   e.g. (1376, 2048)
        qzeros    (num_groups, in_features // 8)     e.g. (86, 256)
        scales    (num_groups, in_features)          e.g. (86, 2048)
        g_idx     (out_features,)                    e.g. (11008,)

    The weight packs along the *output* axis, so unpacking expands the
    nibble dimension on axis 1 and the result is ``[out, in]`` -- the
    nn.Linear convention. The zero points pack along the *input* axis,
    so they expand on the last axis.

    ``g_idx`` has one entry per output row and selects the scale group
    for that row, which is why it is applied to the reconstructed
    matrix rather than to an input axis.
    """
    weight = module.weight

    qweight = weight.qweight.detach().to("cpu")
    qzeros = weight.qzeros.detach().to("cpu")
    scales = weight.scales.detach().to("cpu").float()
    g_idx = weight.g_idx.detach().to("cpu")

    pack_factor = 32 // 4

    out_features = qweight.shape[0] * pack_factor
    in_features = qweight.shape[1]
    num_groups = scales.shape[0]

    if scales.shape[1] != in_features:
        raise RuntimeError(
            f"GPTQ scales second dim {scales.shape[1]} does not match "
            f"qweight columns {in_features}"
        )

    weights = _unpack_4bit(
        qweight, out_features, in_features, axis=1
    )

    # The -1 bias on zero points is applied at pack time, so
    # dequantization adds 1 back (matching AutoGPTQ's `zeros + 1`).
    zeros = (
        _unpack_4bit(qzeros, num_groups, in_features, axis=-1) + 1
    )

    if g_idx.numel() != out_features:
        raise RuntimeError(
            f"GPTQ g_idx has {g_idx.numel()} entries but the weight has "
            f"{out_features} output rows"
        )

    group_index = g_idx.long()

    if int(group_index.max().item()) >= num_groups:
        raise RuntimeError(
            f"g_idx references group {int(group_index.max().item())} "
            f"but only {num_groups} scale groups exist"
        )

    scale_by_row = scales[group_index]
    zero_by_row = zeros[group_index]

    # GPTQ stores the weight as [in_features, out_features] and relies on
    # the runtime transposing at matmul time. nn.Linear.weight is
    # [out_features, in_features], so transpose to the convention every
    # caller here assumes.
    return ((weights - zero_by_row) * scale_by_row).T.contiguous()


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

    Real checkpoint layout (Qwen2.5-3B-Instruct-AWQ, group_size=128,
    zero_point=true):

        qweight   (out_features, in_features // 8)   e.g. (11008, 256)
        qzeros    (num_groups, in_features // 8)    e.g. (86, 256)
        scales    (num_groups, in_features)         e.g. (86, 2048)

    AWQ therefore packs the nibbles along the *input* axis -- the
    opposite orientation from GPTQ -- so unpacking expands the last axis
    and the result is ``[out, in]``.

    **This was wrong before.** The earlier version assumed
    ``[in // 8, out]`` with groups along the input axis, which happens to
    be indistinguishable on a synthetic tensor of the wrong shape but
    transposes the real matrix. A synthetic round-trip test could not
    catch it because the test was built from the same wrong assumption.

    Reconstruction, with groups running along the output axis:

        W[i, j] = (q[i, j] - z[g(i), j]) * s[g(i), j]
        g(i) = i // (out_features // num_groups)

    Two deliberate differences from the GPTQ path:

    -   No activation-order permutation, so groups are positional.
    -   AWQ does **not** bias zero points by -1 at pack time, so no
        ``+ 1`` is applied. Copying GPTQ's correction would shift every
        dequantized weight by one scale unit.
    """
    weight = module.weight

    qweight = weight.qweight.detach().to("cpu")
    qzeros = module.qzeros.detach().to("cpu")
    scales = module.scales.detach().to("cpu").float()

    pack_factor = 32 // 4

    out_features = qweight.shape[0]
    in_features = qweight.shape[1] * pack_factor
    num_groups = scales.shape[0]

    if scales.shape[1] != in_features:
        raise RuntimeError(
            f"AWQ scales second dim {scales.shape[1]} does not match "
            f"qweight columns * 8 = {in_features}"
        )

    values = _unpack_4bit(
        qweight, out_features, in_features, axis=-1
    )
    zeros = _unpack_4bit(qzeros, num_groups, in_features, axis=-1)

    if out_features % num_groups != 0:
        raise RuntimeError(
            f"AWQ scale groups ({num_groups}) do not evenly divide "
            f"out_features ({out_features})"
        )

    group_width = out_features // num_groups

    scale_expanded = scales.repeat_interleave(group_width, dim=0)
    zero_expanded = zeros.repeat_interleave(group_width, dim=0)

    # As with GPTQ, the packed layout yields [in_features, out_features];
    # nn.Linear.weight is [out_features, in_features].
    return (
        (values - zero_expanded) * scale_expanded
    ).T.contiguous()


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
        if dequantized.T.shape == reference.shape:
            # Packed formats store the weight in [in, out] order while
            # nn.Linear.weight is [out, in]. Transposing is required.
            #
            # A plain reshape would have the same element count and
            # therefore "succeed" while silently producing a scrambled
            # matrix -- the failure mode this whole module guards
            # against.
            dequantized = dequantized.T.contiguous()
        elif dequantized.numel() == reference.numel():
            raise RuntimeError(
                "Shape mismatch after dequantization that is not a "
                f"transpose: {tuple(dequantized.shape)} vs "
                f"{tuple(reference.shape)}. Refusing to reshape, which "
                "would scramble the weight matrix."
            )
        else:
            raise RuntimeError(
                f"Shape mismatch after {fmt} dequantization: "
                f"dequantized={tuple(dequantized.shape)} "
                f"reference={tuple(reference.shape)}"
            )

    return reference - dequantized, fmt

def verify_dequantization(
    quantized_module,
    reference_weight: torch.Tensor,
    expected_format: Optional[str] = None,
    min_correlation: float = 0.95,
    max_residual_ratio: float = 0.5,
) -> dict:
    """Check a dequantizer against the true FP16 weight before trusting it.

    A working 4-bit dequantizer reproduces the original weights closely:
    correlation above ~0.95, and a residual that is a small fraction of
    the weight scale. Measured on a real checkpoint:

        GPTQ  corr 0.9903  residual ratio 0.140  -> correct
        AWQ   corr 0.2343  residual ratio 1.027  -> wrong

    Without this gate the AWQ path produces a residual tensor of the
    right *shape* and a plausible size, and an experiment would report a
    clean BER while measuring nothing real. Shape alone catches nothing.

    Branch on the single ``usable`` field.
    """
    report = {
        "format": expected_format,
        "usable": False,
        "correlation": None,
        "residual_ratio": None,
        "dequantized_std": None,
        "reference_std": None,
        "reason": "",
    }

    try:
        dequantized, fmt = dequantize_layer(
            quantized_module, expected_format=expected_format
        )

        reference = reference_weight.detach().float().cpu()

        if dequantized.shape != reference.shape:
            if dequantized.T.shape == reference.shape:
                dequantized = dequantized.T.contiguous()
            else:
                report["reason"] = (
                    f"shape mismatch: {tuple(dequantized.shape)} vs "
                    f"{tuple(reference.shape)}"
                )
                return report

        correlation = float(
            torch.corrcoef(
                torch.stack(
                    [dequantized.flatten().double(), reference.flatten().double()]
                )
            )[0, 1]
        )

        residual = reference - dequantized
        ratio = float(
            residual.std().item()
            / max(reference.std().item(), 1e-12)
        )

        report["correlation"] = correlation
        report["residual_ratio"] = ratio
        report["dequantized_std"] = float(dequantized.std().item())
        report["reference_std"] = float(reference.std().item())

        if correlation < min_correlation:
            report["reason"] = (
                f"correlation {correlation:.4f} < {min_correlation}: "
                f"{fmt} dequantization does not reproduce the reference "
                "weights. The layout is not understood, so residuals "
                "derived from it would be meaningless."
            )
            return report

        if ratio > max_residual_ratio:
            report["reason"] = (
                f"residual ratio {ratio:.3f} > {max_residual_ratio}: "
                "dequantized weights differ from the reference by more "
                "than 4-bit quantization should."
            )
            return report

        report["usable"] = True
        report["reason"] = (
            f"verified: correlation {correlation:.4f}, residual ratio "
            f"{ratio:.4f}"
        )

    except Exception as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"

    return report
