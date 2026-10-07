"""
Format-specific quantization adapters.

Each adapter answers one question: given a quantized linear layer, what
is its dequantized FP weight? Everything downstream — residual
extraction, QACI, embedding — is written against ``R = W_FP16 - W_deq``
and does not care which format produced ``W_deq``.

Keeping this format-specific matters. GPTQ packs weights along a
different axis than NF4, stores group-wise integer zeros and uses a
desc_act permutation, and AWQ stores group scales with a zero point and
-- critically -- packs its nibbles in an interleaved order rather than
in sequence. An adapter that silently treats any of these as NF4, or as
each other, will either crash or, worse, produce plausible-looking
residuals that are meaningless.

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

# GPTQ packs the least-significant nibble first: element ``i`` lives at
# nibble ``i % 8`` of word ``i // 8``. Verified bit-exact against the
# AutoGPTQ reference unpack and against a real checkpoint (correlation
# 0.9903 vs the FP16 reference).
STANDARD_NIBBLE_ORDER = tuple(range(8))


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
    # The four packed tensors live under `.weight` on the exp9
    # checkpoint holder, and at module level on gptqmodel's runtime
    # TorchLinear — whose `.weight` is a metadata shim that carries
    # none of them (reading through it raised AttributeError there).
    src = (
        module
        if getattr(module, "qweight", None) is not None
        else module.weight
    )

    qweight = src.qweight.detach().to("cpu")
    qzeros = src.qzeros.detach().to("cpu")
    scales = src.scales.detach().to("cpu").float()
    g_idx = src.g_idx.detach().to("cpu")

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
        qweight, out_features, in_features, axis=1,
        order=STANDARD_NIBBLE_ORDER,
    )

    # The -1 bias on zero points is applied at pack time, so
    # dequantization adds 1 back (matching AutoGPTQ's `zeros + 1`).
    zeros = (
        _unpack_4bit(
            qzeros, num_groups, in_features, axis=-1,
            order=STANDARD_NIBBLE_ORDER,
        )
        + 1
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
    order: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """Unpack int32-packed unsigned 4-bit values into ``[rows, columns]``.

    Two independent conventions decide whether the result is the weight
    matrix or plausible garbage, and getting either wrong produces a
    tensor of the right shape and plausible magnitude:

    1.  **Which axis packs.** Verified against AutoGPTQ
        ``QuantLinear.forward``. The two packed tensors use different
        nibble axis orders:

            qweight  right_shift(unsqueeze(qweight, 1).expand(-1, 8, -1),
                                 [0,4,...,28].unsqueeze(-1))
                     -> shape ``[words, 8, out]`` -> ``[in, out]``

            qzeros   right_shift(unsqueeze(qzeros, 2).expand(-1, -1, 8),
                                 [0,4,...,28].unsqueeze(0))
                     -> shape ``[groups, out_words, 8]`` -> ``[groups, out]``

        So ``axis=1`` expands the nibble dimension after the packed rows
        (weights), and ``axis=-1`` expands it last (zeros). Using the
        wrong one silently transposes the weight matrix rather than
        erroring.

    2.  **Which nibble holds which element.** ``order[k]`` is the nibble
        position holding the element whose in-word index is ``k``.

        GPTQ packs in order, so ``order`` is ``(0, 1, ..., 7)``. AWQ does
        **not**: AutoAWQ's ``WQLinear_GEMM.from_linear`` packs with
        ``order_map = [0, 2, 4, 6, 1, 3, 5, 7]``, meaning element ``k``
        lands in nibble ``AWQ_NIBBLE_ORDER[k] = [0, 4, 1, 5, 2, 6, 3, 7]``.
        That is an even/odd interleave, not a cyclic rotation, so an
        8-rotation x zero-point search cannot reach it -- which is why
        the AWQ path sat at correlation 0.2343 through 24 attempted
        variants before this was read out of the reference source.
        Measured on a real checkpoint: identity order gives 0.2343, the
        correct order gives 0.9890.

    Element ``i`` lives at nibble ``i % 8`` of int32 word ``i // 8`` (for
    ``axis=1``, word ``i // 8`` of the packed rows), with the mapping
    into nibble positions given by ``order``.
    """
    if order is None:
        order = STANDARD_NIBBLE_ORDER

    if sorted(order) != list(range(8)):
        raise ValueError(
            f"nibble order must be a permutation of 0..7, got {order}"
        )

    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    nibble_axis = torch.as_tensor(order, dtype=torch.long)

    if axis == 1:
        expanded = packed.unsqueeze(1).expand(-1, 8, -1)
        unpacked = (
            expanded >> shifts.to(packed.dtype).reshape(1, 8, 1)
        ) & 0x0F
        # Row r is at nibble r % 8, so permute the nibble axis.
        unpacked = unpacked[:, nibble_axis, :]
    elif axis == -1:
        expanded = packed.unsqueeze(2).expand(-1, -1, 8)
        unpacked = (
            expanded >> shifts.to(packed.dtype).reshape(1, 1, 8)
        ) & 0x0F
        # Column c is at nibble c % 8, so permute the nibble axis.
        unpacked = unpacked[:, :, nibble_axis]
    else:
        raise ValueError(f"Unsupported nibble axis: {axis}")

    return unpacked.reshape(rows, columns).float()


# ---------------------------------------------------------------------
# AWQ
# ---------------------------------------------------------------------

# AutoAWQ's ``WQLinear_GEMM.from_linear`` packs with
#
#     order_map = [0, 2, 4, 6, 1, 3, 5, 7]   # nibble i <- element order_map[i]
#
# so element ``k`` is found in nibble ``order_map.index(k)``:
#
#     AWQ_NIBBLE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
#
# This is an even/odd interleave, not a cyclic rotation, so searching
# rotations of the nibble field cannot reach it. It applies to ``qzeros``
# as well, which is packed by the same loop.
AWQ_NIBBLE_ORDER: Tuple[int, ...] = (0, 4, 1, 5, 2, 6, 3, 7)


def dequantize_awq_layer(module) -> torch.Tensor:
    """Dequantize an AWQ-packed linear layer to FP32 on CPU.

    Real checkpoint layout (Qwen2.5-3B-Instruct-AWQ, group_size=128,
    zero_point=true, version=gemm), for a layer with ``in`` input and
    ``out`` output features::

        qweight   (in, out // 8)        e.g. down_proj (11008, 256)
        qzeros    (in // group, out // 8)   e.g. down_proj (86, 256)
        scales    (in // group, out)        e.g. down_proj (86, 2048)

    Confirmed against the shapes of every module in the checkpoint:
    ``gate_proj`` (in=2048, out=11008) stores ``(2048, 1376)`` /
    ``(16, 1376)`` / ``(16, 11008)``, and ``down_proj`` (in=11008,
    out=2048) stores ``(11008, 256)`` / ``(86, 256)`` / ``(86, 2048)``.

    Two things distinguish this from GPTQ:

    -   **The weight packs along the output axis** (``axis=-1``), the
        opposite of GPTQ, which packs along the input axis.
    -   **The nibble order is interleaved**, not sequential. See
        ``AWQ_NIBBLE_ORDER``. Using sequential order here reproduced
        neither the weights nor the zero points: correlation 0.2343
        against the FP16 reference, residual ratio 1.027.

    Groups run along the *input* axis: group ``g`` covers input rows
    ``[g * group_width, (g + 1) * group_width)`` with
    ``group_width = in_features // num_groups``.

    AWQ does **not** bias zero points by -1 at pack time, so no ``+ 1``
    is applied. Copying GPTQ's correction would shift every dequantized
    weight by one scale unit.

    The packed layout yields ``[in, out]``; ``nn.Linear.weight`` is
    ``[out, in]``, so the result is transposed before returning.
    """
    # qweight sits under `.weight` on the checkpoint holder exp8
    # validated, and at module level on autoawq's runtime class
    # (WQLinear_GEMM has no `.weight` at all); qzeros and scales are
    # read from the module on both paths.
    qweight = (
        module
        if getattr(module, "qweight", None) is not None
        else module.weight
    ).qweight.detach().to("cpu")
    qzeros = module.qzeros.detach().to("cpu")
    scales = module.scales.detach().to("cpu").float()

    pack_factor = 32 // 4

    in_features = qweight.shape[0]
    out_features = qweight.shape[1] * pack_factor
    num_groups = scales.shape[0]

    if scales.shape[1] != out_features:
        raise RuntimeError(
            f"AWQ scales second dim {scales.shape[1]} does not match "
            f"qweight columns * 8 = {out_features}"
        )

    if qzeros.shape != (num_groups, out_features // pack_factor):
        raise RuntimeError(
            "AWQ qzeros shape "
            f"{tuple(qzeros.shape)} does not match the expected "
            f"{(num_groups, out_features // pack_factor)}"
        )

    if in_features % num_groups != 0:
        raise RuntimeError(
            f"AWQ scale groups ({num_groups}) do not evenly divide "
            f"in_features ({in_features})"
        )

    values = _unpack_4bit(
        qweight, in_features, out_features,
        axis=-1, order=AWQ_NIBBLE_ORDER,
    )
    zeros = _unpack_4bit(
        qzeros, num_groups, out_features,
        axis=-1, order=AWQ_NIBBLE_ORDER,
    )

    group_width = in_features // num_groups

    scale_expanded = scales.repeat_interleave(group_width, dim=0)
    zero_expanded = zeros.repeat_interleave(group_width, dim=0)

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

def _correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation, returning NaN instead of raising on constants."""
    try:
        value = float(
            torch.corrcoef(
                torch.stack([a.flatten().double(), b.flatten().double()])
            )[0, 1]
        )
    except Exception:
        return float("nan")
    return value


def _nan_median_along(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Median along ``dim``, ignoring NaN entries.

    NaN marks a position that says nothing about the scale: a value the
    quantizer collapsed to zero, or a reference weight that is exactly
    zero. Because such positions are skipped rather than treated as 0,
    an input channel that collapsed to zero in the dequantized tensor
    cannot be scaled back into a match -- it stays a mismatch and is
    counted against the gate. Dropping the information loss by fitting
    through it is the failure mode this helper exists to avoid.
    """
    valid = ~torch.isnan(t)
    probe = torch.where(valid, t, torch.full_like(t, float("inf")))
    order = torch.argsort(probe, dim=dim)
    ordered = torch.gather(t, dim, order)
    ordered_ok = torch.gather(valid, dim, order)
    count = ordered_ok.sum(dim=dim, keepdim=True)
    middle = ((count - 1) // 2).clamp(min=0)
    taken = torch.gather(ordered, dim, middle)
    return torch.where(count > 0, taken, torch.full_like(taken, float("nan")))


def absorbed_scale(
    dequantized: torch.Tensor,
    reference: torch.Tensor,
    allow_output_scale: bool = False,
    passes: int = 4,
) -> torch.Tensor:
    """Fit the per-channel scale AWQ folds out of the weight matrix.

    AWQ does not quantize ``W``. It quantizes ``W`` after scaling its
    channels, and folds the inverse into whatever precedes the linear
    layer:

        q/k/v/gate/up   input-channel scale  -> preceding LayerNorm gamma
        up/down         paired scale on the intermediate channel
                        (up's output channels, down's input channels)

    So ``dequantized ~= reference * s`` with ``s`` a per-channel factor
    that is *not* a dequantization error -- the original network is
    numerically unchanged, because the same factor reappears in the
    LayerNorm. Measured on the real checkpoint, layer 0:

        input_layernorm   awq / ref median  1.340
        weight factor     awq / ref median  1.322
        post_attention    awq / ref median  1.790   (gate 1.781)

    The two agree, which is what makes the scale an absorbed quantity
    rather than an artifact of our unpacking.

    Returns the multiplicative model of ``|dequantized / reference|``
    in the ``[out, in]`` layout of ``nn.Linear.weight``:

        dim 0 -> per *input* channel (always fitted)
        dim 1 -> per *output* channel (only when ``allow_output_scale``)

    Only ``up_proj`` may set ``allow_output_scale``: its output channels
    are the intermediate channel that carries the up/down pair. For
    every other module the column fit is the whole model, which keeps the
    gate at 252-module scale from becoming a free rescale.

    The alternating medians converge in one pass when a single factor is
    present, so ``passes=1`` is exact for everything except ``up_proj``.
    """
    if not allow_output_scale:
        passes = 1

    valid = (dequantized != 0) & (reference != 0)
    log_ratio = torch.where(
        valid,
        torch.log(dequantized.abs().clamp(min=1e-30))
        - torch.log(reference.abs().clamp(min=1e-30)),
        torch.full_like(dequantized, float("nan")),
    )

    residual = log_ratio.clone()
    for _ in range(max(1, passes)):
        residual = residual - _nan_median_along(residual, 0)
        if allow_output_scale:
            residual = residual - _nan_median_along(residual, 1)

    model = torch.nan_to_num(log_ratio - residual, nan=0.0)
    # exp() must not underflow to 0 or overflow to inf: a zero divisor
    # would turn a collapsed channel into NaN and a NaN correlation
    # compares False against every threshold, passing the gate silently.
    return torch.exp(model.clamp(-60.0, 60.0))


def verify_dequantization(
    quantized_module,
    reference_weight: torch.Tensor,
    expected_format: Optional[str] = None,
    min_correlation: float = 0.95,
    max_residual_ratio: float = 0.5,
    module_name: Optional[str] = None,
) -> dict:
    """Check a dequantizer against the true FP16 weight before trusting it.

    A working 4-bit dequantizer reproduces the original weights closely:
    correlation above ~0.95, and a residual that is a small fraction of
    the weight scale. Measured on a real checkpoint:

        GPTQ  corr 0.9903  residual ratio 0.140  -> correct
        AWQ   corr 0.2343  residual ratio 1.027  -> wrong
        AWQ (correct nibble order) corr 0.9890   -> correct

    The middle row is the instructive one: the AWQ failure was not a
    shape error, which the shape checks would have caught, but a wrong
    nibble order inside each 32-bit word. Every candidate layout was
    tried first and the best reached 0.2343 -- the value this gate
    caught.

    Without this gate the AWQ path produces a residual tensor of the
    right *shape* and a plausible size, and an experiment would report a
    clean BER while measuring nothing real. Shape alone catches nothing.

    AWQ additionally needs the scale it absorbed to be removed before
    comparison, for the reason set out in :func:`absorbed_scale`. The
    thresholds are unchanged; only the comparison basis is corrected.
    Removing that scale does *not* blunt the gate. Run over all 252
    modules of the real checkpoint:

        correct nibble order    249 / 252 pass
        wrong   nibble order      0 / 252 pass

    The three that fail are checkpoint defects, not unpacking defects,
    and they are reported as such.

    Pass ``module_name`` for AWQ so ``up_proj`` gets its output-channel
    scale. Without it the fit is column-only and a real ``up_proj`` can
    fail the gate; a false rejection costs a NOT_RUN, a wrong acceptance
    costs a meaningless BER.

    Branch on the single ``usable`` field.
    """
    report = {
        "format": expected_format,
        "usable": False,
        "correlation": None,
        "residual_ratio": None,
        "raw_correlation": None,
        "raw_residual_ratio": None,
        "scale_corrected": False,
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

        raw_correlation = _correlation(dequantized, reference)
        raw_ratio = float(
            (reference - dequantized).std().item()
            / max(reference.std().item(), 1e-12)
        )

        report["raw_correlation"] = raw_correlation
        report["raw_residual_ratio"] = raw_ratio
        report["dequantized_std"] = float(dequantized.std().item())
        report["reference_std"] = float(reference.std().item())

        correlation, ratio = raw_correlation, raw_ratio
        compared = dequantized
        suffix = ""

        if fmt == "awq":
            scale = absorbed_scale(
                dequantized,
                reference,
                allow_output_scale=bool(
                    module_name and module_name.endswith("up_proj")
                ),
            )
            # A zero stays zero: collapsing to zero is lost information,
            # not a value waiting for the right multiplier.
            compared = torch.where(
                dequantized == 0, dequantized, dequantized / scale
            )
            correlation = _correlation(compared, reference)
            ratio = float(
                (reference - compared).std().item()
                / max(reference.std().item(), 1e-12)
            )
            report["scale_corrected"] = True
            suffix = (
                ", raw correlation "
                f"{raw_correlation:.4f} before removing the absorbed "
                "per-channel scale"
            )

        report["correlation"] = correlation
        report["residual_ratio"] = ratio

        # NaN compares False either way round, which would let a constant
        # or all-zero dequantization through both thresholds.
        if not (correlation >= min_correlation):
            report["reason"] = (
                f"correlation {correlation:.4f} < {min_correlation}: "
                f"{fmt} dequantization does not reproduce the reference "
                "weights"
                + (
                    " once the absorbed per-channel scale is removed"
                    if report["scale_corrected"]
                    else ""
                )
                + ". The layout is not understood, so residuals derived "
                "from it would be meaningless."
            )
            return report

        if not (ratio <= max_residual_ratio):
            report["reason"] = (
                f"residual ratio {ratio:.3f} > {max_residual_ratio}: "
                "dequantized weights differ from the reference by more "
                "than 4-bit quantization should."
            )
            return report

        report["usable"] = True
        report["reason"] = (
            f"verified: correlation {correlation:.4f}, residual ratio "
            f"{ratio:.4f}{suffix}"
        )

    except Exception as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"

    return report
