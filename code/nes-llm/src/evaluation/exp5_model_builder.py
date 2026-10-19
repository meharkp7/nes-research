import copy
import torch
import bitsandbytes.functional as bnb

from src.model.registry import get_layer_module


def _move_quant_state_to_cpu(qstate):
    """
    Make a CPU copy of BitsAndBytes QuantState.

    The original NF4 model's QuantState is not modified.

    Time: O(size of quantization metadata)
    Space: O(size of quantization metadata)
    """
    state = copy.deepcopy(qstate)

    for attr, value in vars(state).items():
        if torch.is_tensor(value):
            setattr(state, attr, value.cpu())

        elif hasattr(value, "__dict__"):
            for sub_attr, sub_value in vars(value).items():
                if torch.is_tensor(sub_value):
                    setattr(value, sub_attr, sub_value.cpu())

    return state


def _dequantize_weight_cpu(nf4_weight):
    """
    Dequantize one NF4 weight completely on CPU.

    Time: O(number of weights)
    Space: O(number of weights)
    """
    packed_weight = nf4_weight.data.detach().cpu()

    qstate = _move_quant_state_to_cpu(
        nf4_weight.quant_state
    )

    dequantized = bnb.dequantize_4bit(
        packed_weight,
        quant_state=qstate
    )

    return dequantized.float()


@torch.no_grad()
def build_embedded_eval_model(
    nf4_model,
    fp16_model,
    embedded_residuals,
    family,
):
    """
    Build the Exp5 embedded evaluation model.

    For every affected layer:

        W_embedded = W_NF4 + R_embedded

    The reconstructed weight is placed into the
    corresponding FP16 model's down_proj layer.

    IMPORTANT:
    - NF4 model is NOT modified.
    - Shared model_loader.py is NOT modified.
    - This builder is specific to Exp5.

    Time: O(total number of affected weights)
    Space: O(size of one dequantized layer)
    """

    num_layers = len(nf4_model.model.layers)

    print("\nBuilding Exp5 embedded evaluation model...")

    for layer_id in range(num_layers):

        if layer_id not in embedded_residuals:
            continue

        # ---------------------------------------------------------
        # Get MLP modules
        # Time: O(1)
        # ---------------------------------------------------------
        nf4_mlp = get_layer_module(
            nf4_model,
            family,
            layer_id,
            "mlp",
        )

        fp16_mlp = get_layer_module(
            fp16_model,
            family,
            layer_id,
            "mlp",
        )

        # The carrier tensor is specifically down_proj.weight
        nf4_weight = nf4_mlp.down_proj.weight
        residual = embedded_residuals[layer_id]

        # ---------------------------------------------------------
        # Dequantize NF4 weight on CPU
        # Time: O(W)
        # Space: O(W)
        # ---------------------------------------------------------
        nf4_fp32 = _dequantize_weight_cpu(
            nf4_weight
        )

        # ---------------------------------------------------------
        # Move embedded residual to CPU
        # Time: O(W)
        # Space: O(W)
        # ---------------------------------------------------------
        residual_cpu = (
            residual.detach()
            .cpu()
            .float()
        )

        # ---------------------------------------------------------
        # Validate shape
        # Time: O(1)
        # ---------------------------------------------------------
        if residual_cpu.numel() != nf4_fp32.numel():
            raise ValueError(
                f"Layer {layer_id}: residual size mismatch. "
                f"Residual={residual_cpu.numel()}, "
                f"NF4 dequantized={nf4_fp32.numel()}"
            )

        residual_cpu = residual_cpu.reshape(
            nf4_fp32.shape
        )

        # ---------------------------------------------------------
        # Reconstruct embedded weight
        #
        # W_embedded = W_NF4 + R_embedded
        #
        # Time: O(W)
        # Space: O(W)
        # ---------------------------------------------------------
        embedded_weight_cpu = (
            nf4_fp32 + residual_cpu
        )

        # ---------------------------------------------------------
        # Put reconstructed weight into FP16 down_proj
        # Time: O(W)
        # ---------------------------------------------------------
        target_weight = fp16_mlp.down_proj.weight

        embedded_weight = embedded_weight_cpu.to(
            device=target_weight.device,
            dtype=target_weight.dtype,
        )

        fp16_mlp.down_proj.weight = torch.nn.Parameter(
            embedded_weight,
            requires_grad=False,
        )

        print(f"  Layer {layer_id}: embedded")

        # Release temporary tensors
        del nf4_fp32
        del residual_cpu
        del embedded_weight_cpu
        del embedded_weight

    print("Exp5 embedded evaluation model ready.")

    return fp16_model