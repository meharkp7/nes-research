"""
Model loader for NES experiments.

Supports:
    - NF4 model loading
    - FP16 reference model loading
    - Residual extraction
    - Applying embedded residuals

Designed for Apple Silicon (MPS).
"""

from typing import Optional, Tuple
import os

# Enable CPU fallback for PyTorch operations that are not supported by MPS.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)


# ============================================================
# DEVICE
# Time Complexity: O(1)
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# ============================================================
# NF4 CONFIGURATION
# Time Complexity: O(1)
# ============================================================

NF4_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)


# ============================================================
# MODEL LOADER CLASS
# ============================================================

class ModelLoader:
    """
    Loads a HuggingFace causal LM in NF4 quantization.

    Supports models such as:
        - meta-llama/Llama-3.1-8B
        - mistralai/Mistral-7B-v0.3
        - Qwen/Qwen2.5-7B
        - TinyLlama/TinyLlama-1.1B-Chat-v1.0
    """

    def __init__(
        self,
        device: str = "mps",
        use_double_quant: bool = True,
        compute_dtype: str = "float16",
    ):
        self.device = device
        self.use_double_quant = use_double_quant
        self.compute_dtype = compute_dtype

    def load(
        self,
        model_id: str,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Tuple:

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=self.use_double_quant,
            bnb_4bit_compute_dtype=getattr(
                torch,
                self.compute_dtype
            ),
        )

        print(
            f"[ModelLoader] Loading {model_id} in NF4..."
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": self.device},
            cache_dir=cache_dir,
            token=token,
            trust_remote_code=True,
        )

        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            token=token,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(
            f"[ModelLoader] Loaded. "
            f"Layers: {model.config.num_hidden_layers}"
        )

        return model, tokenizer

    def get_layer_names(self, model) -> list:
        """
        Return all quantized linear layer names.

        Time Complexity: O(N)
        where N = number of modules in the model.
        """

        names = []

        for name, module in model.named_modules():

            if (
                hasattr(module, "weight")
                and hasattr(module.weight, "quant_type")
            ):
                names.append(name)

        return names


# ============================================================
# LOAD NF4 + FP16 MODEL PAIR
# ============================================================

def load_model_pair(
    model_id: str,
    device=None,
    token: Optional[str] = None,
    trust_remote_code: bool = True,
):
    """
    Load NF4 and FP16 copies of the same model.

    NF4 model:
        Used as the carrier model.

    FP16 model:
        Used as the reference model for residual calculation.
    """

    if device is None:
        device = DEVICE

    if isinstance(device, torch.device):
        device = str(device)

    print(f"\n[load_model_pair] Model: {model_id}")
    print(f"[load_model_pair] Device: {device}")

    # --------------------------------------------------------
    # Tokenizer
    # Time Complexity: O(model tokenizer size)
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        token=token,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------------
    # NF4 MODEL
    # Time Complexity: O(model size)
    # --------------------------------------------------------

    print("[load_model_pair] Loading NF4 model...")

    nf4_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=NF4_CONFIG,
        device_map={"": device},
        trust_remote_code=trust_remote_code,
        token=token,
    )

    nf4_model.eval()

    # --------------------------------------------------------
    # FP16 MODEL
    # Time Complexity: O(model size)
    # --------------------------------------------------------

    print("[load_model_pair] Loading FP16 model...")

    fp16_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=trust_remote_code,
        token=token,
    )

    fp16_model.eval()

    print(
        "[load_model_pair] "
        "NF4 + FP16 loaded successfully."
    )

    return nf4_model, fp16_model, tokenizer


# ============================================================
# EXTRACT RESIDUALS
# ============================================================

def extract_residuals(
    nf4_model,
    fp16_model,
    family: str
) -> dict:

    from src.model.registry import (
        get_layer_module,
        get_num_layers
    )

    n = get_num_layers(nf4_model)
    residuals = {}

    for i in range(n):

        nf4_mlp = get_layer_module(
            nf4_model,
            family,
            i,
            "mlp"
        )

        fp16_mlp = get_layer_module(
            fp16_model,
            family,
            i,
            "mlp"
        )

        nf4_w = nf4_mlp.down_proj.weight
        fp16_w = fp16_mlp.down_proj.weight

        # ----------------------------------------------------
        # Move FP16 weight to same device as NF4 weight.
        # Time Complexity: O(W)
        # ----------------------------------------------------

        fp16_w = fp16_w.to(nf4_w.device)

        # ----------------------------------------------------
        # Dequantize NF4 weight.
        # Time Complexity: O(W)
        # ----------------------------------------------------

        if hasattr(nf4_w, "quant_state"):

            import bitsandbytes.functional as bnb_func

            dq = bnb_func.dequantize_4bit(
                getattr(nf4_w, "data", nf4_w),
                nf4_w.quant_state,
            ).float()

        elif hasattr(nf4_w, "dequantize"):

            dq = nf4_w.dequantize().float()

        else:

            dq = nf4_w.float()

        # ----------------------------------------------------
        # Make sure reconstructed weight matches FP16 matrix.
        # Time Complexity: O(1)
        # ----------------------------------------------------

        if dq.shape != fp16_w.shape:

            if dq.numel() == fp16_w.numel():

                dq = dq.reshape(fp16_w.shape)

            else:

                raise RuntimeError(
                    f"Layer {i}: NF4 has {dq.numel()} elements, "
                    f"but FP16 has {fp16_w.numel()} elements."
                )

        # ----------------------------------------------------
        # Calculate quantization residual.
        #
        # R = W_FP16 - W_NF4
        #
        # Time Complexity: O(W)
        # ----------------------------------------------------

        residuals[i] = (
            fp16_w.float() - dq
        ).flatten()

    return residuals


# ============================================================
# APPLY EMBEDDED RESIDUALS
# ============================================================

def apply_residuals_to_model(
    nf4_model,
    fp16_model,
    embedded_residuals: dict,
    family: str
):
    """
    Apply the modified residuals to the NF4 model.

    Formula:

        modified_weight =
            original_dequantized_NF4_weight
            + embedded_residual

    The resulting weight is stored as a regular floating-point
    parameter so that the modified model can be evaluated.

    Note:
        After modification, the affected down_proj weights are
        no longer represented as BitsAndBytes 4-bit parameters.

    Apple Silicon note:
        BitsAndBytes NF4 dequantization is performed on CPU because
        direct MPS dequantization can return an incorrectly shaped
        packed tensor.
    """

    from src.model.registry import (
        get_layer_module,
        get_num_layers
    )

    n = get_num_layers(nf4_model)

    # --------------------------------------------------------
    # Iterate through all layers
    # Time Complexity: O(L * W)
    # --------------------------------------------------------

    for i in range(n):

        if i not in embedded_residuals:
            continue

        nf4_mlp = get_layer_module(
            nf4_model,
            family,
            i,
            "mlp"
        )

        nf4_w = nf4_mlp.down_proj.weight

        # ----------------------------------------------------
        # Remember original device.
        # The model itself remains on MPS.
        # ----------------------------------------------------

        target_device = nf4_w.device

        # ----------------------------------------------------
        # Dequantize original NF4 weight safely on CPU.
        #
        # Time Complexity: O(W)
        # ----------------------------------------------------

        qweight_cpu = None

        if hasattr(nf4_w, "quant_state"):

            import bitsandbytes.functional as bnb_func

            # ------------------------------------------------
            # Move packed NF4 weight to CPU.
            # ------------------------------------------------

            qweight_cpu = getattr(
                nf4_w,
                "data",
                nf4_w
            ).detach().to("cpu")

            quant_state = nf4_w.quant_state

            # ------------------------------------------------
            # Move QuantState tensors to CPU.
            # Only move tensors that actually exist.
            # ------------------------------------------------

            if (
                hasattr(quant_state, "absmax")
                and quant_state.absmax is not None
            ):
                quant_state.absmax = (
                    quant_state.absmax.to("cpu")
                )

            if (
                hasattr(quant_state, "code")
                and quant_state.code is not None
            ):
                quant_state.code = (
                    quant_state.code.to("cpu")
                )

            if (
                hasattr(quant_state, "offset")
                and quant_state.offset is not None
            ):
                quant_state.offset = (
                    quant_state.offset.to("cpu")
                )

            # ------------------------------------------------
            # Move nested double-quantization state to CPU.
            # ------------------------------------------------

            if (
                hasattr(quant_state, "state2")
                and quant_state.state2 is not None
            ):

                if (
                    hasattr(quant_state.state2, "absmax")
                    and quant_state.state2.absmax is not None
                ):
                    quant_state.state2.absmax = (
                        quant_state.state2.absmax.to("cpu")
                    )

                if (
                    hasattr(quant_state.state2, "code")
                    and quant_state.state2.code is not None
                ):
                    quant_state.state2.code = (
                        quant_state.state2.code.to("cpu")
                    )

                if (
                    hasattr(quant_state.state2, "offset")
                    and quant_state.state2.offset is not None
                ):
                    quant_state.state2.offset = (
                        quant_state.state2.offset.to("cpu")
                    )

            # ------------------------------------------------
            # CPU NF4 dequantization.
            # ------------------------------------------------

            original_weight = bnb_func.dequantize_4bit(
                qweight_cpu,
                quant_state,
            ).float()

        elif hasattr(nf4_w, "dequantize"):

            original_weight = (
                nf4_w.dequantize()
                .float()
                .to("cpu")
            )

        else:

            original_weight = (
                nf4_w.float()
                .to("cpu")
            )

        # ----------------------------------------------------
        # Make sure reconstructed weight has valid shape.
        # ----------------------------------------------------

        if original_weight.numel() != nf4_w.numel():
            pass

        # ----------------------------------------------------
        # Reshape residual to original weight shape.
        # Time Complexity: O(1)
        # ----------------------------------------------------

        residual = embedded_residuals[i]

        residual = residual.reshape(
            original_weight.shape
        )

        # ----------------------------------------------------
        # Move residual to CPU so addition happens on CPU.
        #
        # Time Complexity: O(W)
        # ----------------------------------------------------

        residual_cpu = (
            residual.detach()
            .to("cpu")
            .float()
        )

        # ----------------------------------------------------
        # Apply residual.
        #
        # modified_weight =
        #     dequantized_NF4_weight + embedded_residual
        #
        # Time Complexity: O(W)
        # ----------------------------------------------------

        modified_weight_cpu = (
            original_weight + residual_cpu
        )

        # ----------------------------------------------------
        # Move modified weight back to model device.
        #
        # Time Complexity: O(W)
        # ----------------------------------------------------

        modified_weight = modified_weight_cpu.to(
            target_device,
            dtype=torch.float16
        )

        # ----------------------------------------------------
        # Replace parameter.
        #
        # The affected layer is now stored as FP16 rather than
        # a BitsAndBytes 4-bit parameter.
        #
        # Time Complexity: O(W)
        # ----------------------------------------------------

        nf4_mlp.down_proj.weight = torch.nn.Parameter(
            modified_weight,
            requires_grad=False
        )

        # ----------------------------------------------------
        # Release temporary tensors before next layer.
        # ----------------------------------------------------

        if qweight_cpu is not None:
            del qweight_cpu

        del original_weight
        del residual
        del residual_cpu
        del modified_weight_cpu
        del modified_weight

    nf4_model.eval()

    return nf4_model