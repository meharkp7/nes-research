"""
Residual Extractor — computes true NF4 quantization residuals.

For each target layer:
    W_fp16       = original float16 weight
    W_nf4_dequant = dequantize(quantize(W_fp16))   ← what NF4 stores
    R            = W_fp16 - W_nf4_dequant           ← TRUE residual

NES embeds bits into R. The effective weight perturbation is bounded
by 2×|R| ≈ 0.003, which is within normal NF4 quantization variance.

This means embedding is invisible at the functional level — the model
cannot distinguish embedded weights from normal NF4 quantization error.
"""

from typing import Dict, List, Optional, Tuple
import torch

LLAMA_TARGET_MODULES      = ["down_proj", "gate_proj", "up_proj"]
CONSERVATIVE_TARGET_MODULES = ["down_proj"]


class ResidualExtractor:
    """
    Extracts true NF4 quantization residuals from a float16 model.

    Unlike the previous version (which returned dequantized weights),
    this computes the actual quantization residual:
        R = W_fp16 - dequantize(quantize(W_fp16))

    Returns:
        residuals:       {layer_id: R}             — embed bits here
        nf4_dequant:     {layer_id: W_nf4_dequant} — needed for patching
        module_refs:     {layer_id: module}         — for weight patching
        module_names:    {layer_id: str}            — for logging
    """

    def __init__(
        self,
        target_modules: List[str] = None,
        layer_range:    Optional[Tuple[int, int]] = None,
    ):
        self.target_modules = target_modules or LLAMA_TARGET_MODULES
        self.layer_range    = layer_range

    def extract(self, model) -> Tuple[
        Dict[int, torch.Tensor],   # residuals
        Dict[int, torch.Tensor],   # nf4_dequant
        Dict[int, object],         # module_refs
        Dict[int, str],            # module_names
    ]:
        """
        Extract true quantization residuals from all target layers.

        Args:
            model: Float16 HuggingFace model (NOT NF4 BitsAndBytes model).
                   Load with: AutoModelForCausalLM.from_pretrained(..., dtype=torch.float16)

        Returns:
            (residuals, nf4_dequant, module_refs, module_names)
        """
        import bitsandbytes.functional as bnb_func

        residuals    = {}
        nf4_dequant  = {}
        module_refs  = {}
        module_names = {}
        layer_id     = 0

        num_layers = model.config.num_hidden_layers
        lo, hi     = self.layer_range if self.layer_range else (0, num_layers)

        for layer_idx in range(num_layers):
            if not (lo <= layer_idx < hi):
                continue

            layer = model.model.layers[layer_idx]

            for mod_name in self.target_modules:
                module = getattr(layer.mlp, mod_name, None)
                if module is None or not hasattr(module, 'weight'):
                    continue

                try:
                    W_fp16 = module.weight.data.float().cpu()

                    # Simulate NF4 quantization round-trip
                    q_weight, q_state = bnb_func.quantize_4bit(
                        W_fp16, quant_type="nf4", compress_statistics=True
                    )
                    W_nf4 = bnb_func.dequantize_4bit(q_weight, q_state).float()

                    # True residual = FP16 - NF4_dequant (magnitude ~0.001)
                    R = W_fp16 - W_nf4

                    residuals[layer_id]   = R
                    nf4_dequant[layer_id] = W_nf4
                    module_refs[layer_id] = module
                    module_names[layer_id] = f"layer{layer_idx}.{mod_name}"
                    layer_id += 1

                except Exception as e:
                    print(f"[ResidualExtractor] Skipping layer{layer_idx}.{mod_name}: {e}")

        total_params = sum(t.numel() for t in residuals.values())
        mean_mag     = sum(t.abs().mean().item() for t in residuals.values()) / max(len(residuals), 1)
        print(f"[ResidualExtractor] {layer_id} tensors, {total_params:,} params, "
              f"mean residual magnitude={mean_mag:.6f}")
        return residuals, nf4_dequant, module_refs, module_names