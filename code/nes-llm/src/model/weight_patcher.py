"""
Weight Patcher — applies embedded residuals back into float16 model weights.

Correct formula:
    W_new = W_nf4_dequant + R_embedded

Where:
    W_nf4_dequant = dequantize(quantize(W_fp16))  — stable NF4 baseline
    R_embedded    = sign_embed(R_original)         — modified residual

This guarantees the effective weight perturbation is bounded by
2 × |R_original| ≈ 0.003, invisible at the functional level.
"""

from typing import Dict
import torch


class WeightPatcher:
    """
    Applies embedded residuals to float16 model weights.

    Usage:
        patcher = WeightPatcher()
        patcher.patch(module_refs, nf4_dequant, embedded_residuals)
        # ... evaluate model ...
        patcher.restore(module_refs, nf4_dequant, original_residuals)
    """

    def patch(
        self,
        module_refs:        Dict[int, object],
        nf4_dequant:        Dict[int, torch.Tensor],
        embedded_residuals: Dict[int, torch.Tensor],
    ) -> int:
        """
        Apply embedded residuals: W_new = W_nf4_dequant + R_embedded

        Args:
            module_refs:        {layer_id: nn.Module}
            nf4_dequant:        {layer_id: W_nf4_dequant tensor}
            embedded_residuals: {layer_id: R_embedded tensor}

        Returns:
            Number of layers patched.
        """
        patched = 0
        with torch.no_grad():
            for lid, module in module_refs.items():
                if lid not in embedded_residuals or lid not in nf4_dequant:
                    continue
                try:
                    W_nf4 = nf4_dequant[lid]
                    R_emb = embedded_residuals[lid]
                    W_new = (W_nf4 + R_emb).to(
                        module.weight.dtype
                    ).to(module.weight.device)
                    module.weight.data.copy_(W_new)
                    patched += 1
                except Exception as e:
                    print(f"[WeightPatcher] Failed lid={lid}: {e}")

        print(f"[WeightPatcher] Patched {patched}/{len(module_refs)} layers")
        return patched

    def restore(
        self,
        module_refs:       Dict[int, object],
        nf4_dequant:       Dict[int, torch.Tensor],
        original_residuals:Dict[int, torch.Tensor],
    ) -> int:
        """
        Restore original weights: W_original = W_nf4_dequant + R_original

        Args:
            module_refs:        {layer_id: nn.Module}
            nf4_dequant:        {layer_id: W_nf4_dequant tensor}
            original_residuals: {layer_id: R_original tensor}

        Returns:
            Number of layers restored.
        """
        restored = 0
        with torch.no_grad():
            for lid, module in module_refs.items():
                if lid not in original_residuals or lid not in nf4_dequant:
                    continue
                try:
                    W_fp16 = (nf4_dequant[lid] + original_residuals[lid]).to(
                        module.weight.dtype
                    ).to(module.weight.device)
                    module.weight.data.copy_(W_fp16)
                    restored += 1
                except Exception as e:
                    print(f"[WeightPatcher] Failed restore lid={lid}: {e}")

        print(f"[WeightPatcher] Restored {restored}/{len(module_refs)} layers")
        return restored