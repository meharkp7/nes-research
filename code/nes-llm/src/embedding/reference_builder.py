import torch

from bitsandbytes.functional import (
    quantize_4bit,
    dequantize_4bit,
)


class ReferenceBuilder:
    """
    Builds the V2 reference residual.

    The reference residual is obtained by
    performing one additional real NF4
    quantization cycle.

    Pipeline

        FP16 Weight
              │
              ▼
        Original Residual
              │
              ▼
        Reconstruct Weight
              │
              ▼
        NF4 Quantize
              │
              ▼
        NF4 Dequantize
              │
              ▼
        Reference Residual
    """

    @staticmethod
    def build(
        fp16_weight: torch.Tensor,
        nf4_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the residual after one
        additional NF4 quantization cycle.
        """

        fp16_weight = fp16_weight.float().cpu()
        nf4_weight = nf4_weight.float().cpu()

        #
        # Original residual
        #
        original_residual = (
            fp16_weight - nf4_weight
        )

        #
        # Reconstruct FP16 weight
        #
        reconstructed_weight = (
            nf4_weight + original_residual
        )

        #
        # Real NF4 quantization
        #
        qweight, qstate = quantize_4bit(
            reconstructed_weight,
            quant_type="nf4",
        )

        requantized_weight = dequantize_4bit(
            qweight,
            quant_state=qstate,
        ).float()

        #
        # Reference residual
        #
        reference_residual = (
            reconstructed_weight
            - requantized_weight
        )

        return reference_residual

    @staticmethod
    def build_from_residual(
        residual: torch.Tensor,
        nf4_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience overload.

        Starts from an already computed
        residual tensor.
        """

        fp16_weight = (
            nf4_weight + residual
        )

        return ReferenceBuilder.build(
            fp16_weight,
            nf4_weight,
        )