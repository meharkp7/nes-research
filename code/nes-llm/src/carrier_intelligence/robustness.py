import torch


class RobustnessAnalyzer:
    """
    Computes robustness-related features for each carrier.

    Features
    --------
    1. Quantization Error
    2. Stability
    3. Robustness Score
    """

    def __init__(
        self,
        eps: float = 1e-8,
    ):
        self.eps = eps

    def quantization_error(
        self,
        fp16_weight: torch.Tensor,
        quantized_weight: torch.Tensor,
    ) -> torch.Tensor:

        return (
            fp16_weight.float()
            - quantized_weight.float()
        ).abs()

    def stability(
        self,
        quantization_error: torch.Tensor,
    ) -> torch.Tensor:

        return 1.0 / (
            quantization_error + self.eps
        )

    def robustness(
        self,
        fp16_weight: torch.Tensor,
        quantized_weight: torch.Tensor,
    ):

        q_error = self.quantization_error(
            fp16_weight,
            quantized_weight,
        )

        stability = self.stability(
            q_error
        )

        robustness = (
            0.5 * stability
            + 0.5 * (
                1.0
                / (
                    1.0
                    + q_error
                )
            )
        )

        return {
            "quantization_error": q_error,
            "stability": stability,
            "robustness": robustness,
        }