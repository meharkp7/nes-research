import torch


class QuantizationModel:
    """
    Generic quantization model.

    Provides utilities required by
    Quantization-Aware Embedding (QAE).

    This version implements uniform
    quantization.

    Later versions will support

        - NF4
        - GPTQ
        - AWQ
        - INT8
    """

    def __init__(
        self,
        step_size,
    ):
        self.step_size = float(step_size)

    def quantize(
        self,
        values,
    ):
        """
        Quantize values.
        """

        return (
            torch.round(
                values / self.step_size
            )
            * self.step_size
        )

    def dequantize(
        self,
        values,
    ):
        """
        Uniform quantization is symmetric.

        This function exists for API
        consistency.
        """

        return values

    def nearest_level(
        self,
        values,
    ):
        """
        Return nearest quantization level.
        """

        return self.quantize(values)

    def distance_to_level(
        self,
        values,
    ):
        """
        Distance to nearest quantization level.
        """

        return (
            values
            - self.nearest_level(values)
        ).abs()

    def decision_boundary_distance(
        self,
        values,
    ):
        """
        Distance to nearest decision boundary.
        """

        half = self.step_size / 2

        distance = self.distance_to_level(
            values
        )

        return half - distance

    def safety_margin(
        self,
        values,
    ):
        """
        Alias used by the embedding
        algorithm.

        Larger margin
        =
        more robust carrier.
        """

        return self.decision_boundary_distance(
            values
        )