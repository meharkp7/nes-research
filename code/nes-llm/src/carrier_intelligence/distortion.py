import torch


class DistortionAnalyzer:
    """
    Estimates embedding distortion.
    """

    def __init__(
        self,
        eps=1e-8,
    ):
        self.eps = eps

    def embedding_cost(
        self,
        residual,
    ):

        return 1.0 / (
            residual.abs()
            + self.eps
        )

    def zscore(
        self,
        residual,
    ):

        mean = residual.mean()

        std = residual.std()

        return (
            residual - mean
        ) / (
            std + self.eps
        )

    def distortion(
        self,
        residual,
    ):

        cost = self.embedding_cost(
            residual
        )

        z = self.zscore(
            residual
        ).abs()

        score = (
            cost + z
        ) / 2

        return {
            "cost": cost,
            "zscore": z,
            "distortion": score,
        }