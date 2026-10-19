import torch
import torch.nn.functional as F


class InformationAnalyzer:
    """
    Information richness of every carrier.
    """

    def __init__(
        self,
        window=3,
    ):
        self.window = window

    def magnitude(
        self,
        residual,
    ):

        return residual.abs()

    def local_variance(
        self,
        residual,
    ):

        x = residual.float().unsqueeze(0).unsqueeze(0)

        mean = F.avg_pool2d(
            x,
            self.window,
            stride=1,
            padding=self.window // 2,
        )

        mean_sq = F.avg_pool2d(
            x * x,
            self.window,
            stride=1,
            padding=self.window // 2,
        )

        var = mean_sq - mean * mean

        return var.squeeze()

    def local_std(
        self,
        residual,
    ):

        return torch.sqrt(
            self.local_variance(
                residual
            )
            + 1e-8
        )

    def information(
        self,
        residual,
    ):

        magnitude = self.magnitude(
            residual
        )

        variance = self.local_variance(
            residual
        )

        std = self.local_std(
            residual
        )

        score = (
            magnitude
            + variance
            + std
        ) / 3

        return {
            "magnitude": magnitude,
            "variance": variance,
            "std": std,
            "information": score,
        }