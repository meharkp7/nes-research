import torch


class ConfidenceEstimator:
    """
    Estimates carrier confidence before
    embedding.

    High confidence means a carrier is
    expected to preserve embedded bits
    under perturbation.
    """

    def __init__(
        self,
        eps=1e-8,
    ):

        self.eps = eps

    def compute(
        self,
        residual,
        margins,
    ):

        residual = residual.abs()

        confidence = residual / (

            residual

            + margins

            + self.eps

        )

        return confidence

    def statistics(
        self,
        confidence,
    ):

        return {

            "mean": confidence.mean().item(),

            "std": confidence.std().item(),

            "min": confidence.min().item(),

            "max": confidence.max().item(),

        }


def main():

    residual = torch.randn(10000)

    margins = torch.rand(10000) * 0.02

    estimator = ConfidenceEstimator()

    confidence = estimator.compute(

        residual,

        margins,

    )

    print()

    print(

        estimator.statistics(

            confidence

        )

    )


if __name__ == "__main__":

    main()