import torch


class QCAEObjective:
    """
    Multi-objective scoring function.

    J =
        (lambda_R * Robustness
        + lambda_I * Information)
        /
        (1 + lambda_D * Distortion)
    """

    def __init__(
        self,
        lambda_r=0.45,
        lambda_i=0.40,
        lambda_d=0.15,
    ):

        self.lambda_r = lambda_r
        self.lambda_i = lambda_i
        self.lambda_d = lambda_d

    def _normalize(
        self,
        x,
    ):

        return (
            x - x.min()
        ) / (
            x.max() - x.min()
            + 1e-8
        )

    def score(
        self,
        feature_matrix,
    ):

        features = self._normalize(
            feature_matrix
        )

        robustness = (
            features[:,3]
            + features[:,4]
            + features[:,5]
        ) / 3

        information = (
            features[:,0]
            + features[:,1]
            + features[:,2]
        ) / 3

        distortion = (
            features[:,6]
            + features[:,7]
            + features[:,8]
        ) / 3

        score = (

            self.lambda_r * robustness

            +

            self.lambda_i * information

        ) / (

            1

            +

            self.lambda_d * distortion

        )

        return score