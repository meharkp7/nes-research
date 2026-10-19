import torch


class CarrierReliability:
    """
    Computes the statistical reliability
    of every carrier using Mahalanobis distance.

    Smaller distance
        -> higher reliability

    Larger distance
        -> lower reliability
    """

    def __init__(

        self,

        eps=1e-8,

    ):

        self.eps = eps

    ##########################################################

    def compute(

        self,

        features,

    ):

        features = features.float()

        ######################################################
        # Mean feature vector
        ######################################################

        mean = features.mean(
            dim=0,
            keepdim=True,
        )

        ######################################################
        # Center features
        ######################################################

        centered = features - mean

        ######################################################
        # Covariance
        ######################################################

        cov = (

            centered.T @ centered

        ) / (

            features.shape[0] - 1

        )

        ######################################################
        # Regularization
        ######################################################

        cov += (

            self.eps

            * torch.eye(

                cov.shape[0],

                device=features.device,

            )

        )

        ######################################################
        # Inverse covariance
        ######################################################

        inv_cov = torch.linalg.pinv(
            cov
        )

        ######################################################
        # Mahalanobis distance
        ######################################################

        distance = torch.sqrt(

            torch.sum(

                (centered @ inv_cov)

                * centered,

                dim=1,

            )

        )

        ######################################################
        # Convert distance -> reliability
        ######################################################

        reliability = 1.0 / (

            1.0 + distance

        )

        ######################################################
        # Normalize
        ######################################################

        reliability = (

            reliability

            - reliability.min()

        ) / (

            reliability.max()

            - reliability.min()

            + self.eps

        )

        return reliability