import torch


class CarrierFeatures:
    """
    Computes carrier features from residual tensors.

    Every method returns one feature value
    per residual element.
    """

    @staticmethod
    def magnitude(
        residual_tensor,
    ):
        """
        Absolute residual magnitude.
        """

        return residual_tensor.abs()

    @staticmethod
    def variance(
        residual_tensor,
    ):
        """
        Placeholder.

        Added in V2.2.
        """

        raise NotImplementedError

    @staticmethod
    def entropy(
        residual_tensor,
    ):
        """
        Placeholder.

        Added in V2.3.
        """

        raise NotImplementedError

    @staticmethod
    def kurtosis(
        residual_tensor,
    ):
        """
        Placeholder.

        Added in V2.3.
        """

        raise NotImplementedError

    @staticmethod
    def layer_prior(
        residual_tensor,
    ):
        """
        Placeholder.

        Added in V2.4.
        """

        raise NotImplementedError