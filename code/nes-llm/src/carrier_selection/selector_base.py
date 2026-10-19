from abc import ABC, abstractmethod


class CarrierSelector(ABC):
    """
    Base class for all carrier selection algorithms.

    A carrier selector determines which residual positions
    will be used for embedding.
    """

    @abstractmethod
    def select(
        self,
        residual_tensor,
        num_bits,
        secret_key,
    ):
        """
        Select carrier positions for embedding.

        Parameters
        ----------
        residual_tensor : torch.Tensor
            Residual tensor.

        num_bits : int
            Number of payload bits.

        secret_key : str
            Secret key controlling deterministic selection.

        Returns
        -------
        list[int]
            Carrier positions.
        """
        pass