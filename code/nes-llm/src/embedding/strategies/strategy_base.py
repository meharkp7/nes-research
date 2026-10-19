from abc import ABC, abstractmethod


class EmbeddingStrategy(ABC):
    """
    Base interface for all embedding algorithms.

    Every embedding strategy in NES must implement
    the same API so that the ResidualEmbedder can
    use them interchangeably.
    """

    @abstractmethod
    def embed(
        self,
        residual_tensor,
        bits,
        positions,
    ):
        """
        Embed a sequence of bits into the supplied
        residual tensor using the selected carrier
        positions.

        Returns
        -------
        torch.Tensor
            Embedded residual tensor.
        """
        pass

    @abstractmethod
    def extract(
        self,
        embedded_tensor,
        positions,
    ):
        """
        Recover embedded bits from the supplied
        residual tensor.

        Returns
        -------
        list[int]
            Extracted bit sequence.
        """
        pass