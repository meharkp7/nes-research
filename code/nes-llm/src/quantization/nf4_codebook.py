import torch

from bitsandbytes.functional import (
    quantize_4bit,
    dequantize_4bit,
)


class NF4Codebook:
    """
    Interface to the real NF4 quantizer.

    This class exposes the quantization
    geometry required by the
    Quantization-Aware Embedding (QAE)
    algorithm.

    It never performs embedding.

    It only answers:

        • nearest centroid
        • quantize
        • dequantize
        • reconstruction error
    """

    def __init__(
        self,
        blocksize=64,
    ):
        self.blocksize = blocksize

    def quantize(
        self,
        tensor,
    ):
        """
        Quantize tensor using
        the real NF4 implementation.
        """

        qweight, qstate = quantize_4bit(
            tensor,
            quant_type="nf4",
            blocksize=self.blocksize,
        )

        return qweight, qstate

    def dequantize(
        self,
        qweight,
        qstate,
    ):
        """
        Recover FP values.
        """

        return dequantize_4bit(
            qweight,
            quant_state=qstate,
        )

    def reconstruct(
        self,
        tensor,
    ):
        """
        One complete NF4 cycle.
        """

        qweight, qstate = self.quantize(
            tensor
        )

        return self.dequantize(
            qweight,
            qstate,
        )

    def reconstruction_error(
        self,
        tensor,
    ):
        """
        Reconstruction error
        after NF4 quantization.
        """

        reconstructed = self.reconstruct(
            tensor
        )

        return (
            tensor - reconstructed
        ).abs()

    def stability_score(
        self,
        tensor,
    ):
        """
        Stability score.

        Larger score
        =
        survives NF4 better.
        """

        error = self.reconstruction_error(
            tensor
        )

        return 1.0 / (
            1.0 + error
        )