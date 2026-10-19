import torch

class AdaptiveEmbedder:

    @staticmethod
    def embed_bits(
        residual,
        bits,
        positions,
        alpha=0.05
    ):

        modified = residual.clone()

        flat = modified.view(-1)

        for bit, pos in zip(bits, positions):

            mag = abs(flat[pos])

            delta = alpha * mag

            if delta == 0:
                delta = alpha * 1e-5

            if bit == 1:
                flat[pos] += delta
            else:
                flat[pos] -= delta

        return modified