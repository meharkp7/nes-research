from src.carrier_selection.index_sampler import IndexSampler


class KeyedResidualEmbedder:

    EPSILON = 0.001

    @staticmethod
    def embed_bits(
        residual_tensor,
        bits,
        secret_key,
    ):

        flat = residual_tensor.clone().flatten()

        positions = (
            IndexSampler.sample_positions(
                secret_key,
                len(flat),
                len(bits),
            )
        )

        for bit, pos in zip(bits, positions):

            value = max(
                abs(flat[pos].item()),
                KeyedResidualEmbedder.EPSILON,
            )

            if bit == 1:
                flat[pos] = value
            else:
                flat[pos] = -value

        return flat

    @staticmethod
    def extract_bits(
        embedded_tensor,
        secret_key,
        num_bits,
    ):

        flat = embedded_tensor.flatten()

        positions = (
            IndexSampler.sample_positions(
                secret_key,
                len(flat),
                num_bits,
            )
        )

        bits = []

        for pos in positions:

            if flat[pos] >= 0:
                bits.append(1)
            else:
                bits.append(0)

        return bits