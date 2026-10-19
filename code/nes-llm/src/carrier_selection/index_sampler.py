import hashlib
import random


class IndexSampler:

    @staticmethod
    def sample_positions(
        secret_key,
        capacity,
        num_positions
    ):

        seed = int(
            hashlib.sha256(
                secret_key.encode()
            ).hexdigest(),
            16
        )

        rng = random.Random(seed)

        return rng.sample(
            range(capacity),
            num_positions
        )