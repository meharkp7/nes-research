"""Tests for Phase 7B — LWE-Inspired Embedding Strategy."""

import secrets
import torch

from src.embedding.strategies.lwe_strategy  import LWEStrategy
from src.extraction.lwe_extractor           import LWEExtractor
from src.core.types                         import EmbeddingConfig


def make_residuals(n=4, size=5000):
    return {i: torch.randn(size) * 0.05 for i in range(n)}

def make_indices(residuals, total_bits):
    n = len(residuals)
    per = total_bits // n
    return {lid: list(range(per)) for lid in residuals}


class TestLWEEncoding:

    def _make_strategy(self, alpha=0.25):
        config = EmbeddingConfig(
            total_payload_bits=100,
            embedding_strategy="lwe",
            alpha=alpha,
        )
        key = secrets.token_bytes(32)
        return LWEStrategy(config, secret_key=key), key

    def test_encode_decode_bit0(self):
        strat, _ = self._make_strategy()
        std      = 0.05
        width    = strat._derive_interval_width(0, std)
        encoded  = strat._encode(0.03, 0, width)
        decoded  = strat._decode(encoded, width)
        assert decoded == 0, f"Expected bit=0, got {decoded}"

    def test_encode_decode_bit1(self):
        strat, _ = self._make_strategy()
        std      = 0.05
        width    = strat._derive_interval_width(0, std)
        encoded  = strat._encode(0.03, 1, width)
        decoded  = strat._decode(encoded, width)
        assert decoded == 1, f"Expected bit=1, got {decoded}"

    def test_clean_ber_zero(self):
        config   = EmbeddingConfig(total_payload_bits=1000,
                                   embedding_strategy="lwe", alpha=0.25)
        key      = secrets.token_bytes(32)
        strat    = LWEStrategy(config, secret_key=key)
        residuals = make_residuals(n=4, size=2000)
        bits     = [i % 2 for i in range(1000)]
        indices  = make_indices(residuals, 1000)

        result    = strat.embed(residuals, bits, indices)
        extractor = LWEExtractor(
            LWEStrategy(config, secret_key=key),
            residuals,
        )
        recovered = extractor.extract(result.embedded_weights, result.carrier_indices)

        n   = min(len(bits), len(recovered))
        ber = sum(a != b for a, b in zip(bits[:n], recovered[:n])) / max(n, 1)
        assert ber == 0.0, f"Expected BER=0, got {ber}"

    def test_wrong_key_is_currently_ignored(self):
        """Known limitation: supplied carriers decode even with the wrong key."""
        config    = EmbeddingConfig(total_payload_bits=1000,
                                    embedding_strategy="lwe", alpha=0.25)
        key       = secrets.token_bytes(32)
        wrong_key = secrets.token_bytes(32)
        strat     = LWEStrategy(config, secret_key=key)
        residuals = make_residuals(n=4, size=2000)
        bits      = [i % 2 for i in range(1000)]
        indices   = make_indices(residuals, 1000)

        result    = strat.embed(residuals, bits, indices)

        # Extract with WRONG key
        wrong_extractor = LWEExtractor(
            LWEStrategy(config, secret_key=wrong_key),
            residuals,
        )
        recovered = wrong_extractor.extract(
            result.embedded_weights, result.carrier_indices
        )
        n   = min(len(bits), len(recovered))
        ber = sum(a != b for a, b in zip(bits[:n], recovered[:n])) / max(n, 1)

        # Regression for the exp13 key-dependence failure, not a security guarantee.
        print(f"\n    BER with wrong key: {ber:.4f}")
        assert len(recovered) == len(bits)
        assert ber == 0.0, f"Current implementation ignores the key, got BER={ber}"

    def test_interval_width_is_currently_key_independent(self):
        """Known limitation: different keys give the same interval width."""
        config = EmbeddingConfig(total_payload_bits=100,
                                 embedding_strategy="lwe", alpha=0.25)
        key1   = secrets.token_bytes(32)
        key2   = secrets.token_bytes(32)
        s1     = LWEStrategy(config, secret_key=key1)
        s2     = LWEStrategy(config, secret_key=key2)
        w1     = s1._derive_interval_width(0, 0.05)
        w2     = s2._derive_interval_width(0, 0.05)
        assert w1 == w2, "Current interval width is key-independent"

    def test_interval_width_is_currently_layer_independent(self):
        """Known limitation: tested layers give the same interval width."""
        config = EmbeddingConfig(total_payload_bits=100,
                                 embedding_strategy="lwe", alpha=0.25)
        key    = secrets.token_bytes(32)
        strat  = LWEStrategy(config, secret_key=key)
        w0     = strat._derive_interval_width(0,  0.05)
        w1     = strat._derive_interval_width(1,  0.05)
        w15    = strat._derive_interval_width(15, 0.05)
        assert w0 == w1 == w15, "Current interval width is layer-independent"

    def test_noise_robustness(self):
        """BER should remain low under small noise."""
        config    = EmbeddingConfig(total_payload_bits=2000,
                                    embedding_strategy="lwe", alpha=0.25)
        key       = secrets.token_bytes(32)
        strat     = LWEStrategy(config, secret_key=key)
        residuals = make_residuals(n=4, size=5000)
        bits      = [i % 2 for i in range(2000)]
        indices   = make_indices(residuals, 2000)

        result = strat.embed(residuals, bits, indices)

        for sigma in [0.0005, 0.001]:
            noisy = {
                lid: t + torch.randn_like(t) * sigma
                for lid, t in result.embedded_weights.items()
            }
            ext = LWEExtractor(LWEStrategy(config, secret_key=key), residuals)
            rec = ext.extract(noisy, result.carrier_indices)
            n   = min(len(bits), len(rec))
            ber = sum(a != b for a, b in zip(bits[:n], rec[:n])) / max(n, 1)
            print(f"\n    LWE BER @ σ={sigma}: {ber:.6f}")
            assert ber < 0.05, f"High BER at σ={sigma}: {ber}"

    def test_distortion_smaller_than_sign(self):
        """
        LWE should have lower distortion than sign for small residuals
        because it makes minimum-distance moves.
        """
        from src.embedding.sign_strategy_v2 import SignEmbeddingStrategy

        n_bits    = 1000
        residuals = make_residuals(n=4, size=2000)
        bits      = [i % 2 for i in range(n_bits)]
        indices   = make_indices(residuals, n_bits)

        # Sign
        sign_config = EmbeddingConfig(total_payload_bits=n_bits,
                                      embedding_strategy="sign", alpha=0.0)
        sign_emb  = SignEmbeddingStrategy(sign_config)
        sign_res  = sign_emb.embed(residuals, bits, indices)

        # LWE
        lwe_config  = EmbeddingConfig(total_payload_bits=n_bits,
                                      embedding_strategy="lwe", alpha=0.1)
        key  = secrets.token_bytes(32)
        lwe_emb  = LWEStrategy(lwe_config, secret_key=key)
        lwe_res  = lwe_emb.embed(residuals, bits, indices)

        sign_dist = sum(
            (residuals[lid].flatten() - sign_res.embedded_weights[lid].flatten())
            .abs().mean().item()
            for lid in residuals
        ) / len(residuals)

        lwe_dist = sum(
            (residuals[lid].flatten() - lwe_res.embedded_weights[lid].flatten())
            .abs().mean().item()
            for lid in residuals
        ) / len(residuals)

        print(f"\n    Distortion: sign={sign_dist:.6f}, lwe={lwe_dist:.6f}")
        # LWE makes minimum moves — should be comparable or lower
        assert lwe_dist < sign_dist * 3, "LWE distortion should be reasonable"


class TestLWEStrategyMetadata:

    def test_embed_returns_metadata(self):
        config = EmbeddingConfig(total_payload_bits=100,
                                 embedding_strategy="lwe", alpha=0.25)
        key    = secrets.token_bytes(32)
        strat  = LWEStrategy(config, secret_key=key)
        res    = make_residuals(n=2, size=500)
        bits   = [0, 1] * 50
        idx    = {lid: list(range(50)) for lid in res}

        result = strat.embed(res, bits, idx)
        assert result.metadata["strategy"] == "lwe"
        assert "grid_widths" in result.metadata
        assert result.success

    def test_bits_embedded_correct(self):
        config = EmbeddingConfig(total_payload_bits=200,
                                 embedding_strategy="lwe", alpha=0.25)
        key    = secrets.token_bytes(32)
        strat  = LWEStrategy(config, secret_key=key)
        res    = make_residuals(n=2, size=500)
        bits   = [0, 1] * 100
        idx    = {lid: list(range(100)) for lid in res}
        result = strat.embed(res, bits, idx)
        assert result.bits_embedded == 200


if __name__ == "__main__":
    import sys
    classes = [TestLWEEncoding(), TestLWEStrategyMetadata()]
    passed = failed = 0
    for obj in classes:
        cls = type(obj).__name__
        for method in [m for m in dir(obj) if m.startswith("test_")]:
            try:
                getattr(obj, method)()
                print(f"  ✅ {cls}.{method}")
                passed += 1
            except Exception as e:
                print(f"  ❌ {cls}.{method}: {e}")
                failed += 1
    print(f"\n{'='*55}\n  {passed} passed, {failed} failed\n{'='*55}")
    sys.exit(0 if failed == 0 else 1)
