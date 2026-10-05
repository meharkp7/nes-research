"""
Tests for the embedding strategy registry.

The important test is the round trip WITHOUT the cover. A steganographic
extractor only ever holds the stego weights; a strategy whose extraction
needs the original residuals cannot be used for hiding, however good its
security claim looks. That failure is silent unless it is probed, so it
is asserted here.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.core.types import EmbeddingConfig, EmbeddingResult  # noqa: E402
from src.embedding.strategy_registry import (  # noqa: E402
    DEFAULT_STRATEGY,
    REGISTRY,
    available,
    build,
    embed_with,
    extract_with,
    probe_extraction,
    spec,
    structural_report,
    CoverRequired,
)


def synthetic_residuals(layers: int = 4, size: int = 4096, seed: int = 0):
    """Deterministic pseudo-residuals shaped like real layer residuals.

    Real NF4 residuals are roughly centred and small, so the synthetic
    version is scaled the same way. Values are float32.
    """
    generator = torch.Generator().manual_seed(seed)
    residuals = {}
    for layer_id in range(layers):
        residuals[layer_id] = (
            torch.randn(
                size, generator=generator, dtype=torch.float32
            )
            * 0.002
        )
    return residuals


def carriers_for(residuals, per_layer: int = 16):
    return {
        layer_id: list(range(per_layer))
        for layer_id in residuals
    }


class RegistryTests(unittest.TestCase):
    def test_registry_is_non_empty(self):
        self.assertIn("sign", REGISTRY)
        self.assertGreaterEqual(len(REGISTRY), 4)

    def test_sign_is_the_default_and_marked_ready(self):
        self.assertEqual(DEFAULT_STRATEGY, "sign")
        self.assertEqual(spec("sign").status, "READY")

    def test_unknown_strategy_raises(self):
        with self.assertRaises(KeyError):
            spec("does_not_exist")

    def test_every_spec_has_notes(self):
        # A strategy registered without a justification is how a blocked
        # scheme sneaks into a run.
        for name, s in REGISTRY.items():
            self.assertTrue(
                s.notes.strip(),
                f"{name} has no notes",
            )

    def test_lwe_no_longer_requires_the_cover(self):
        """Phase 3: the grid width is derived from the stego tensor.

        Previously extract() required residuals_ref, which made the
        scheme unusable -- a real extractor holds only stego weights.
        The embedding is sparse enough that the embedded tensor's own std
        equals the cover's.
        """
        self.assertFalse(spec("lwe").extract_needs_cover)
        self.assertEqual(spec("lwe").status, "READY")

        """A parity grid carries no sign information to read.

        Handing it to the sign-based extractor returns noise, so the
        scheme-specific extractor must be selected.
        """
        from src.embedding.extractors import (
            LweParityExtractor,
            SignBasedExtractor,
            extractor_for,
        )

        self.assertIsInstance(extractor_for("lwe", None), LweParityExtractor)
        self.assertIsInstance(
            extractor_for("sign", None), SignBasedExtractor
        )

    def test_sign_flip_flagged_for_sign_and_magnitude_aware(self):
        self.assertTrue(spec("sign").forces_sign_flip)
        self.assertTrue(spec("magnitude_aware").forces_sign_flip)

    def test_structural_report_is_serialisable(self):
        report = structural_report()
        self.assertEqual(len(report), len(REGISTRY))
        for row in report:
            for key in (
                "strategy", "status", "forces_sign_flip",
                "extract_needs_cover",
            ):
                self.assertIn(key, row)


class RoundTripTests(unittest.TestCase):
    """Extraction must work from stego weights alone."""

    def setUp(self):
        self.residuals = synthetic_residuals()
        self.carriers = carriers_for(self.residuals)
        # Layer ids are ints, matching how QACI keys carrier selection.
        per_layer = len(self.carriers[0])
        self.bits = [1, 0, 1, 1, 0, 0, 1, 0] * (per_layer * 4)

    def _config(self, name):
        return EmbeddingConfig(
            total_payload_bits=len(self.bits),
            embedding_strategy=name,
            model_family="qwen",
            num_hidden_layers=len(self.residuals),
        )

    def test_sign_round_trips_without_cover(self):
        strategy = build(self._config("sign"), "sign")
        result = embed_with(
            strategy, self.residuals, self.bits, self.carriers
        )
        self.assertIsInstance(result, EmbeddingResult)

        recovered = extract_with(
            strategy, result.embedded_weights, result.carrier_indices
        )

        # Carriers run out well before the supplied bits, so compare
        # against the bits actually embedded.
        embedded_count = result.bits_embedded
        self.assertEqual(embedded_count, len(recovered))
        self.assertEqual(
            recovered, self.bits[:embedded_count]
        )

    def test_magnitude_aware_round_trips_without_cover(self):
        strategy = build(
            self._config("magnitude_aware"), "magnitude_aware"
        )
        result = embed_with(
            strategy, self.residuals, self.bits, self.carriers
        )
        recovered = extract_with(
            strategy, result.embedded_weights, result.carrier_indices
        )
        embedded_count = result.bits_embedded
        self.assertEqual(embedded_count, len(recovered))
        self.assertEqual(
            recovered, self.bits[:embedded_count]
        )

    def test_extract_with_raises_when_cover_withheld(self):
        """A strategy that genuinely needs the cover must still refuse."""
        strategy = build(self._config("lwe"), "lwe")

        # lwe no longer needs it, so simulate a strategy that does.
        class NeedyStrategy:
            def extract(self, weights, carrier_indices, residuals_ref):
                return []

        with self.assertRaises(CoverRequired):
            extract_with(
                NeedyStrategy(),
                {"0": torch.zeros(4)},
                {0: [0]},
                residuals_ref=None,
            )

    def test_probe_reports_for_every_registered_strategy(self):
        for name in available():
            report = probe_extraction(
                name, self.residuals, self.bits, self.carriers
            )
            # A probe must always produce a verdict, never raise.
            self.assertIn("structurally_usable", report)
            self.assertIn("error", report)

    def test_sign_strategy_is_actually_selectable(self):
        """embedding_strategy must not be silently ignored."""
        from src.embedding.intelligent_embedder import (
            IntelligentEmbedder,
        )

        embedder = IntelligentEmbedder(self._config("magnitude_aware"))
        self.assertEqual(embedder.strategy_name, "magnitude_aware")
        self.assertEqual(
            type(embedder.strategy).__name__, "MagnitudeAwareStrategy"
        )

    def test_default_embedder_is_sign(self):
        from src.embedding.intelligent_embedder import (
            IntelligentEmbedder,
        )

        embedder = IntelligentEmbedder(self._config("sign"))
        self.assertEqual(embedder.strategy_name, "sign")
        self.assertEqual(
            type(embedder.strategy).__name__, "SignEmbeddingStrategy"
        )

    def test_lwe_round_trips_without_cover(self):
        strategy = build(self._config("lwe"), "lwe")
        result = embed_with(
            strategy, self.residuals, self.bits, self.carriers
        )
        recovered = extract_with(
            strategy,
            result.embedded_weights,
            result.carrier_indices,
            residuals_ref=None,
            strategy_name="lwe",
        )
        embedded_count = result.bits_embedded
        self.assertEqual(embedded_count, len(recovered))
        self.assertEqual(recovered, self.bits[:embedded_count])

    def test_lwe_uses_a_parity_extractor_not_a_sign_extractor(self):
        """A parity grid carries no sign information to read.

        Handing it to the sign-based extractor returns noise, so the
        scheme-specific extractor must be selected.
        """
        from src.embedding.extractors import (
            LweParityExtractor,
            SignBasedExtractor,
            extractor_for,
        )

        self.assertIsInstance(
            extractor_for("lwe", None), LweParityExtractor
        )
        self.assertIsInstance(
            extractor_for("sign", None), SignBasedExtractor
        )

    def test_qae_round_trips_without_cover(self):
        """W1.1: the dict adapter must behave like any other strategy.

        ``QaeDictAdapter`` wraps the per-tensor ABC; if its stream
        slicing or layer order drifted from ``BaseEmbedder``'s, the
        bits would decode in the wrong order and only a round trip
        would notice.
        """
        strategy = build(self._config("qae"), "qae")
        before = {
            lid: t.clone() for lid, t in self.residuals.items()
        }
        result = embed_with(
            strategy, self.residuals, self.bits, self.carriers
        )
        self.assertIsInstance(result, EmbeddingResult)

        recovered = extract_with(
            strategy, result.embedded_weights, result.carrier_indices
        )
        embedded_count = result.bits_embedded
        self.assertGreater(embedded_count, 0)
        self.assertEqual(embedded_count, len(recovered))
        self.assertEqual(recovered, self.bits[:embedded_count])

        # The adapter must not mutate the cover it was handed.
        for lid, original in before.items():
            self.assertTrue(
                torch.equal(self.residuals[lid], original),
                f"layer {lid} mutated",
            )

    def test_qae_statuses_and_nf4_qae_fails_loudly(self):
        """W1.1: qae READY with sign-flip marked; nf4_qae BLOCKED.

        A blocked strategy whose factory returned a silent fallback
        would embed with the wrong mechanism and report success — the
        factory must raise instead, with the diagnosis in the message.
        """
        self.assertEqual(spec("qae").status, "READY")
        self.assertTrue(spec("qae").forces_sign_flip)
        self.assertFalse(spec("qae").extract_needs_cover)
        self.assertEqual(spec("nf4_qae").status, "BLOCKED")
        self.assertTrue(spec("nf4_qae").extract_needs_cover)

        with self.assertRaises(RuntimeError) as ctx:
            build(self._config("nf4_qae"), "nf4_qae")
        self.assertIn("BLOCKED", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)