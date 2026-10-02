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

    def test_lwe_is_flagged_as_needing_the_cover(self):
        # Established by reading extract(): it calls
        # residuals_ref[layer].std() to derive the grid width.
        self.assertTrue(spec("lwe").extract_needs_cover)
        self.assertEqual(spec("lwe").status, "BLOCKED")

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

    def test_lwe_cannot_extract_without_the_cover(self):
        """The defect this suite exists to catch.

        LWE-style encoding is the most promising scheme structurally, but
        its extract() needs residuals_ref to size the grid. This asserts
        the limitation rather than letting it look usable.
        """
        report = probe_extraction(
            "lwe", self.residuals, self.bits, self.carriers
        )

        self.assertTrue(
            report["embed_ok"],
            "LWE embed should work; only extraction is blocked",
        )
        self.assertFalse(report["extract_without_cover_ok"])
        self.assertFalse(report["structurally_usable"])
        self.assertIn("CoverRequired", report["error"])

    def test_extract_with_raises_when_cover_withheld(self):
        strategy = build(self._config("lwe"), "lwe")
        result = embed_with(
            strategy, self.residuals, self.bits, self.carriers
        )

        with self.assertRaises(CoverRequired):
            extract_with(
                strategy,
                result.embedded_weights,
                result.carrier_indices,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)