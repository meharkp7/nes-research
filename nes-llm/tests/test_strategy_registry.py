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


    def test_split_round_trips_without_cover(self):
        """W5.3: the dial's mixture must round trip like the rest.

        Parity and sign are delegated on disjoint carriers; if the
        partition, slice boundary, or reassembly order drifted from
        what the extractor replays, only a full round trip notices.
        """
        before = {
            lid: t.clone() for lid, t in self.residuals.items()
        }
        for fraction in (0.0, 0.5, 1.0):
            with self.subTest(fraction=fraction):
                config = self._config("split")
                config.split_fraction = fraction
                strategy = build(config, "split")
                result = embed_with(
                    strategy, self.residuals, self.bits, self.carriers
                )
                self.assertIsInstance(result, EmbeddingResult)
                self.assertGreater(result.bits_embedded, 0)

                recovered = extract_with(
                    strategy,
                    result.embedded_weights,
                    result.carrier_indices,
                )
                # Carriers run out before the supplied bits (fixture
                # supplies more bits than positions), so compare
                # against the bits actually embedded.
                self.assertEqual(
                    result.bits_embedded, len(recovered)
                )
                self.assertEqual(
                    recovered, self.bits[: len(recovered)]
                )

        # The strategy must not mutate the cover it was handed.
        for lid, original in before.items():
            self.assertTrue(
                torch.equal(self.residuals[lid], original),
                f"layer {lid} mutated",
            )

    def test_split_partition_and_spec(self):
        """W5.3: endpoints are pure mechanisms; the spec is honest."""
        config = self._config("split")

        config.split_fraction = 0.0
        pure_sign = build(config, "split")
        parity, sign = pure_sign.partition(self.carriers)
        self.assertEqual(parity, {})
        self.assertTrue(sign)

        config.split_fraction = 1.0
        pure_parity = build(config, "split")
        parity, sign = pure_parity.partition(self.carriers)
        self.assertTrue(parity)
        self.assertEqual(sign, {})

        # Both endpoints must decode through their own mechanism.
        for strategy in (pure_sign, pure_parity):
            result = embed_with(
                strategy, self.residuals, self.bits, self.carriers
            )
            recovered = extract_with(
                strategy,
                result.embedded_weights,
                result.carrier_indices,
            )
            self.assertEqual(recovered, self.bits[: len(recovered)])

        s = spec("split")
        self.assertEqual(s.status, "READY")
        self.assertFalse(s.extract_needs_cover)
        self.assertFalse(s.needs_trained_model)
        # Sign carriers exist for the default fraction (0.5); the
        # fraction-dependence lives in the notes.
        self.assertTrue(s.forces_sign_flip)
        self.assertIn("fraction-dependent", s.notes)

        # A fraction outside [0, 1] must fail loudly at build time.
        config.split_fraction = 1.5
        with self.assertRaises(ValueError):
            build(config, "split")

    def test_lwe_width_rule_global_default(self):
        """W5.4: the default rule is byte-compatible with exp10/11/12.

        Whatever the layer's noise, a default-config strategy derives
        the shipped absolute width — the number exp11's frontier was
        measured at. A silent per-layer default would move every
        historical artifact's meaning.
        """
        from src.embedding.strategies.lwe_strategy import (
            DEFAULT_GRID_WIDTH,
        )

        strategy = build(self._config("lwe"), "lwe")
        self.assertEqual(strategy.width_rule, "global")
        for lid, std in ((0, 0.00114), (1, 0.00241), (2, 0.00262)):
            self.assertEqual(
                strategy._derive_interval_width(lid, std),
                DEFAULT_GRID_WIDTH,
            )

    def test_lwe_width_rule_per_layer(self):
        """W5.4: width follows each layer's noise, clipped to the
        exp11 window, and the round trip still holds — which is also
        the embed/extract agreement test, because embed derives from
        the original std and extract from the stego std."""
        from src.embedding.strategies.lwe_strategy import (
            PER_LAYER_WIDTH_CAP,
            PER_LAYER_WIDTH_FLOOR,
            PER_LAYER_WIDTH_SCALE,
        )

        config = self._config("lwe")
        config.lwe_width_rule = "per_layer"
        strategy = build(config, "lwe")
        self.assertEqual(strategy.width_rule, "per_layer")

        w_quiet = strategy._derive_interval_width(0, 0.00114)
        w_median = strategy._derive_interval_width(1, 0.00241)
        w_noisy = strategy._derive_interval_width(2, 0.00262)
        w_loud = strategy._derive_interval_width(3, 0.00900)

        # Proportional where inside the window, clipped at both ends.
        # (std is coarsened to 4 decimals first: round(0.00241, 4)
        # = 0.0024 — the bucketing is the mechanism, not an accident.)
        self.assertEqual(w_quiet, PER_LAYER_WIDTH_FLOOR)
        self.assertAlmostEqual(
            w_median, PER_LAYER_WIDTH_SCALE * round(0.00241, 4),
            places=12,
        )
        self.assertAlmostEqual(
            w_noisy, PER_LAYER_WIDTH_SCALE * round(0.00262, 4),
            places=12,
        )
        self.assertEqual(w_loud, PER_LAYER_WIDTH_CAP)
        for w in (w_quiet, w_median, w_noisy, w_loud):
            self.assertGreaterEqual(w, PER_LAYER_WIDTH_FLOOR)
            self.assertLessEqual(w, PER_LAYER_WIDTH_CAP)

        # A different strategy instance (the extractor's) must derive
        # the same widths and decode the embedder's stream.
        result = embed_with(
            strategy, self.residuals, self.bits, self.carriers
        )
        reader_config = self._config("lwe")
        reader_config.lwe_width_rule = "per_layer"
        reader = build(reader_config, "lwe")
        recovered = extract_with(
            reader, result.embedded_weights, result.carrier_indices
        )
        self.assertEqual(recovered, self.bits[: len(recovered)])

    def test_lwe_per_layer_width_bucket_stability(self):
        """W5.4: embed and extract derive the width from different
        views of the same layer (original vs stego std). The measured
        per-layer shift from an LWE embed is <= 0.0153% on
        Qwen2.5-3B; within that band the 4-decimal coarsening must
        bucket identically, or the extractor's grid would drift."""
        config = self._config("lwe")
        config.lwe_width_rule = "per_layer"
        embedder = build(config, "lwe")
        reader = build(config, "lwe")

        for lid, std in enumerate(
            (0.00114, 0.00156, 0.00191, 0.00241, 0.00262)
        ):
            stego_std = std * (1 + 1.53e-4)
            self.assertEqual(
                embedder._derive_interval_width(lid, std),
                reader._derive_interval_width(lid + 100, stego_std),
                f"layer {lid}: width drifted between views",
            )

        # The spec must say the option exists — a rule nobody can
        # discover from the registry is a rule nobody can verify.
        self.assertIn("per_layer", spec("lwe").notes)

    def test_lwe_layer_rank_noise_invariance(self):
        """W5.4 layer_rank: noise moves every layer's std (through
        sqrt(std^2 + sigma^2)) but preserves their ORDER, so both
        sides must derive identical widths — the property per_layer
        lacks, and the whole reason this rule exists."""
        import math

        config = self._config("lwe")
        config.lwe_width_rule = "layer_rank"
        embedder = build(config, "lwe")

        original = {0: 0.00114, 1: 0.00156, 2: 0.00241, 3: 0.00262}
        for sigma in (0.001, 0.002, 0.005):
            noisy = {
                lid: math.sqrt(s * s + sigma * sigma)
                for lid, s in original.items()
            }
            self.assertEqual(
                embedder._rank_widths(original),
                embedder._rank_widths(noisy),
                f"widths diverged at sigma={sigma}",
            )

        # The ladder is fixed constants: quietest layer gets the
        # floor, noisiest the global default, monotone in between.
        widths = embedder._rank_widths(original)
        self.assertAlmostEqual(widths[0], 0.005, places=12)
        self.assertAlmostEqual(widths[3], 0.010, places=12)
        self.assertLess(widths[0], widths[1])
        self.assertLess(widths[1], widths[2])
        self.assertLess(widths[2], widths[3])

        # An unknown rule must fail loudly at build time, not fall
        # through to some default grid.
        config.lwe_width_rule = "mystery"
        with self.assertRaises(ValueError):
            build(config, "lwe")

        # And the round trip must hold through the registry path.
        config.lwe_width_rule = "layer_rank"
        strategy = build(config, "lwe")
        result = embed_with(
            strategy, self.residuals, self.bits, self.carriers
        )
        reader = build(config, "lwe")
        recovered = extract_with(
            reader, result.embedded_weights, result.carrier_indices
        )
        self.assertEqual(recovered, self.bits[: len(recovered)])


if __name__ == "__main__":
    unittest.main(verbosity=2)