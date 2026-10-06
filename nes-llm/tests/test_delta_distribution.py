"""
W8 tests — delta-only distribution (W8.1 recipient tool, W8.2
integrity metadata).

Three invariants must hold before a delta can ship instead of a
checkpoint:

1. Round trip: base residuals + delta + key recover the exact message,
   and the reconstructed residuals read the same signs production
   reads (the extraction quantity).
2. Integrity: every silent-corruption class fails loudly — tampered
   values, forged counts, a forged payload length (even one re-signed
   with a fresh hash), a truncated file.
3. Honesty: a delta whose embed changed values the format cannot
   represent refuses to build, and a non-sign scheme refuses to
   decode rather than returning noise.
"""

import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.core.types import EmbeddingConfig  # noqa: E402
from src.delta import (  # noqa: E402
    DeltaIntegrityError,
    UnsupportedStrategyError,
    build_delta,
    canonical_sha256,
    load_delta,
    recover_payload,
    reconstruct_residuals,
    save_delta,
    verify_delta,
)
from src.embedding.intelligent_embedder import (  # noqa: E402
    IntelligentEmbedder,
)

MODEL = "test/delta-model"
FAMILY = "test"
MESSAGE = "W8 delta round trip — base model + delta + key"


def make_residuals(n=4, size=5000, magnitude=0.001):
    """Synthetic residuals matching real NF4 residual magnitude."""
    torch.manual_seed(7)
    return {i: torch.randn(size) * magnitude for i in range(n)}


def re_sign(payload):
    """Recompute the hash after editing metadata — simulates an
    attacker who re-signs what they forged, so only the semantic
    cross-checks can catch them."""
    payload["metadata"]["sha256"] = canonical_sha256(
        payload["metadata"],
        payload["positions"],
        payload["values"],
    )
    return payload


class DeltaTestCase(unittest.TestCase):
    """One embed shared by every test; tamper tests work on copies."""

    @classmethod
    def setUpClass(cls):
        cls.residuals = make_residuals()
        cls.message = MESSAGE
        config = EmbeddingConfig(
            total_payload_bits=2000,
            embedding_strategy="sign",
            model_family=FAMILY,
            num_hidden_layers=len(cls.residuals),
        )
        cls.result = IntelligentEmbedder(config).embed(
            cls.message, cls.residuals
        )
        cls.payload = build_delta(
            cls.residuals,
            cls.result,
            model_id=MODEL,
            family=FAMILY,
        )

    def fresh(self):
        return copy.deepcopy(self.payload)


class TestDeltaRoundTrip(DeltaTestCase):
    def test_round_trip_recovers_message(self):
        message, report = recover_payload(
            self.residuals, self.fresh(), self.result.key
        )
        self.assertEqual(message, self.message)
        self.assertTrue(report["ok"])
        self.assertEqual(
            report["payload_bits_from_header"],
            report["payload_bits"],
        )
        self.assertGreaterEqual(
            report["carrier_count"], report["total_bits"]
        )

    def test_reconstruction_matches_production_signs(self):
        """The extraction quantity — the sign at each carrier — must
        read identically to production's embedded residuals, and
        non-carrier values must be untouched."""

        reconstructed = reconstruct_residuals(
            self.residuals, self.fresh()
        )

        for lid, pos in self.payload["positions"].items():
            rec = reconstructed[lid].flatten()
            prod = self.result.embedded_residuals[lid].flatten()
            clean = self.residuals[lid].flatten()

            self.assertTrue(
                torch.equal((rec[pos] >= 0).int(), (prod[pos] >= 0).int()),
                f"layer {lid}: reconstructed signs differ from production",
            )

            mask = torch.zeros(clean.numel(), dtype=torch.bool)
            mask[pos] = True
            self.assertTrue(
                torch.equal(rec[~mask], clean[~mask]),
                f"layer {lid}: non-carrier values changed",
            )

    def test_report_counts_match_arrays(self):
        report = verify_delta(self.fresh())
        carriers = sum(
            p.numel() for p in self.payload["positions"].values()
        )
        changed = sum(
            int((v != 0).sum()) for v in self.payload["values"].values()
        )
        self.assertEqual(report["carrier_count"], int(carriers))
        self.assertEqual(report["changed_count"], changed)
        self.assertEqual(report["model_id"], MODEL)
        self.assertEqual(report["strategy"], "sign")


class TestDeltaIntegrity(DeltaTestCase):
    """W8.2: silent corruption must fail, not extract as noise."""

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "patch.nesdelta"
            save_delta(self.fresh(), path)
            loaded = load_delta(path)

        for lid in self.payload["positions"]:
            self.assertTrue(
                torch.equal(
                    loaded["positions"][lid],
                    self.payload["positions"][lid],
                )
            )
            self.assertTrue(
                torch.equal(
                    loaded["values"][lid],
                    self.payload["values"][lid],
                )
            )
        self.assertEqual(
            loaded["metadata"]["sha256"],
            self.payload["metadata"]["sha256"],
        )

    def test_tampered_value_fails_hash(self):
        payload = self.fresh()
        lid = next(iter(payload["values"]))
        values = payload["values"][lid]
        nonzero = torch.nonzero(values != 0)
        self.assertTrue(len(nonzero), "fixture delta has no changed values")
        # Negate a changed value: bytes differ (hash must catch it)
        # but the nonzero count does not (so only the hash can).
        values[int(nonzero[0])] = -values[int(nonzero[0])]
        with self.assertRaises(DeltaIntegrityError) as ctx:
            verify_delta(payload)
        self.assertIn("sha256 mismatch", str(ctx.exception))

    def test_forged_carrier_count_fails(self):
        payload = self.fresh()
        payload["metadata"]["carrier_count"] += 1
        with self.assertRaises(DeltaIntegrityError) as ctx:
            verify_delta(payload)
        self.assertIn("carrier count", str(ctx.exception))

    def test_forged_payload_length_fails_consistency(self):
        payload = self.fresh()
        payload["metadata"]["payload_bits"] -= 8
        with self.assertRaises(DeltaIntegrityError) as ctx:
            verify_delta(payload)
        self.assertIn("payload length inconsistent", str(ctx.exception))

    def test_resigned_payload_length_fails_at_header(self):
        """A re-signed lie about the payload length must still fail:
        the header the recipient decodes disagrees with metadata."""

        payload = self.fresh()
        payload["metadata"]["payload_bits"] -= 8
        payload["metadata"]["total_bits"] -= 8
        re_sign(payload)
        self.assertTrue(verify_delta(payload)["ok"])  # passes the hash…

        with self.assertRaises(DeltaIntegrityError) as ctx:
            recover_payload(self.residuals, payload, self.result.key)
        self.assertIn("payload length mismatch", str(ctx.exception))

    def test_truncated_file_fails_to_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "patch.nesdelta"
            save_delta(self.fresh(), path)
            with open(path, "r+b") as f:
                f.truncate(128)
            with self.assertRaises(Exception):
                load_delta(path)

    def test_wrong_key_fails_loudly(self):
        with self.assertRaises(DeltaIntegrityError):
            recover_payload(self.residuals, self.fresh(), os.urandom(32))


class TestDeltaHonesty(DeltaTestCase):
    """Refuse rather than produce something that reads as noise."""

    def test_non_sign_strategy_refused(self):
        payload = self.fresh()
        payload["metadata"]["strategy"] = "lwe"
        re_sign(payload)

        with self.assertRaises(UnsupportedStrategyError):
            recover_payload(self.residuals, payload, self.result.key)

    def test_build_rejects_change_outside_carriers(self):
        """The format can only represent changes at recorded carriers;
        a change anywhere else must abort, never be dropped."""

        bad = copy.deepcopy(self.result)
        lid = next(
            l for l, idx in self.result.carrier_indices.items() if idx
        )
        used = set(self.result.carrier_indices[lid])
        free = next(
            i
            for i in range(self.residuals[lid].numel())
            if i not in used
        )
        bad.embedded_residuals[lid].view(-1)[free] += 0.5

        with self.assertRaises(ValueError) as ctx:
            build_delta(
                self.residuals, bad, model_id=MODEL, family=FAMILY
            )
        self.assertIn("outside the recorded carriers", str(ctx.exception))

    def test_build_rejects_truncated_payload(self):
        """A payload that did not fit its carrier budget would extract
        as noise — refuse to ship it."""

        bad = SimpleNamespace(
            success=True,
            bits_embedded=5,
            total_bits=10,
            carrier_indices={},
            embedded_residuals={},
            strategy="sign",
            key_id="deadbeef",
        )
        with self.assertRaises(ValueError) as ctx:
            build_delta(
                self.residuals, bad, model_id=MODEL, family=FAMILY
            )
        self.assertIn("did not fit", str(ctx.exception))


class TestGradTensors(unittest.TestCase):
    """Live model-derived residuals carry requires_grad=True (the
    recipient CLI extracts without a cache). The delta is a
    distribution artifact, not a graph — it must build, hash and
    recover from grad-carrying inputs."""

    def test_requires_grad_residuals_round_trip(self):
        torch.manual_seed(3)
        residuals = {
            i: (torch.randn(3000) * 0.001).requires_grad_(True)
            for i in range(2)
        }
        message = "grad-safe delta"
        config = EmbeddingConfig(
            total_payload_bits=1000,
            embedding_strategy="sign",
            num_hidden_layers=2,
        )
        result = IntelligentEmbedder(config).embed(message, residuals)
        payload = build_delta(
            residuals, result, model_id=MODEL, family=FAMILY
        )
        self.assertTrue(verify_delta(payload)["ok"])

        recovered, report = recover_payload(
            residuals, payload, result.key
        )
        self.assertEqual(recovered, message)
        self.assertTrue(report["ok"])


if __name__ == "__main__":
    unittest.main()
