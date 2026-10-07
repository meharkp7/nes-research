"""
Tests for the W8 paper-hardening experiments (exp25/26/27).

Everything here is pure logic — policy index construction, gate
evaluation, partial-access stream arithmetic, and the SVG figure — so
it pins without loading a model or a residual cache. The experiments
spend their model load on measurements, not on re-verifying logic the
suite can check in milliseconds.

Each test also pins a failure mode that would otherwise produce a
plausible-looking artifact that measured nothing:
  * a policy that silently ignores the shared allocation,
  * a gate that passes a failing point,
  * partial access that reports coverage against the wrong stream
    order,
  * a figure that is not well-formed XML.
"""

import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.carrier_intelligence.qaci_pipeline import (  # noqa: E402
    CarrierSelectionResult,
)
from src.carrier_intelligence.selector import CarrierSelector  # noqa: E402
from src.experiments.exp25_selection_ablation import (  # noqa: E402
    _PolicySelect,
    _ber,
    allocation_digest,
    build_policy_indices,
    evaluate_gate as exp25_gate,
)
from src.experiments.exp26_capacity_scaling import (  # noqa: E402
    evaluate_gate as exp26_gate,
    message_for,
    write_svg,
)
from src.experiments.exp27_threat_model_boundary import (  # noqa: E402
    partial_layer_sets,
    stream_slices,
    wrong_keys,
)


def _residuals(n_layers=3, n=128):
    torch.manual_seed(0)
    return {
        lid: torch.randn(n) * 0.02 for lid in range(n_layers)
    }


def _allocation(residuals, k=8):
    return {lid: k for lid in residuals}


class PolicyIndexTests(unittest.TestCase):
    def test_all_policies_respect_the_shared_allocation(self):
        """The ablation's whole control variable: every policy must
        pick EXACTLY the allocated count per layer — no more, no
        fewer, no silent drops."""
        residuals = _residuals()
        alloc = _allocation(residuals, k=8)
        variants = {
            "magnitude": {},
            "random": {"seed": 7},
            "keyed": {"key": "k1"},
        }
        for policy, kwargs in variants.items():
            idx = build_policy_indices(policy, residuals, alloc, **kwargs)
            self.assertEqual(sorted(idx), sorted(alloc))
            for lid, want in alloc.items():
                got = idx[lid]
                self.assertEqual(len(got), want, policy)
                self.assertEqual(len(set(got)), want,
                                 f"{policy}: duplicate positions")
                self.assertTrue(
                    all(0 <= p < residuals[lid].numel() for p in got),
                    f"{policy}: position outside tensor",
                )

    def test_magnitude_equals_the_public_rule(self):
        """The 'publicly re-derivable' claim rests on this arm being
        bit-identical to CarrierSelector.select_by_magnitude — the
        very function the production fallback and the keyless
        attacker re-run."""
        residuals = _residuals()
        alloc = _allocation(residuals, k=8)
        idx = build_policy_indices("magnitude", residuals, alloc)
        for lid, k in alloc.items():
            want = CarrierSelector().select_by_magnitude(
                residuals[lid].flatten(), k
            )
            self.assertEqual(list(idx[lid]), list(want))

    def test_keyed_is_deterministic_per_key_and_differs_across_keys(self):
        """Key-derived positions are a function of the key alone:
        same key -> same carriers (reproducibility), different key ->
        different carriers (the security property being measured)."""
        residuals = _residuals()
        alloc = _allocation(residuals, k=8)
        a1 = build_policy_indices("keyed", residuals, alloc, key="k1")
        a2 = build_policy_indices("keyed", residuals, alloc, key="k1")
        b = build_policy_indices("keyed", residuals, alloc, key="k2")
        self.assertEqual(a1, a2)
        self.assertNotEqual(a1, b)

    def test_random_is_reproducible_per_seed_and_differs_across_seeds(self):
        residuals = _residuals()
        alloc = _allocation(residuals, k=8)
        r1 = build_policy_indices("random", residuals, alloc, seed=1)
        r2 = build_policy_indices("random", residuals, alloc, seed=1)
        r3 = build_policy_indices("random", residuals, alloc, seed=2)
        self.assertEqual(r1, r2)
        self.assertNotEqual(r1, r3)

    def test_keyed_and_random_reject_key_seed_ambiguity(self):
        """Exactly one of key/seed — a silent both-or-neither would
        make an arm's identity ambiguous in the artifact."""
        residuals = _residuals()
        alloc = _allocation(residuals, k=4)
        for policy in ("keyed", "random"):
            with self.assertRaises(ValueError):
                build_policy_indices(policy, residuals, alloc)
            with self.assertRaises(ValueError):
                build_policy_indices(
                    policy, residuals, alloc, key="k", seed=1
                )
        with self.assertRaises(ValueError):
            build_policy_indices("nope", residuals, alloc, seed=1)

    def test_zero_allocation_layer_gets_no_carriers(self):
        residuals = _residuals()
        alloc = {0: 8, 1: 0, 2: 4}
        idx = build_policy_indices("keyed", residuals, alloc, key="k")
        self.assertEqual(idx[1], [])
        self.assertEqual(len(idx[0]), 8)
        self.assertEqual(len(idx[2]), 4)

    def test_allocation_digest_is_order_independent_and_value_sensitive(self):
        """The artifact's claim 'allocation shared across arms' rests
        on this digest: same allocation -> same digest regardless of
        dict order; any changed count -> different digest."""
        a = {0: 8, 1: 4, 2: 12}
        b = {2: 12, 1: 4, 0: 8}
        c = {0: 8, 1: 5, 2: 12}
        self.assertEqual(allocation_digest(a), allocation_digest(b))
        self.assertNotEqual(allocation_digest(a), allocation_digest(c))


class PolicySelectWrapperTests(unittest.TestCase):
    def _result(self, residuals, alloc):
        indices = {
            lid: list(range(k)) for lid, k in alloc.items()
        }
        return CarrierSelectionResult(
            selected_indices=indices,
            layer_allocation=dict(alloc),
            layer_profiles=[],
            quality_scores={},
            total_selected=sum(alloc.values()),
        )

    def test_magnitude_arm_passes_production_result_through_untouched(self):
        """The magnitude arm IS production: the wrapper must not swap
        anything, or the ablation would compare a modified arm
        against itself."""
        residuals = _residuals()
        alloc = _allocation(residuals, k=8)
        base = self._result(residuals, alloc)
        calls = []

        def original(**kwargs):
            calls.append(kwargs)
            return base

        wrapped = _PolicySelect(original, "magnitude")
        out = wrapped.select(residuals=residuals, total_payload_bits=8)
        self.assertIs(out, base)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["total_payload_bits"], 8)

    def test_random_arm_swaps_positions_but_keeps_allocation(self):
        residuals = _residuals()
        alloc = _allocation(residuals, k=8)
        base = self._result(residuals, alloc)

        def original(**kwargs):
            return base

        wrapped = _PolicySelect(original, "random", seed=5)
        alloc_before = dict(base.layer_allocation)
        out = wrapped.select(residuals=residuals, total_payload_bits=8)
        # the wrapper swaps positions on the production result in
        # place (the embedder discards the original anyway) — what
        # must hold is: allocation untouched, positions swapped.
        self.assertEqual(out.layer_allocation, alloc_before)
        want = build_policy_indices(
            "random", residuals, alloc, seed=5
        )
        self.assertEqual(out.selected_indices, want)
        self.assertEqual(
            out.total_selected,
            sum(len(v) for v in want.values()),
        )


class BerTests(unittest.TestCase):
    def test_ber_counts_errors_over_the_compared_prefix(self):
        ber, errors, n = _ber([0, 1, 1, 0], [0, 1, 0, 0])
        self.assertEqual((ber, errors, n), (0.25, 1, 4))

    def test_ber_handles_truncation_and_empty(self):
        ber, errors, n = _ber([0, 1, 1], [0, 1])
        self.assertEqual((ber, errors, n), (0.0, 0, 2))
        self.assertEqual(_ber([], []), (None, 0, 0))


class Exp26GateTests(unittest.TestCase):
    def _point(self, size, ber=0.0, acc=0.50, kl=0.001):
        return {
            "payload_bits": size,
            "ber": ber,
            "bit_errors": 0 if ber == 0.0 else 10,
            "bits_compared": size,
            "detector_accuracy": acc,
            "kl_divergence": kl,
        }

    def test_gate_passes_all_green_points(self):
        points = [self._point(s) for s in (1000, 10000, 50000)]
        gate = {"max_ber": 0.0, "max_detector_accuracy": 0.55,
                "max_kl_divergence": 0.05}
        status, failures = exp26_gate(points, gate)
        self.assertEqual(status, "PASS")
        self.assertEqual(failures, [])

    def test_gate_fails_and_names_the_offending_size(self):
        """A failed size must stay in the artifact AND be named in
        the failure — dropping or anonymizing it would erase the
        measured limit."""
        points = [self._point(1000), self._point(10000, ber=0.01),
                  self._point(50000, acc=0.57, kl=0.06)]
        gate = {"max_ber": 0.0, "max_detector_accuracy": 0.55,
                "max_kl_divergence": 0.05}
        status, failures = exp26_gate(points, gate)
        self.assertEqual(status, "FAIL")
        self.assertEqual(len(failures), 3)
        self.assertTrue(any("10000 bits" in f for f in failures))
        self.assertTrue(any("50000 bits" in f for f in failures))

    def test_message_for_covers_the_payload(self):
        self.assertEqual(len(message_for(1000)) * 8 >= 1000, True)
        self.assertTrue(message_for(1000).startswith("A"))


class Exp25GateTests(unittest.TestCase):
    def _arms(self):
        def rep(**over):
            base = {
                "replicate": 0,
                "ber": 0.0,
                "detector_accuracy": 0.50,
                "kl_divergence": 0.001,
                "robustness_ber_sigma_0_001": 0.01,
            }
            base.update(over)
            return base

        def reps(n):
            return [rep(replicate=i) for i in range(n)]

        return {
            "random": {"replicates": reps(3)},
            "magnitude": {"replicates": reps(1)},
            "keyed": {"replicates": reps(3)},
        }

    GATE = {
        "max_ber_all_arms": 0.0,
        "max_detector_accuracy": 0.55,
        "max_kl_divergence": 0.05,
        "max_production_ber_at_sigma_0_001": 0.02,
    }

    def test_gate_passes_when_every_arm_round_trips(self):
        status, failures = exp25_gate(self._arms(), self.GATE)
        self.assertEqual(status, "PASS")
        self.assertEqual(failures, [])

    def test_gate_fails_on_any_arm_ber(self):
        """H1 applies to EVERY arm: a baseline that cannot carry the
        payload invalidates the comparison, not just its own row."""
        arms = self._arms()
        arms["keyed"]["replicates"][2]["ber"] = 0.03
        status, failures = exp25_gate(arms, self.GATE)
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("keyed[2]" in f for f in failures))

    def test_gate_gates_the_production_arm_on_detectability(self):
        arms = self._arms()
        arms["magnitude"]["replicates"][0]["detector_accuracy"] = 0.61
        status, failures = exp25_gate(arms, self.GATE)
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("production" in f for f in failures))

    def test_baseline_detectability_is_flagged_not_gated(self):
        """Documented asymmetry: baseline arms are reported against
        the reference lines but do not gate the artifact's status."""
        arms = self._arms()
        arms["random"]["replicates"][0]["detector_accuracy"] = 0.61
        status, _ = exp25_gate(arms, self.GATE)
        self.assertEqual(status, "PASS")

    def test_gate_fails_on_production_robustness(self):
        arms = self._arms()
        arms["magnitude"]["replicates"][0][
            "robustness_ber_sigma_0_001"
        ] = 0.05
        status, failures = exp25_gate(arms, self.GATE)
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("robustness" in f for f in failures))


class Exp27HelpersTests(unittest.TestCase):
    def test_wrong_keys_are_distinct_and_deterministic(self):
        keys = wrong_keys()
        self.assertEqual(len(keys), 10)
        self.assertEqual(len(set(keys)), 10)
        self.assertEqual(keys, wrong_keys())
        self.assertTrue(all(len(k) == 32 for k in keys))
        # must never equal a real production key by construction —
        # they are sha256 labels, not RNG draws that could collide
        # with the embed's key generation.
        self.assertEqual(len({k[:4] for k in keys}), 10)

    def test_partial_layer_sets_counts_sortedness_and_prefix(self):
        layers = list(range(36))
        for fraction, want_k in ((0.5, 18), (0.25, 9), (0.1, 3)):
            sets = partial_layer_sets(layers, fraction)
            self.assertEqual(
                sorted(sets), [f"prefix_{int(fraction*100)}pct",
                               f"scattered_{int(fraction*100)}pct"]
            )
            prefix = sets[f"prefix_{int(fraction*100)}pct"]
            scattered = sets[f"scattered_{int(fraction*100)}pct"]
            self.assertEqual(prefix, layers[:want_k])
            self.assertEqual(len(scattered), want_k)
            self.assertEqual(scattered, sorted(scattered))
            self.assertEqual(len(set(scattered)), want_k)
        # deterministic across calls
        self.assertEqual(
            partial_layer_sets(layers, 0.25),
            partial_layer_sets(layers, 0.25),
        )

    def test_stream_slices_reproduce_base_embedder_order(self):
        """Coverage claims are only right if offsets follow the write
        order: layers sorted, each carrying len(carriers[lid]) bits,
        contiguous."""
        carriers = {2: [1, 2], 0: [5], 1: [3, 4, 6]}
        offsets = stream_slices(carriers)
        self.assertEqual(offsets[0], (0, 1))
        self.assertEqual(offsets[1], (1, 4))
        self.assertEqual(offsets[2], (4, 6))
        # total stream is the last end
        self.assertEqual(offsets[2][1], 6)


class SvgFigureTests(unittest.TestCase):
    def test_figure_is_well_formed_xml_with_six_points(self):
        """The committed figure must parse as XML and be generated
        from exactly the points passed — a broken or stale SVG would
        be the one artifact reviewers see without running anything."""
        points = [
            {
                "payload_bits": size,
                "ber": 0.0,
                "detector_accuracy": 0.5028,
                "kl_divergence": 0.00003,
                "mean_abs_delta": 4.7e-2,
            }
            for size in (1000, 2500, 5000, 10000, 20000, 50000)
        ]
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "fig.svg"
            write_svg(points, "Qwen/Qwen2.5-3B", target)
            text = target.read_text(encoding="utf-8")
            root = ET.fromstring(text)  # raises if malformed
            self.assertTrue(root.tag.endswith("svg"))
            # four panels, six markers each -> 24 circles
            circles = [
                el for el in root.iter()
                if el.tag.endswith("circle")
            ]
            self.assertEqual(len(circles), 24)
            for size in ("1k", "2.5k", "50k"):
                self.assertIn(size, text)


if __name__ == "__main__":
    unittest.main()
