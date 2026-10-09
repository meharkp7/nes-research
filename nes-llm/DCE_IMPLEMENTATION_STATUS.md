# Distribution-Constrained Embedding (DCE): implementation status

## Status: isolated prototype implemented; not a validated embedding strategy

This milestone implements the reusable candidate-search primitives proposed in
the NES V2 plan:

- src/optimization/candidate_generator.py: enumerate allowed carrier values
  and evaluate decoding before and after a caller-supplied quantizer.
- src/optimization/cost_function.py: explicit weighted objective with
  separate payload error, post-quantization error, distribution-cost proxy,
  and squared perturbation terms.
- src/optimization/optimizer.py: deterministic minimum-cost selection and
  machine-readable component breakdown.
- tests/test_distribution_constrained_optimizer.py: synthetic tests of the
  optimization contract.
- scripts/dce_candidate_benchmark.py: a controlled comparison of DCE against
  nearest-feasible candidate selection on identical seeded synthetic data.
- tests/test_dce_candidate_benchmark.py: deterministic benchmark and metrics tests.

The code is intentionally **not wired into the production strategy registry**
yet. It is a research prototype, not evidence that DCE beats sign, LWE-inspired
parity/grid, QACI, or any other baseline.

## First local validation gate

From nes-llm/:

```bash
../.venv/bin/python -m unittest discover -s tests -p 'test_distribution_constrained_optimizer.py' -v
../.venv/bin/python -m unittest discover -s tests -p 'test_dce_candidate_benchmark.py' -v
../.venv/bin/python scripts/dce_candidate_benchmark.py --carriers 2000 --seed 20261009
```

The benchmark compares the nearest feasible candidate against the DCE weighted
objective using the same cover values, payload bits, candidate sets and toy
quantizer. It reports post-quantization BER, mean squared perturbation,
changed-carrier fraction, histogram total variation distance, and smoothed
empirical KL in both directions. All parameters and weights are recorded.

## Important limits

1. The benchmark's uniform scalar quantizer is a toy mechanism, not
   BitsAndBytes NF4. Candidate values are code centers, so zero BER is expected
   by construction when feasible candidates are available.
2. The candidate-level distribution term uses a smoothed empirical histogram
   of the synthetic cover as a proxy. The benchmark separately measures global
   distribution changes after embedding; neither metric is a trained detector.
3. The optimizer chooses each carrier independently. It does not solve a
   globally coupled allocation or guarantee a global optimum.
4. A model-level DCE claim requires a matched payload, fixed receiver contract,
   appropriate controls, utility evaluation, independent steganalysis, and
   transformation-specific recovery measurements.
5. No existing B1.4/B1.5 files or prior experiment artifacts are changed by
   this prototype.

## First benchmark result (seed 20261009; 2,000 carriers)

The initial single-seed result did **not** support the current DCE objective:

| Metric | Nearest-feasible baseline | DCE | DCE minus baseline |
|---|---:|---:|---:|
| BER after toy quantization | 0.000000 | 0.000000 | 0.000000 |
| Mean squared perturbation | 0.021024 | 0.064852 | +0.043828 |
| Histogram TV distance | 0.036264 | 0.111773 | +0.075509 |
| Cover-to-embedded histogram KL (nats) | 0.006574 | 0.068248 | +0.061674 |

DCE's mean squared perturbation was about 3.08 times the baseline, and its
histogram TV distance was about 3.08 times the baseline. Its cover-to-embedded
KL was about 10.38 times the baseline. Zero BER is expected from the toy
candidate construction and is not a differentiating result. This is a negative
result for this configuration, not evidence against every possible DCE design.

Interpretation: the current per-candidate distribution proxy does not control
the aggregate histogram well enough. Do not promote this configuration or
choose a replacement weight from this single seed.

## Next validation gate: repeated seeds and weight sweep

Added `scripts/dce_weight_sweep.py` to compare the same baseline against DCE
across declared seeds and distribution weights. It reports per-run paired
deltas, means, population standard deviations, ranges, and whether each metric
beats baseline on every seed. The default exploratory grid is five seeds and
weights 0, 0.1, 0.5, 1, 2, 5, and 10.

From nes-llm/:

```bash
../.venv/bin/python -m unittest discover -s tests -p 'test_dce_weight_sweep.py' -v
../.venv/bin/python scripts/dce_weight_sweep.py --carriers 2000 --output ../cache/dce_weight_sweep_20261009.json
```

This sweep is exploratory, not confirmatory: report all weights and seeds, and
do not describe the best observed setting as validated without a separately
declared held-out evaluation. Even a better synthetic sweep would not establish
NF4 compatibility, model utility preservation, undetectability, or checkpoint
robustness.

After reviewing the sweep, decide whether to redesign the objective around
aggregate batch-level distribution constraints or stop this DCE variant. Only
then proceed to a representation-specific NF4 candidate generator/decoder and
a controlled Qwen2.5-3B pilot.
