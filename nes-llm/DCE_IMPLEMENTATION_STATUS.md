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

## Repeated-seed weight sweep: negative result for current objective

The exploratory sweep compared the same nearest-feasible baseline against DCE
for five seeds (20261009–20261013), 2,000 carriers per run, and distribution
weights 0, 0.1, 0.5, 1, 2, 5, and 10. The user-provided terminal excerpt
contains the summaries for weights 0.5, 1, 2, 5, and 10; the excerpt does not
include the summaries for weights 0 and 0.1.

For the five visible weights, the mean paired differences (DCE minus baseline)
were:

| Distribution weight | Δ mean squared perturbation | Δ histogram TV | Δ cover-to-embedded histogram KL (nats) |
|---:|---:|---:|---:|
| 0.5 | +0.041679 | +0.073205 | +0.055425 |
| 1.0 | +0.110604 | +0.151870 | +0.189192 |
| 2.0 | +0.251897 | +0.281189 | +0.550910 |
| 5.0 | +0.507106 | +0.474271 | +1.322033 |
| 10.0 | +0.646421 | +0.555017 | +1.731373 |

Lower is better for all three reported metrics. The visible results show DCE
losing to nearest-feasible selection on perturbation and both distribution
metrics at every displayed weight. BER is zero for both methods by construction
of the toy candidate set and does not differentiate them. Increasing the
distribution weight makes the observed losses larger, rather than fixing them.

This is an exploratory synthetic result, not a model-level or NF4 result. Do
not select a weight based on this sweep, claim DCE is stealthier, or wire this
configuration into the production registry. The result supports stopping the
current independent per-carrier objective.

## Decision and next research gate

**Decision: do not continue tuning the current per-carrier distribution proxy.**
It scores individual candidates using cover-bin frequency, but the measured
outcome is a property of the complete embedded batch. Selecting individually
common bins can over-concentrate the aggregate histogram and increase both
distortion and distribution mismatch.

If DCE remains worth pursuing, the next design should make the batch-level
constraint explicit—for example, choose payload-feasible candidates while
tracking aggregate histogram counts against the cover histogram, and compare
against the same nearest-feasible baseline under identical inputs. The new
objective must have tests for its aggregate accounting and report every seed
and parameter. If a batch-level approach cannot improve the baseline without
unacceptable perturbation, stop DCE and prioritize the better-supported
embedding direction.

Only after a new synthetic method passes a declared validation gate should a
representation-specific NF4 candidate generator/decoder and controlled
Qwen2.5-3B pilot be considered. Even improved synthetic metrics would not
establish NF4 compatibility, model utility preservation, undetectability, or
checkpoint robustness.
