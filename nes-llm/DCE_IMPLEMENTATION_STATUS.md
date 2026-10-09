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

## Next milestone after synthetic validation

1. Inspect whether DCE improves the distribution/distortion trade-off against
   the nearest-feasible baseline; do not tune weights on a single reported seed.
2. Add repeated seeds and weight sweeps with a declared selection protocol.
3. Implement a representation-specific candidate generator/decoder against the
   actual NF4 packed representation and explicit artifact-only receiver contract.
4. Only after the mechanism is correct should it be registered as an experimental
   strategy and run as a controlled Qwen2.5-3B pilot.
