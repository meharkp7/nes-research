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

The code is intentionally **not wired into the production strategy registry**
yet. It is a research primitive, not evidence that DCE beats sign, LWE-inspired
parity/grid, QACI, or any other baseline.

## Important limits

1. The quantizer and decoder are supplied by the experiment. The unit tests use
   a toy rounding quantizer; this is not a BitsAndBytes NF4 simulator.
2. The distribution term is a caller-supplied per-candidate proxy. It is not
   a measured batch-level KL divergence. A later batch optimizer must evaluate
   actual distribution statistics on the resulting tensor/artifact.
3. The current optimizer chooses each carrier independently. It does not
   solve a globally coupled allocation or guarantee a global optimum.
4. A model-level DCE claim requires a matched payload, fixed receiver contract,
   appropriate controls, utility evaluation, independent steganalysis, and
   transformation-specific recovery measurements.
5. No existing B1.4/B1.5 files or prior experiment artifacts are changed by
   this prototype.

## First validation gate

From nes-llm/, run:

```bash
../.venv/bin/python -m unittest discover -s tests -p 'test_distribution_constrained_optimizer.py' -v
```

Passing these tests validates only candidate generation and objective selection.
It does not validate model-level DCE performance.

## Next milestone after local tests

1. Add an experiment harness that compares uniform nearest-candidate selection
   against DCE cost minimization with identical candidate sets and payload bits.
2. Add batch-level distribution measurements (including empirical KL with
   explicit smoothing, plus a histogram-distance control) rather than treating
   the candidate proxy as KL.
3. Add the representation-specific quantizer and decoder only after defining
   the precise receiver contract and the exact transformation under test.
4. Only then register DCE as an experimental strategy and run a controlled
   Qwen2.5-3B pilot against the strongest relevant baseline.
