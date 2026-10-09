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


## Batch-level prototype: implementation gate

Added `src/optimization/batch_optimizer.py` as a separate greedy prototype.
Unlike the previous per-carrier histogram-frequency proxy, it tracks selected
quantized-bin counts and scores each feasible candidate by the projected
aggregate squared-count mismatch for the current prefix plus perturbation.
The optimizer requires one payload-feasible candidate per carrier and checks
that the target histogram total matches the carrier count.

This is a **greedy heuristic**, not a global optimizer. The candidate ordering
can affect the result, and the squared-count prefix objective is not identical
to minimizing final TV or KL. The first gate is correctness and transparent
synthetic comparison, not a claim of superiority.

Tests are in `tests/test_batch_distribution_optimizer.py`. Run from
`nes-llm/`:

```bash
../.venv/bin/python -m unittest discover -s tests -p 'test_batch_distribution_optimizer.py' -v
../.venv/bin/python -m unittest discover -s tests -p 'test_distribution_constrained_optimizer.py' -v
../.venv/bin/python -m unittest discover -s tests -p 'test_dce_candidate_benchmark.py' -v
```

Next, construct a matched synthetic comparison using identical cover values,
payload bits, candidate sets, and toy quantizer for (1) nearest-feasible,
(2) the rejected independent per-carrier DCE objective, and (3) this batch
heuristic. Report all seeds, final TV/KL, perturbation, and BER. Do not infer
NF4 compatibility, model utility, or undetectability from synthetic results.


## Matched three-way benchmark added

Added `scripts/dce_three_way_comparison.py` and
`tests/test_dce_three_way_comparison.py`. The script compares, on identical
seeded synthetic cover values, payload bits, candidate sets, and toy
quantization:

1. nearest-feasible baseline;
2. existing independent per-carrier DCE;
3. greedy batch-level DCE.

For each method it reports BER, perturbation, histogram TV, and smoothed
histogram KL; paired differences versus the baseline are included. The
comparison is synthetic mechanics only. It does not establish NF4 compatibility
or any model-level security, utility, or stealth claim. The batch method is a
greedy heuristic and may still lose to the baseline.

Run from `nes-llm/`:

```bash
../.venv/bin/python -m unittest discover -s tests -p 'test_dce_three_way_comparison.py' -v
../.venv/bin/python scripts/dce_three_way_comparison.py --carriers 2000 --seed 20261009 --output ../cache/dce_three_way_comparison_20261009.json
```

The next decision is based on the matched comparison across multiple seeds,
not a single favorable run. Any follow-up weight tuning must be declared and
reported separately from a held-out evaluation.


## Matched three-way single-seed result (20261009; 2,000 carriers)

The user ran the matched comparison with the same synthetic cover, payload bits,
candidate sets, and toy quantizer for all methods.

| Metric | Nearest baseline | Independent DCE | Greedy batch DCE |
|---|---:|---:|---:|
| BER after toy quantization | 0.000000 | 0.000000 | 0.000000 |
| Mean squared perturbation | 0.021024 | 0.064852 | 0.182516 |
| Histogram TV distance | 0.036264 | 0.111773 | 0.005957 |
| Cover-to-embedded histogram KL (nats) | 0.006574 | 0.068248 | 0.001111 |

The greedy batch method reduced TV by about 83.6% and cover-to-embedded KL by
about 83.1% versus nearest-feasible on this seed, but mean squared perturbation
was about 8.68 times the baseline. Thus it shows a distribution/distortion
tradeoff, not an overall win. Zero BER is expected by construction of the toy
candidate sets. A single synthetic seed is not sufficient to select the method.

Added `scripts/dce_three_way_multiseed.py` to report per-method mean,
population standard deviation, range, strict wins against the baseline, and
paired deltas over multiple seeds. Tests are in
`tests/test_dce_three_way_multiseed.py`.

Run from `nes-llm/`:

```bash
../.venv/bin/python -m unittest discover -s tests -p 'test_dce_three_way_multiseed.py' -v
../.venv/bin/python scripts/dce_three_way_multiseed.py --carriers 2000 --seeds 20261009,20261010,20261011,20261012,20261013 --output ../cache/dce_three_way_multiseed_20261009.json
```

Decision gate: assess the full paired multi-seed tradeoff before changing
weights or proposing an NF4 implementation. This remains synthetic scalar
quantizer evidence only.


## Repeated-seed matched three-way result (five seeds; 2,000 carriers each)

The user ran the repeated-seed script successfully. Its three unit tests passed.
Across seeds 20261009–20261013, the method means were:

| Metric | Nearest-feasible baseline | Independent DCE | Greedy batch DCE |
|---|---:|---:|---:|
| BER after toy quantization | 0.000000 | 0.000000 | 0.000000 |
| Mean squared perturbation | 0.021028 | 0.062707 | 0.182154 |
| Histogram TV distance | 0.037735 | 0.110940 | 0.006250 |
| Cover-to-embedded histogram KL (nats) | 0.007226 | 0.062651 | 0.001368 |

Paired batch-DCE minus baseline means were:
- mean squared perturbation: +0.161127 (batch DCE has about 8.66x the baseline mean distortion);
- histogram TV: -0.031484;
- cover-to-embedded KL: -0.005858 nats.

The batch method strictly improved TV and cover-to-embedded KL in all five seeds (5/5 each), but it did not beat the baseline on perturbation in any seed (0/5). Independent DCE strictly beat the baseline on none of the reported metrics. BER is zero for every method by construction of the synthetic candidate set, so it is not evidence of comparative robustness.

**Decision:** the aggregate histogram improvement of greedy batch DCE is reproducible within this toy setup, but the distortion cost is too large to call it an overall improvement. Do not wire it into the production strategy registry and do not move directly to an NF4 pilot. Treat this as a reproducible synthetic trade-off, not evidence of stealth or model utility.

A sensible final DCE design gate, if this line is pursued, is a distortion-budgeted batch objective: constrain mean squared perturbation to a declared budget relative to the nearest-feasible baseline, then optimize the aggregate histogram within that budget. Report infeasible runs as failures rather than silently relaxing the budget. Compare on the same five seeds, and add held-out seeds before making a method-selection claim. If the histogram advantage disappears under a reasonable distortion cap, stop DCE and redirect effort to the established sign-embedding approach and real transformation/utility tests.


## Distortion-budgeted batch optimizer: validation gate

Added `src/optimization/budgeted_batch_optimizer.py`, which treats distortion
as a hard constraint rather than a soft penalty. Its budget is the minimum
payload-feasible per-carrier perturbation total (the nearest-feasible baseline)
multiplied by (1 + b), where (b) is the declared maximum relative increase.
At each greedy step, it reserves the minimum feasible distortion required by
all remaining carriers. It raises an error if no budget-feasible choice exists;
it never silently relaxes the budget.

Added a matched benchmark option and
`scripts/dce_distortion_budget_sweep.py`. The proposed exploratory sweep uses
five fixed seeds and budgets of 0%, 5%, 10%, and 25% above baseline distortion.
It reports paired perturbation, TV, and KL changes per seed and in aggregate.
The zero-budget setting is a control: it should select minimum-distortion
candidates, so it cannot be expected to improve distribution matching.

This is still synthetic scalar-quantizer evidence. Passing unit tests or
respecting the budget does not establish model utility, NF4 compatibility,
steganographic undetectability, cryptographic security, or checkpoint
robustness. The sweep should be reviewed before any representation-specific
pilot is considered.


### Matched-comparator requirement added after the first budget sweep

The first five-seed sweep respected all four hard budgets, but the gains over
nearest-feasible baseline were small and mixed: TV improved in 4/5 seeds at
5%, 5/5 numerically at 10% (one change was effectively zero), and 4/5 at 25%;
cover-to-embedded KL improved in 2/5, 3/5, and 4/5 seeds respectively. The
zero-budget control was identical to baseline. These results do not establish
a robust win.

The sweep now also records per-seed paired deltas between distortion-budgeted
batch DCE and the independent-DCE and unconstrained greedy batch-DCE methods,
in addition to nearest-feasible baseline. This is necessary to tell whether
the hard budget adds value beyond the prior synthetic methods. Re-run the
sweep after pulling the updated comparator output; do not use the earlier
report as evidence for the new cross-method comparisons.


\n\n## Final matched distortion-budget comparison (five seeds; 2,000 carriers each)\n\nThe updated sweep was rerun with paired comparisons against nearest-feasible,\nindependent DCE, and unconstrained greedy batch DCE for budgets of 0%, 5%,\n10%, and 25%. The report status remains `SYNTHETIC_MECHANICS_ONLY`.\n\n| Budget above baseline | Mean relative MSE increase | Mean paired Δ TV vs nearest baseline | Mean paired Δ KL vs nearest baseline |\n|---:|---:|---:|---:|\n| 0% | 0.000% | 0.000000 | 0.000000 |\n| 5% | 4.999% | -0.000695 | -0.000109 |\n| 10% | 9.999% | -0.001192 | -0.000158 |\n| 25% | 25.000% | -0.000794 | -0.000186 |\n\nAll 20 runs stayed within their declared budgets and reported BER 0 in the toy\nquantizer. That BER is expected by construction and is not evidence of real NF4\nrecovery. The mean distribution gains over the nearest-feasible baseline are\nsmall; per-seed KL outcomes are mixed. At 10%, mean TV improves by about 0.00119\nwhile mean squared perturbation rises by about 10%.\n\nCompared with independent DCE, budgeted batch DCE improves the mean TV/KL metrics\nwhile using less perturbation in this synthetic setup. Compared with greedy\nbatch DCE, it uses much less perturbation but has substantially worse TV/KL.\nNo method dominates across the measured objectives.\n\n**Final decision:** retain the synthetic-only result; Candidate B remains open pending real NF4 execution. Do not\nadd another objective, tune weights further, or promote DCE to the production\nregistry based on these results. A real NF4 DCE pilot has now been implemented but is **not yet executed**; the earlier synthetic implementation\nthe current implementation**, because the existing optimizer operates on toy\nscalar candidate sets and is not integrated with the B1.4 packed-code carrier\ncontract. Reopening DCE requires a separately specified NF4-aware selection\nprotocol with a receiver-reproducible carrier map, followed by a matched pilot\nthat measures payload recovery, perturbation, utility and detectability. Until\nthat exists, the paper should report DCE as a synthetic trade-off and negative\nresult, not as a validated contribution.\n

## Follow-up correction: Candidate B remains open pending real NF4 execution

The synthetic DCE results above are not a real-NF4 result and must not be used to
close Candidate B. A separate first-stage real-NF4 tensor pilot now exists at
`scripts/real_nf4_candidate_eval.py`. It uses a real BitsAndBytes NF4
quantization state and compares nearest-feasible code selection with a
histogram-aware DCE selector using the same keyed positions and payload. The
script is syntax-checked in CI, but **has not yet been run against the user's
local FP16 model cache**.

The pilot is tensor-level only. If DCE looks promising, follow it with artifact
save/reload and matched utility/detectability measurements. Until those results
exist, Candidate B is **NOT RUN on real NF4**, not a real-NF4 failure or success.
