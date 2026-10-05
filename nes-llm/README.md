# NES — measurement suite for steganography in quantized LLM weights

This package is the runnable half of the repository. It embeds payloads into
quantization residual streams, extracts them, and measures whether the result
survives fidelity, noise and detection gates. Everything asserted in
`../RESEARCH_LOG.md` and `../RESEARCH_PLAN.md` was produced by the commands
below.

**Suite state: 35 PASS / 6 FAIL / 0 NOT_RUN / 0 ERROR** — 9 model ids, three
quantization formats (NF4, GPTQ, AWQ). Do not take that from this paragraph:
`python claim_audit.py` re-derives it from `results/`.

## Layout

| path | what it is |
|---|---|
| `src/experiments/` | one module per experiment, plus `experiment_registry.py` (ids, gates) and `manifest.py` (cell states) |
| `src/experiments/experiments/` | the experiment bodies — `run(context) -> artifact` |
| `src/quantization/` | pack/unpack, format dequantizers, `verify_dequantization` |
| `src/carrier_intelligence/` | layer profiling and embedding strategies |
| `src/steganalysis/` | statistical and neural detectors |
| `src/reporting/` | matrix and summary generation |
| `run_nes_experiments.py` | orchestrator: run, audit, write reports |
| `check_consistency.py` | cross-artifact invariants (9 checks) |
| `claim_audit.py` | re-derives every MEASURED claim from `results/*.json` |
| `tests/` | 38 unittest cases |

Artifacts are written to the **repo-root** `results/` (`paths.RESULTS_DIR`) —
two modules once wrote to `nes-llm/results` instead and drifted. Superseded
versions go to `results/_archive/`. Residual caches live under `cache/models/`
(gitignored, 1.0–8.6 GB per model).

## Running

```bash
cd nes-llm
../.venv/bin/python run_nes_experiments.py --models Qwen/Qwen2.5-3B
../.venv/bin/python run_nes_experiments.py --models Qwen/Qwen2.5-3B --exp exp3 exp6
```

A completed cell (PASS **or** FAIL) is skipped unless `--force` — a FAIL is a
measurement and is never rerun until it passes. Run **one model per process**:
residual caches alone reach 8.6 GB on a 26 GB machine.

## Verifying

```bash
../.venv/bin/python run_nes_experiments.py --audit  # cell states
../.venv/bin/python check_consistency.py            # cross-artifact
../.venv/bin/python claim_audit.py                  # measured claims
../.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

`claim_audit.py` is the strictest of the three: it exits non-zero on any claim
it cannot verify against its artifact, and it checks *counts* — models
measured, layers compared, entries present — because the value is the part
nobody doubts and the count is the part that goes stale. It has caught four
bad claims so far; `../RESEARCH_LOG.md` §17 records them rather than quietly
correcting them.

## Experiments

| id | measures | gate | state |
|---|---|---|---|
| exp1 | residual extraction → QACI round trip | 0 layer mismatch | PASS ×7 |
| exp2 | residual fingerprint (is it informative?) | ≥ 80% of layers with `mag_mean > 0.002` | 3 PASS / 4 FAIL |
| exp3 | clean embed → extract → decrypt | BER 0 | PASS ×7 (48,256 bits each) |
| exp4 | capacity curve | ≥ 500k bits at BER 0 | PASS (measured ≥ 10M) |
| exp5 | perplexity degradation | ≤ 2% | PASS (+0.0053%) |
| exp6 | BER under Gaussian noise | ≤ 0.02 @ σ=0.001, ≤ 0.10 @ σ=0.002 | PASS ×7 |
| exp7 | statistical undetectability | KL ≤ 0.05, accuracy ≤ 55% | PASS ×7 |
| exp7_neural | carrier-centred neural steganalysis | accuracy ≤ 55% | **FAIL** — 70.5% |
| exp8 | cross-model aggregation of exp2–exp7 | every sub-gate | **FAIL** (inherits exp7_neural) |
| exp9 | GPTQ / AWQ clean BER through *their own* dequantizers | BER 0, corr ≥ 0.95, residual ratio ≤ 0.5 | PASS — GPTQ 36/36 layers, AWQ 35/36 |
| exp10–12 | LWE-style strategy: extractability, grid width, cross-model | BER ≤ 0.02, detector ≤ 55% | measured; see `../RESEARCH_PLAN.md` §1 |

Thresholds live in `experiment_registry.THRESHOLDS`. No experiment edits its
own, and `claim_audit.py` pins them, so a gate cannot be quietly relaxed.

A FAIL above is a result, not a bug: each was investigated and the verdict is
recorded rather than rewritten. The six current FAILs are exp2 ×4,
exp7_neural, and exp8 inheriting it.

## Reading order

1. `../RESEARCH_LOG.md` — what was done, what broke, what was decided (§16 is
   the AWQ story, §17 the final state and claim-audit findings).
2. `../RESEARCH_PLAN.md` — every claim sorted MEASURED / FAIL / NOT_RUN, and
   what to do next (Phases A–D).
3. `../results/final_research_summary.md` — the generated report.
4. `../NES_MultiModel_Experiment_Guide.pdf` — the original specification.
