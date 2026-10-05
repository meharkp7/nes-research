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
| exp13 | keyless recovery of the LWE grid (W4.2): can an attacker read the channel without the key? | attacker BER ≥ 0.5 **and** width not locatable to 1% | **FAIL** — full keyless read at BER 0.0, width is a public constant |
| exp14 | blind-patch adversary (W3.2): detector cut at positions that know nothing about carriers | blind accuracy ≤ 55% (exp7's number, reused) | **PASS** — 50.0% blind vs 68.75% carrier-centered control; exp7's 70.5% is placement-conditioned |
| exp15 | LWE fidelity (W2): three-way perplexity in exp5's protocol, per model | embedding-specific PPL Δ < 2% (exp5's number, reused) | **PASS ×2** — +0.0077% (Qwen2.5-3B), +0.0501% (gemma-2-2b) |
| exp16 | cross-scheme detector (W3.1): train on one scheme, test on the other | both cross directions ≤ 55% (exp7's number, reused) | **PASS** — sign→LWE 50.00% with 62.85% control; LWE-side control collapsed (exp12-consistent), `controls_valid: false` |
| exp17 | QAE round trip (W1.1): the dict-adapter over the per-tensor ABC, exp3's production path | BER 0.0 (exp3's number, reused) | **PASS** — 0.0 over 48,256 bits, decrypt + match; `nf4_qae` registered BLOCKED with recorded diagnosis |
| exp18 | strategy × model matrix (W1.3, first pass): exp10's three axes × 4 strategies × 3 models | round trip 0.0, robustness 0.02/0.10, detector ≤ 55% (exp3/exp6/exp7's numbers, reused) | **12/12 round trips BER 0.0, 12/12 robustness pass — only detectability separates**: LWE 0.50 ×3 (wins 3/3), sign-family 0.59–0.84 everywhere; `neural`/`nf4_qae` excluded by name |
| exp19 | adaptive routing as designed (W5.1): σ-estimator routes to lwe/neural/sign, round trip through the routed branch | BER 0.0 for every round trip that runs (exp3's number, reused); routing choice is measurement, not gate | **three models → three branches** (gemma→lwe, Qwen→neural, Llama→sign); Qwen's route fails design-as-written (no trained model) — recorded, fallback branches both 0.0 |
| exp20 | sign/parity split dial (W5.3): parity share of carriers sweeps 0.0→1.0, exp10's three axes per cell | round trip 0.0, robustness 0.02/0.10, detector ≤ 55% (exp3/exp6/exp7's numbers, reused) | **the trade-off is a dial**: BER@σ0.002 0→0.0127 and detector 0.7875→0.50 with parity share; all 5 round trips 0.0; only pure parity wins, reproducing exp18's lwe cell exactly (delta 0.0) |

Thresholds live in `experiment_registry.THRESHOLDS`. No experiment edits its
own, and `claim_audit.py` pins them, so a gate cannot be quietly relaxed.

A FAIL above is a result, not a bug: each was investigated and the verdict is
recorded rather than rewritten. The six current **manifest** FAILs are exp2 ×4,
exp7_neural, and exp8 inheriting it. exp13's FAIL and exp14's/exp15's/exp16's/exp17's
PASSes are gate verdicts outside the manifest grid (standalone artifacts, like
exp12), so coverage stays 35 PASS / 6 FAIL — exp13's scope caveat is that the
payload is AES-GCM ciphertext: the *channel* is readable, the *message* is not,
exp16's is that one cross direction is uninformative by its own recorded
control, exp17's is that `qae` round-trips but produced no stealth number, and
exp18 carries per-cell verdicts (a measurement table, not one gate verdict),
and exp19's gate covers only the round trips that ran — which branch the
σ-estimate selects is measurement, with Qwen's unavailable route recorded as
the design's own failure. exp20's per-cell verdicts are exp18's rule applied
to five fractions of one scheme; the fraction sweep itself is the result.

## Reading order

1. `../RESEARCH_LOG.md` — what was done, what broke, what was decided (§16 is
   the AWQ story, §17 the final state and claim-audit findings, §19–§26 the
   Phase-B results so far: W4.2 keyless recovery, W3.2 blind patches, W2 LWE
   fidelity, W3.1 cross-scheme, W1.1 QAE round trip, W1.3 strategy matrix,
   W5.1 adaptive routing, W5.3 split dial).
2. `../RESEARCH_PLAN.md` — every claim sorted MEASURED / FAIL / NOT_RUN, and
   what to do next (Phases A–D).
3. `../results/final_research_summary.md` — the generated report.
4. `../NES_MultiModel_Experiment_Guide.pdf` — the original specification.
