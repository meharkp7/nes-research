# Research Log — Decisions, Results, and Reasoning

Every non-obvious choice, every measured result, and every thing I chose
**not** to do. Written as I go so the reasoning is auditable later, not
reconstructed after the fact.

Convention: a result is *measured* only if an artifact exists for it.
Anything stated from reading code is labelled **[read]** and is a claim
about implementation, not behaviour.

---

## 0. Starting point

Task: run the autonomous agent handoff (`NES_AUTONOMOUS_AGENT_HANDOFF.md`),
then diagnose the recorded FAILs and fill the NOT_RUN cells.

Handoff safety rules that constrain everything below:
- never turn a FAIL into a PASS
- never silently change a research threshold
- never overwrite a completed result without preserving it
- preserve completed work rather than recomputing it

---

## 1. Step 1 — Exp8 reported a completed detector as NOT_RUN

**Handoff said:** fix `exp8_result_adapter.py`, which holds both
`get_exp5_result` and `find_neural_detector_result`.

**What I found:** the adapter was already correct. It resolves candidates
off `Path(__file__).resolve()` and
`parents[3]/"results"/"exp7_neural_detector_results.json"` exists and
matches.

The real defect was that nobody called it. `exp8_real_cross_model_table.py`
defined its own `find_neural_detector_result` that only checked a
CWD-relative `results/` dir and only understood a nested per-model format,
while the saved artifact is flat. It returned `None`, so `g6_neural`
showed NOT_RUN despite a completed 70.5% result.

**Decision:** fix the call site, not the adapter. Import the adapter
function and delete the duplicate. Also made `RESULT_DIR` resolve from
`__file__` so discovery is CWD-independent.

**Result:** `g6_neural_detector` → FAIL @ 0.705, `overall_status` → FAIL.
That is the expected end state, not a regression.

---

## 2. Infrastructure

Built `src/experiments/` and `src/reporting/`:

| Module | Purpose |
|---|---|
| `paths.py` | repo-relative artifact locations, cwd-independent |
| `environment_check.py` | device detection + recorded platform caveats |
| `artifact_manager.py` | structured JSON writes; archives prior versions |
| `manifest.py` | PASS/FAIL/NOT_RUN/MISSING/IMPL_ONLY/ERROR kept distinct |
| `experiment_registry.py` | gates and target models in one place |
| `runner.py` | orchestration, skip-if-completed |
| `residual_source.py` | reuse the 100GB+ on-disk residual cache |

**Decision:** a completed cell (PASS *or* FAIL) is skipped unless
`--force`. Rerunning a FAIL until it passes is result-shopping, and the
skip is what prevents it. `--force` archives the previous artifact rather
than overwriting it.

**Decision:** the runner is **model-major**, not experiment-major.
Residuals for an 8B model are several GB of float32; holding every model's
residuals at once is ~24GB for four models and OOMs this machine.

**Decision:** gates live in the registry, never inside an experiment body,
so no experiment can relax its own threshold.

---

## 3. Bugs found while running

Each of these produced a *plausible-looking* wrong answer rather than a
crash, which is why they were worth the time.

### 3.1 BER was never measured

`DecryptPipeline`'s stats dict reports `success` and `bits_extracted` but
has **no `ber` key**. `stats.get("ber")` returned `None`, and
`None <= 0.0` is a TypeError; worse, a `stats.get("ber") or 0.0` pattern
reports a clean round trip without comparing a single bit.

**Fix:** Exp3/Exp4/Exp9/Exp11 compare transmitted against extracted bits
directly. Exp3 now reports `bits_compared`, `bit_errors` and `ber` from a
real comparison.

### 3.2 `bits_embedded` is an int, not a list

The `EmbedResult` docstring in `intelligent_embedder.py` is wrong.
`len(result.bits_embedded)` raised `TypeError`.

### 3.3 Exp9 was measuring NF4 and labelling it GPTQ

`src/model/exp9_alternative_quant.py` called `load_model_pair`, which
applies an NF4 `BitsAndBytesConfig` to whatever model id it is given. A
GPTQ checkpoint would be re-quantized as NF4 and then reported as
"GPTQ BER = 0".

**Fix:** new `src/quantization/adapters.py` with format-specific
dequantizers, a dispatch guard that refuses to read a GPTQ layer through
the AWQ layout, and tests asserting **bit-exact** equality against the
AutoGPTQ reference unpack.

### 3.4 The GPTQ nibble axis is transposed, silently

AutoGPTQ unpacks weights on axis 1 but zero points on axis -1. Using the
same axis for both produces a transposed weight matrix of the right shape,
so it neither errors nor looks obviously wrong. Caught by asserting
bit-exact equality with the reference rather than by inspection.

Also: AWQ does **not** apply GPTQ's `-1` zero-point bias, so copying the
`+1` shifts every weight by one scale unit. Asserted that the two paths
genuinely differ.

### 3.5 The 100GB residual cache was never being hit

`CACHE_ROOT` was the relative path `"cache/models"`, which resolves to
`nes-llm/cache/models` whenever the suite runs from `nes-llm/`. Every run
was redoing a full NF4 dequantization pass over every layer.

**Fix:** repo-absolute path. Exp4 went from many minutes to 65 seconds.

**Verified** the cache is bit-exact against live extraction before relying
on it (max per-layer diff 3.5e-9 on Qwen2.5-3B, ~4e-6 relative).

### 3.6 Exp1 loaded NF4 + FP16 to count layers

`load_model_pair` materialises both (~19GB for a 7B model). Exp1 exists to
validate the residual/QACI path and only ever needed a layer count — which
the residual dictionary already states authoritatively, since those tensors
came out of the real model.

Qwen2.5-7B and Mistral-7B hit
`MPS backend out of memory (16.11 GiB allocated, 14.43 GiB other, max 30.19 GiB)`
and Exp1 recorded **FAIL**, conflating "could not load" with "broken
architecture".

**Fix:** Exp1 takes the layer count from the residuals and never loads a
model, so Exp1–Exp7 are cache-only for every model. Exp1 also reports
NOT_RUN (not FAIL) when residuals are unavailable.

**Secondary fix:** while editing Exp1 I consumed its
`if context.has_residuals:` guard, leaving unreachable code, so `qaci_runs`
was never recorded. Caught because Exp1 failed for Qwen2.5-7B even after
the OOM fix.

**Secondary fix (Exp8):** Exp8 now runs the four model-free gates first and
loads weights only for `g4_ppl`, isolating a PPL failure to that one gate.

### 3.7 Exp5 stalled 20+ minutes on MPS

`build_embedded_eval_model` uploads 36 × 90MB reconstructed weights to MPS
and the PPL scalar syncs block behind that upload queue in
`_local_scalar_dense_mps` — the hazard the handoff describes. `stdout` was
also block-buffered, so it looked hung rather than slow.

**Fix:** lazy `ModelContext.ensure_models()`; Exp5 reuses the completed
three-way PPL measurement via the existing `exp8_result_adapter`.
`--recompute-exp5` is a separate opt-in so a routine cell refresh cannot
trigger an hours-long run.

### 3.8 `gate_for` raised on Exp7

`gate_for(experiment)` looked the name up in `EXPERIMENTS`, but Exp7 passes
its gate key (`exp7_statistical`). Now accepts either.

### 3.9 Latent squeeze bug in the Exp7 neural training path

`Detector.forward` already applies `squeeze(-1)`, so the extra
`.squeeze(1)` raised on the 1-D result. Never surfaced because the module
reused the recorded 70.5% result instead of training.

---

## 4. Diagnosing the two recorded FAILs

Both FAILs are **measurements, not bugs**. I did not attempt to make them
green; the handoff forbids that and it would falsify the work.

### 4.1 Exp2 — residual magnitude gate

Gate: mean `mag_mean > 0.002` for ≥80% of layers.

**Measured across all 7 profiles: 1/7 pass** (Mistral-7B at 81.2%). Six
fail, including the 3B model with the largest residuals of the set.

A criterion no model meets is a statement about the criterion.

I checked whether the module choice explained it — `down_proj`,
`gate_proj`, `up_proj` all sit at ~0.0019 for Qwen2.5-3B, so no.

I then measured the same statistic across quantization formats
(`exp2_criterion_calibration`):

| format | mean magnitude | layers > 0.002 |
|---|---|---|
| nf4_double_quant | 0.001879 | 0% |
| nf4_single_quant | 0.001877 | 0% |
| **fp4** | **0.002526** | **100%** |

**Finding:** the criterion tracks the *quantization format*, not the model.

**Decision:** did **not** change the threshold and did **not** rewrite any
Exp2 verdict. Recalibrating is a research decision for the authors.
Switching to FP4 to pass would also invalidate the entire NF4 premise.

### 4.2 Exp7 neural — 70.5% vs a 55% gate

The parameter study (`exp7_neural_parameter_study`) swept alpha ×100,
gamma ×5, payload ×10:

| variant | accuracy |
|---|---|
| baseline (alpha=1e-3, gamma=2.5) | 71.88% |
| alpha 1e-4 | 71.88% |
| alpha 1e-2 | 70.00% |
| gamma 1.0 | 69.38% |
| gamma 5.0 | 74.38% |
| payload 100k | 74.38% |

Range 69.38–74.38%, spread 5%. **No configuration comes near the gate**, so
this is not fixable by retuning.

**Mechanism:** sign embedding rewrites a carrier to `±|r|` selected by the
bit. That changes nothing when the bit already agrees with the carrier's
sign, so only ~25 of 4096 values per patch differ and some pairs are
byte-identical. The scheme forces a ~50/50 sign split at carriers
regardless of the clean distribution.

### 4.3 A study that was wrong, twice

My first parameter study scored **exactly 50.00% on every variant**. I
treated that as a result for far too long before checking it.

50.00% is chance: the detector learned nothing. Sign embedding changes so
few values per patch that 120 pairs cannot train a detector to find it. The
recorded 500-pair result reaches 70.5%.

**Decision:** added an explicit `INVALID STUDY` guard for an all-at-chance
sweep, so a future run cannot report a power failure as a security win.
I repeated this mistake in the Phase 2 comparison and caught it the same
way.

### 4.4 A verification I got wrong

While investigating the 50% reading for LWE, I compared it against sign by
reusing the parameter study's `build_dataset` — which never sets
`embedding_strategy`, so **both columns were sign**. The datasets were
identical because they were the same scheme.

The correct comparison needed its own builder. This is why I re-derived the
result rather than reporting the convenient reading.

---

## 5. Strategy registry (Phase 1)

Eight strategies on disk; production hardcoded `SignEmbeddingStrategy`, so
`EmbeddingConfig.embedding_strategy` was accepted and **ignored**. No
experiment could select another scheme.

**Decision:** give each strategy one contract, and record the properties
that decide viability rather than discovering them at run time:

- `forces_sign_flip` — the leak behind 70.5%
- `extract_needs_cover` — a scheme needing the cover cannot hide anything
- `status` — READY / BLOCKED / NEEDS_TRAINING

The load-bearing test is a round trip **without the cover**. A real
extractor only holds stego weights; a scheme needing the cover was never
usable outside its own test harness.

That check immediately found two things I would otherwise have shipped:
- `lwe` required `residuals_ref` to size its grid
- `magnitude_aware` is sign embedding with a bigger margin
  (`boosted if bit == 1 else -boosted`), not an alternative

---

## 6. Phase 2 — head-to-head

| strategy | extractable | BER@σ=0.001 | detector | verdict |
|---|---|---|---|---|
| sign | yes | 0.0000 | 72.50% | works, detected |
| magnitude_aware | yes | 0.0000 | 67.50% | works, detected |
| lwe | **no** | **0.5036** | 50.00% | carries no data |
| neural | no | — | — | needs training |

**LWE's BER was 0.5036 — chance, not degradation.** Production extraction
is `SignExtractor`, which reads carrier *signs*; it cannot decode a parity
grid. And LWE's own `extract()` needed the cover.

**Decision:** report no winner rather than rounding 67.50% into progress.
`magnitude_aware`'s 5-point gain is noise on a 55-point scale, not a
mechanism difference — both schemes force the identical sign flip.

**Also found:** `sign_bias` is statistically indistinguishable across
sign (0.01479), magnitude_aware (0.01450) and lwe (0.01486). The simple
statistical gates cannot separate these schemes; only the neural detector
can. That is itself worth reporting — statistical undetectability is a
weak proxy for neural security.

---

## 7. Phase 3 — making LWE usable

### 7.1 Removing the cover dependency

The grid width depends on per-layer `residual_std`, and the embedding is
sparse — a 10,000-bit payload touches ~10,256 of ~811M values, about
0.001%. So the embedded tensor's own std equals the cover's.

**Decision:** derive the width from the stego tensor when no cover is
supplied. This is valid because of the sparsity, and I recorded the
sparsity figure as the justification rather than asserting it.

Round trip went from **BER 0.5036 (chance) to 0.0000** without the cover.

### 7.2 A scheme-specific extractor

`DecryptPipeline` was hardcoded to `SignExtractor`. Added
`embedding/extractors.py` mapping a scheme to the extractor that can decode
it, so non-sign schemes are not handed to the sign-based one.

`DecryptPipeline` now takes `strategy_name`; the default stays `sign`, so
nothing else changes behaviour.

### 7.3 The grid width was pinned by a floor

Before sweeping, I checked what actually controls the width:

```
interval_width = max(std * alpha * scale, min_magnitude * 2)
```

With `std ≈ 0.002`, `alpha = 0.001` → `2e-6`, while `min_magnitude * 2`
= `2e-3`. **The floor dominates by ~1000× and alpha has no effect at all.**
The shipped defaults sat at a width where the scheme is undetectable but
collapses under noise (BER 0.586 at σ=0.001).

A sweep of alpha alone would have been a **no-op**. Good catch before
spending an hour proving nothing changed.

### 7.4 The frontier

Sweeping grid width directly:

| grid width | BER@σ=0.001 | robustness | detector | both gates |
|---|---|---|---|---|
| 0.002 | 0.5857 | FAIL | 50.00% | no |
| 0.005 | 0.0127 | PASS | 50.00% | **YES** |
| 0.010 | 0.0000 | PASS | 50.00% | **YES** |
| 0.020 | 0.0000 | PASS | 50.00% | **YES** |
| 0.050 | 0.0000 | PASS | **70.62%** | no |

There is a real window where both gates pass.

**The 0.050 row is what makes this trustworthy.** At that width the
detector jumps to 70.62% — essentially sign's number. So the detector
*does* learn when signal exists, and the 50% readings are real
undetectability rather than a broken detector. That is the check I had
twice fooled myself out of, and here it passes.

`DEFAULT_GRID_WIDTH = 0.010` — middle of the window, margin on both sides,
every other width still reachable via the `grid_width` argument.

### 7.5 Neural: why it produced nothing

**It was not slow. It was unrunnable.** `NeuralEmbeddingTrainer.train()`
does, before the first batch:

```
torch.cat(2.44B residuals)      9.8 GB
.abs()                          9.8 GB
torch.randperm(n_total)        19.6 GB   (int64)
kthvalue                        full pass
```

≈39 GB on a machine where NF4+FP16 already OOMs at 30 GB.

Added `train_sampled()`, which samples per layer with `randint` (no
permutation over the full range) and estimates the magnitude threshold from
a bounded subsample. **Written but not yet run** — see open items.

**Design concern recorded, not yet measured [read]:** the objective
rewards matching the value distribution's mean and std, which a sign flip
satisfies trivially. Nothing in the loss penalises the sign leak, so the
encoder may well converge to the same sign-based solution. Testing that is
the point of running it.

---

## 8. Laptop memory

26 GB RAM, ~30 GB MPS ceiling, 107 GB model cache, 205 GB free disk.

Residuals are the largest single allocation: 3.2 GB (Qwen-3B) to 8.6 GB
(Gemma-9B) in float32.

**float16 is not a safe substitute.** ~2.4% of residual values per layer
are subnormal in fp16 and flush to zero. I verified this:
`mag_mean` is unchanged (so Exp2 is unaffected) but it would zero 2.4% of
the weight matrix in any reconstruction, which is catastrophic for
perplexity.

**Decision:** keep float32 exactly and stop holding every layer pinned.
Residuals are materialized once as raw `.npy` and opened `mmap_mode='r'`,
so the OS owns the pages — clean mapped pages are purgeable and reclaimed
under pressure, which a tensor in a Python dict cannot be.

Verified bit-exact against the in-RAM load. Measured RSS 2.03 → 1.82 GB
immediately; the real benefit is reclaimability under pressure, not the
immediate delta.

**Honest note:** the immediate saving is modest (~10%). This is insurance
for the larger models, not a fix for today's runs.

---

## 9. Open items, in the order agreed

1. **Verify Qwen-7B's residual cache.** `mag_mean` is 0.009412 — 7.9× every
   other model — with a 10× spread *within* the model while every other
   model is tightly clustered. Never completed: two attempts hit MPS OOM
   because the model run held the memory. If the cache is bad, every
   Qwen-7B number in the manifest is wrong.
2. **exp9** — GPTQ/AWQ checkpoints not cached; needs `--download-exp9`.
3. **Phi-3-mini** — only config cached (24 KB); needs ~7 GB.

Each proceeds only if the previous one comes back clean.

**Still unresolved from earlier:**
- `bits_embedded` (48,256) is consistently below `payload_bits` (50,000),
  a ~3.5% shortfall. Not chased.
- `src/model/loader.py` (3-value return) and `model_loader.py` (dict return)
  remain two divergent residual implementations; legacy `scripts/exp*.py`
  still import the former.
- The `LWE-Inspired` naming: there is no lattice, no matrix A and no
  SIS/LWE instance, so the post-quantum claim in `lwe_strategy.py`'s
  docstring is not supported by the implementation. Flagged, **not**
  renamed — renaming a claimed contribution is the authors' call.
---

## 10. Phase 3b — cross-model confirmation (`exp12`)

The grid width was found on one model. A setting that passes both gates on
one model is a lead, not a result, so it was re-measured on every cached
model.

| model | layers | BER@σ=0.001 | detector | both gates |
|---|---|---|---|---|
| Qwen/Qwen2.5-3B | 36 | 0.0000 | 50.00% | **PASS** |
| mistralai/Mistral-7B-v0.3 | 32 | 0.0000 | 50.00% | **PASS** |
| Qwen/Qwen2.5-7B | 28 | 0.0000 | 50.00% | **PASS** |
| meta-llama/Llama-3.1-8B | 32 | 0.0000 | 50.00% | **PASS** |
| google/gemma-2-9b | 42 | 0.0000 | 50.00% | **PASS** |
| TinyLlama-1.1B | 22 | — | — | SKIPPED (0-layer cache) |

**5 of 5 measured models satisfy both gates**, across 4 families and 22–42
layers. Same 55% gate and same detector architecture as the recorded 70.5%
sign result.

Why the exactly-50.00% readings are credible, given I was fooled twice by
this number earlier: the same detector, same code, same settings returns
72–74% against sign and 70.62% against LWE at grid width 0.050. A detector
that returns chance is therefore not finding signal, because signal it can
find produces 70%+ immediately.

**Honest limits of this result:**
- The detector is *carrier-centered*: paired patches from the same layers
  of the same model. This is a deliberately favourable setting for the
  adversary, but it is **not** a general claim that the embedding is
  undetectable.
- A detector trained against one scheme and tested on another has not been
  tried. That is a strictly stronger adversary and would be the natural
  next experiment.
- Four of the five results come from models whose residual caches have not
  been independently re-derived (see open item 1).

**Decision:** promote LWE to production-selectable (already wired in §7.2,
with `DEFAULT_GRID_WIDTH = 0.010`). Keep sign as the default — changing the
default is a research decision, not something to do silently in either
direction.

---

## 11. Item 1 — Qwen2.5-7B residual cache verification

**Question:** `mag_mean` was 0.009412, ~7.9× every other model, with a 10×
spread *within* the model while all others are tightly clustered. That does
not look like a model property.

**Verification** (`src/model/verify_residual_cache.py`) compared cached
residuals against freshly extracted ones, layer by layer, releasing each
before the next so memory stays bounded:

```
layer  0: exact=True  live_mag=0.009930  cached_mag=0.009930  max_diff=0.000e+00
layer 14: exact=True  live_mag=0.012713  cached_mag=0.012713  max_diff=0.000e+00
layer 27: exact=True  live_mag=0.001212  cached_mag=0.001212  max_diff=0.000e+00

bit-exact on all layers : True
VERDICT                 : cache matches live extraction
```

**The cache is correct.** The large magnitude is real: Qwen2.5-7B genuinely
has a **bimodal** residual distribution — layers 25/27 sit at ~0.0012 while
layers 0–20 sit at ~0.010–0.013.

### The actual bug was in the recorded profile, not the cache

Comparing the cache against the profile committed at `000f660`:

| layer | cache mag_mean | old profile | ratio |
|---|---|---|---|
| 0 | 0.009930 | 0.001070 | 0.108 |
| 14 | 0.012713 | 0.001221 | 0.096 |
| 20 | 0.013017 | 0.001251 | 0.096 |
| 25 | 0.001273 | 0.001273 | **1.000** |
| 27 | 0.001212 | 0.001212 | **1.000** |

The old profile matches the cache **exactly** for the small-magnitude
layers and under-reports by ~10× for the large ones. That is a profiler
biased toward low-magnitude values, not cache drift.

Why it went unnoticed: the bug is invisible on every other model because
their layers all sit near the low end (Qwen-3B at ~0.0018), where a bias
toward small values barely moves the mean. Qwen-7B is the only model whose
layer distribution is spread enough to expose it. The Qwen-3B profile
reproduced to 4e-6 relative precisely because that model could not reveal
the error.

**Decision:** the current profile (recomputed from the verified cache) is
correct and supersedes the old one. The old artifact is preserved in git
history. **Not** re-derived further — the cache is bit-exact against live
extraction, which is the strongest available check.

**Consequence:** the Qwen2.5-7B numbers in the manifest are valid. The
earlier concern about them was wrong, and is now closed rather than
outstanding.

**Not investigated:** the exact mechanism of the old profiler's bias. That
profiling path has been replaced by `LayerProfiler`; the old numbers are
only in git history. Worth knowing if any pre-existing artifact used it.

---

## 12. Phase 3 — status

**Complete and successful.** Recapping what Phase 3 produced, since it is
the substantive result of this work:

| Step | Outcome |
|---|---|
| 3a. LWE cover dependency removed | BER 0.5036 (chance) → **0.0000** |
| 3b. Scheme-specific extractor added | `DecryptPipeline` no longer assumes sign |
| 3c. Grid width found to be floor-pinned | alpha was a **no-op**; sweeping it would have proved nothing |
| 3d. Frontier swept | width 0.005–0.020 satisfies **both** gates |
| 3e. `DEFAULT_GRID_WIDTH = 0.010` set | middle of the window, margin both sides |
| 3f. Confirmed across models | **5 of 5 measured models pass both gates** |

Measured comparison, sign vs LWE at the chosen width:

| | sign | LWE (0.010) |
|---|---|---|
| BER @ σ=0.001 | 0.0000 | 0.0000 |
| neural detector | **72.50%** | **50.00%** |
| detectability gate (≤55%) | FAIL | PASS |

Cross-model, LWE at width 0.010: Qwen2.5-3B, Mistral-7B, Qwen2.5-7B,
Llama-3.1-8B, Gemma-2-9B — all BER 0.0000, all detector 50.00%, all both
gates PASS. TinyLlama SKIPPED (zero-layer cache).

**Why the 50.00% readings are credible**, having been fooled twice by that
exact number earlier: the same detector on the same code returns 72–74%
against sign and 70.62% against LWE at width 0.050. A detector that returns
chance is not finding signal, because signal it can find produces 70%+
immediately.

### What this does and does not establish

Does establish: under a carrier-centered detector on paired patches, LWE
grid encoding at width 0.010 satisfies both the robustness and the
detectability gate, on 5 models from 4 families.

Does **not** establish:
- that the embedding is undetectable in general. The detector is
  carrier-centered: paired patches from the same layers of the same model.
  That is deliberately favourable to the adversary, but it is one adversary.
- anything about a detector **trained on one scheme and tested on another**,
  which is a strictly stronger test and not yet run.
- anything about LWE's security claim. The scheme is LWE-*inspired*
  (key-derived grid, parity encoding). There is no lattice, no matrix A and
  no SIS/LWE instance, so it is not post-quantum secure and the docstring
  in `lwe_strategy.py` claiming otherwise is unsupported. Not renamed —
  see §16.

**Decision:** LWE is production-selectable via
`EmbeddingConfig.embedding_strategy='lwe'`. **Sign remains the default.**
Changing the default is a research decision and should not happen silently
in either direction.

---

## 13. Item 2 — Exp9, and why it was NOT_RUN

**The stated reason was wrong.** The handoff and my own audit both said
"no GPTQ/AWQ checkpoint in the local cache". The actual reason:

```
Qwen/Qwen2.5-3B-GPTQ-Int4  → 404 RepositoryNotFoundError
Qwen/Qwen2.5-3B-AWQ        → 404 RepositoryNotFoundError
```

**Those model ids do not exist on the Hub.** The targets were wrong, not
merely uncached.

**Why nobody noticed:** `exp9` returned NOT_RUN *before* attempting to load
anything, so a nonexistent model and an uncached model were indistinguishable.
That is a design flaw in the reporting, not just a bad constant — the
status carried no information about which of the two it was.

**What actually exists** (verified via the Hub API, all ungated):

| model | size |
|---|---|
| Qwen/Qwen2.5-3B-Instruct | 6.18 GB |
| Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 | 2.08 GB |
| Qwen/Qwen2.5-3B-Instruct-AWQ | 2.70 GB |

Only **Instruct** variants are published for these formats.

**Consequence I had to handle:** `R = W_FP16 − W_dequant` is only
meaningful against the *same* model in FP16. The old code reached for the
reference by string-stripping the suffix
(`model_id.replace("-GPTQ-Int4","")`), which would have silently produced
the base model — different weights, plausible numbers, no meaning.

**Fix:** each target now declares an explicit `fp16_reference`, and the run
errors if it is absent rather than guessing.

**Real checkpoint parameters** (read from config, so this is measured not
assumed):
- GPTQ: `bits=4, group_size=128, desc_act=false`
- AWQ: `bits=4, group_size=128, zero_point=true, quant_method=awq`

Both match the assumptions the adapters were written against.

**This is the first real test of the adapters.** Until now they were only
verified against a synthetic reference — bit-exact against AutoGPTQ's
unpack, but never against an actual checkpoint. If the nibble axis or the
zero-point bias were wrong, a real GPTQ checkpoint is where that shows.

---

## 14. Item 2 — Exp9, the result, and what it exposed

### The targets never existed

```
Qwen/Qwen2.5-3B-GPTQ-Int4  -> 404
Qwen/Qwen2.5-3B-AWQ        -> 404
```

Both were hardcoded in `exp9_alternative_quant.py`. The handoff and my own
audit recorded the reason as "no GPTQ/AWQ checkpoint in the local cache" —
**that was wrong**. The model ids do not exist.

**Why it stayed hidden:** `exp9` returned NOT_RUN *before* attempting to
load anything, so "this model id is fictional" and "this model is not
downloaded yet" produced the identical status. The status carried no
information about which failure it was. Fixed by making the failure
modes distinguishable.

### Only Instruct variants are published

Real, ungated checkpoints:

| model | size |
|---|---|
| Qwen/Qwen2.5-3B-Instruct | 6.18 GB |
| Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 | 2.08 GB |
| Qwen/Qwen2.5-3B-Instruct-AWQ | 2.70 GB |

The old code derived the FP16 reference by string-stripping the suffix,
which would have pulled the **base** model — different weights, plausible
numbers, no meaning. Each target now declares an explicit
`fp16_reference` and errors without one.

### transformers cannot load either format here

GPTQ requires `optimum`, AWQ requires `gptqmodel`; neither is installed.
Rather than add two heavy dependencies for a path that only needs to read
four packed tensors, `packed_loader.py` reads them straight from the
safetensors shards. The adapters already operate at tensor level, so this
also keeps the dequantization logic under our own test rather than
delegating the interesting part to a third-party runtime.

### Two adapter bugs that only real checkpoints could expose

| | GPTQ | AWQ |
|---|---|---|
| `qweight` shape | (1376, 2048) = [in/8, out] | (11008, 256) = [in, out/8] |
| packs nibbles along | **input** | **output** |
| `scales` | (86, 2048), groups over in | (86, 2048), groups over in |

Both store the weight as `[in, out]`; `nn.Linear.weight` is `[out, in]`, so
a transpose is required.

The earlier synthetic tests could not catch either, because the synthetic
tensors were built from the same assumptions being tested. **A round-trip
test built on an assumption cannot detect that the assumption is wrong.**

A third bug sat in `residual_for_layer`: on a shape mismatch it called
`.reshape()`, which has the same element count and therefore *succeeds*
while scrambling the matrix. It now transposes when the shapes are
swapped, and raises rather than reshaping in any other mismatch.

### The verification gate, and why it exists

`verify_dequantization` checks a dequantizer against the true FP16 weight
before any residual is computed:

| format | correlation | residual ratio | verdict |
|---|---|---|---|
| GPTQ | **0.9903** | 0.140 | correct |
| AWQ | **0.2343** | 1.027 | wrong |

I brute-forced 24 AWQ variants (8 nibble rotations × 3 zero-point
treatments × 3 formulas) and the best correlation was 0.2343. That is not
a dequantizer that is slightly off; the layout is not understood.

### Result

| target | status | evidence |
|---|---|---|
| Qwen2.5-3B-Instruct-GPTQ-Int4 | **PASS** | clean **BER 0.0**, mean abs residual 0.002787, corr 0.9903 |
| Qwen2.5-3B-Instruct-AWQ | **NOT_RUN** | dequantization unverified (corr 0.2343) |

**NES works beyond NF4.** GPTQ carries a payload at BER 0.0 through a
format-specific dequantization path verified against the FP16 reference.

AWQ is recorded NOT_RUN rather than given a BER. A wrong dequantizer
produces a residual of the right shape and a plausible magnitude —
0.0196 mean abs, the same order as the working NF4 residuals — so nothing
downstream would have objected. The experiment would have reported a
number and measured nothing. That is the specific failure this session
has been guarding against, and it is now gated in code.

**Not done:** AWQ's layout. The honest next step is installing `gptqmodel`
or `autoawq` and diffing against its unpack, rather than continuing to
guess. `zero_point: true` in the config and the output-axis packing are
the likely areas.
