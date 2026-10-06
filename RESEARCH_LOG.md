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

**Measured across all 7 exp2 cells: 3 pass, 4 fail.** gemma-2-9b (88.1% of
layers above threshold), Phi-3-mini (100%) and Mistral-7B (81.2%) pass;
Qwen2.5-3B (11.1%), Qwen2.5-7B (75%), TinyLlama (0%) and Llama-3.1-8B
(68.8%) fail.

> **Audit correction (§17, finding 2 — closed).** This section previously
> read *"1/7 pass (Mistral-7B at 81.2%). Six fail."* That was the
> `exp2_criterion_calibration` snapshot: a different set of files (the
> legacy `residual_profile_*.json` profiles, then without Phi-3), written
> between 11:22 and 11:34 on Oct 2, before gemma-2-9b and Llama-3.1-8B
> were re-run — its recorded values for those two, 0.000382 and 0.000879,
> against today's 0.003605 and 0.022642. It has since been regenerated
> from the profiles as they stand (eight now, Phi-3 included) and agrees
> with exp2 on every shared model. The cells above are what the manifest
> reports; they are what any claim about exp2 must match.

A criterion that 4 of 7 models fail is a statement about the criterion as
much as about the models.

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

> **Audit note (§17, finding 1 — closed).** At audit time the artifact
> for this run held **three** measured models: Qwen2.5-3B and Mistral-7B
> were absent from the file altogether, not SKIPPED, and no archived
> version had ever contained them. It has been re-run across all six
> cached models and now records **five measured, one skipped**
> (TinyLlama, incomplete cache) at BER 0.0000 / detector 50.00% on every
> measured row. The table above is now backed by the artifact rather
> than by console output, which is the only reason 5/5 is citable.

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
| 3f. Confirmed across models | **5 of 5 measured models pass both gates** (re-run closes the audit finding, §17) |

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
  see §17.

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

**One non-NF4 format round-trips at BER 0.0.** GPTQ carries a payload
through a format-specific dequantization path verified against the FP16
reference. One format, one model, one payload size — see §16 for the
narrowed claim that now covers both formats.

AWQ is recorded NOT_RUN rather than given a BER. A wrong dequantizer
produces a residual of the right shape and a plausible magnitude —
0.0196 mean abs, the same order as the working NF4 residuals — so nothing
downstream would have objected. The experiment would have reported a
number and measured nothing. That is the specific failure this session
has been guarding against, and it is now gated in code.

**Not done at the time:** AWQ's layout. Recorded here because the plan I
wrote was wrong in a useful way: the next step I proposed was installing
`gptqmodel` or `autoawq` and diffing against its unpack. Reading the
unpack was enough, and it cost nothing — §16.

---

## 15. Item 3 — Phi-3-mini

Downloaded 7.1 GB, extracted residuals, ran exp1/2/3/6/7.

**Blocked first by an incompatible remote implementation.** Phi-3 ships
bundled modelling code that reads `rope_scaling['type']`, but current
configs emit `rope_type`:

```
File ".../modeling_phi3.py", line 296, in _init_rope
    scaling_type = self.config.rope_scaling["type"]
KeyError: 'type'
```

`load_model_pair` hardcoded `trust_remote_code=True`, so it always pulled
the stale bundled code even though transformers has had a native Phi3
implementation for several versions.

**Decision:** added `trust_remote_code: bool = True` to
`load_model_pair` (default unchanged, so no other model is affected) and
passed `False` for Phi-3, which uses the native implementation.

**Residuals:** 32 layers, mean abs **0.002933**, tightly clustered
(0.002610–0.003106). Worth noting the contrast with Qwen2.5-7B, whose
layers span 0.0012–0.0130 in the same model — Phi-3's narrow spread is
what a healthy residual distribution looks like, and it makes the 7B
outlier easier to recognise as abnormal in future.

Cached all 32 layers so later runs are cache-only.

**Results — all five cells pass:**

| | exp1 | exp2 | exp3 | exp6 | exp7 |
|---|---|---|---|---|---|
| Phi-3-mini-4k-instruct | PASS | **PASS** | PASS | PASS | PASS |

Phi-3 is only the second model to pass exp2, alongside Mistral-7B and
Gemma-9B.

---

## 16. Item 2 — Exp9 AWQ: what the gate was actually comparing

### The layout was a guess only until it was read from source

§14 closed with "install `gptqmodel`/`autoawq` and diff against its unpack,
rather than continuing to guess". The useful version of that turned out to
be reading the unpack, not installing it. `awq/utils/packing_utils.py`:

```
AWQ_ORDER         = [0, 2, 4, 6, 1, 3, 5, 7]
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
```

`unpack_awq` fills column-wise, nibble `i` receiving element `order_map[i]`
of each group of 8, so element `k` lands in nibble
`AWQ_NIBBLE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)` — an even/odd interleave,
which is exactly why the 24-variant rotation search could not reach it.
`dequantize_gemm` then does `repeat_interleave(group_size)` on scales and
zeros and computes `(iweight - izeros) * scales`.

`dequantize_awq_layer` is equivalent to that, line for line. The layout is
now **read**, not inferred. But that only moves the question: if the order
is right, why did the gate say 0.2343, and why did 52 of 252 modules still
fail after the fix?

### AWQ does not quantize the weight you think it does

AWQ scales channels and folds the inverse into whatever precedes the linear
layer, so the checkpoint holds `W / c` while the LayerNorm holds `~c`. The
network is numerically unchanged. Measured on layer 0 of the real
checkpoint, `c` from `W ≈ c · dequantized`:

| module | factor `c` | the LayerNorm that carries `1/c` | measured LN ratio (awq/ref) |
|---|---|---|---|
| `q_proj` | 1.322 | `input_layernorm` | **1.340** |
| `gate_proj` | 1.781 | `post_attention_layernorm` | **1.790** |
| `down_proj` | 1.882 | *(none — absorbed into `up_proj`)* | — |

The first two rows are the point: two independently measured quantities,
the weight factor and the LayerNorm ratio, agree to ~1%. That is what makes
the factor an absorbed scale rather than a bug in our unpacking. The third
row is the same mechanism one stage later — AWQ scales the intermediate
channel by scaling `up_proj`'s output and compensating in `down_proj`'s
input, because a LayerNorm does not precede `down_proj`.

So the fair comparison is against `W · s` with `s` a per-channel factor,
not against `W`. **The thresholds stay at 0.95 / 0.5.** Only the basis of
the comparison changes, and the correction is deliberately narrow: a column
fit for every module, plus a row fit for `up_proj` alone (its output
channels are the intermediate channel). A free rescale would have been a
real risk, so the gate was checked against a control — see below.

### The layer-2 anomaly, timeboxed and then stopped

`model.layers.2.mlp.up_proj` dequantizes to std 0.419 against a reference
std of 0.014 (30×, raw correlation 0.171), and `model.layers.2.mlp.down_proj`
quantizes 8959 of 11008 input rows **exactly to the zero point** (81%, in
only 86 distinct values — one per group, i.e. `q == z`). Its stored `scales`
are inconsistent with its own indices: 92.8% of the log-ratio variance is
explained by a row × column factor, so the values are right in distribution
and wrong in slot.

What the checkpoint does *not* do is break:

| reconstruction | perplexity |
|---|---|
| reference model | 33.59 |
| AWQ, as published | **35.25** |
| AWQ with layer-2 `up_proj` zeroed | 40.65 |
| AWQ with layer-2 `up_proj` set to the reference weight | **inf** |
| AWQ with layer-2 `up_proj` + `down_proj` both set to reference | 6554.5 |

The `inf` row is the informative one. The *correct* reference weight is the
one that overflows, because layer 2's `down_proj` is co-adapted to its own
`up_proj`; a clean weight pair wedged into half of an already-scaled pair
does not make the layer correct. Traced to fp16: layer 2's MLP output
reaches max 3252 / sd 4.18 where every other layer sits near sd 0.4, the
residual stream jumps ~8× at layer 2 and stays elevated, and RMSNorm
normalizes it away downstream — which is why perplexity only moves 5%.

Verdict: a **checkpoint-level defect in 3 of 252 modules**, not a
dequantizer defect. Timeboxed, documented, not resolved by moving a
threshold.

### The gate, and the control that keeps it honest

Run over all 252 modules of the real checkpoint through the shipped
`verify_dequantization`, plus GPTQ as the unchanged reference path:

| condition | modules passing 0.95 / 0.5 |
|---|---|
| AWQ, correct order, raw comparison | 200 / 252 |
| AWQ, correct order, absorbed scale removed | **249 / 252** |
| AWQ, **sequential** order, *same* correction | **0 / 252** |
| GPTQ (path untouched by this change) | **252 / 252** |

The third row is the one that matters. Removing a per-channel scale is the
kind of correction that could quietly paper over a layout bug — a wrong
nibble order rearranges values *within* each group of eight, and a
sufficiently generous fit might have absorbed it. It does not: 0 of 252
survive, with correlations around 0.20–0.28. The gate keeps its teeth.

The three that fail the corrected gate, and are reported as failures:

| module | correlation | residual ratio |
|---|---|---|
| `model.layers.2.mlp.down_proj` | 0.9337 | 0.358 |
| `model.layers.33.self_attn.v_proj` | 0.9147 | 0.404 |
| `model.layers.32.self_attn.v_proj` | 0.9211 | 0.389 |

One latent bug fell out of writing this: the gate compared with
`correlation < min_correlation`, and `nan < 0.95` is **False** — a constant
or all-zero dequantization would have sailed through both thresholds and
been reported as verified. Both comparisons are now written as
`not (x >= threshold)`, and there is a test that asserts an all-zero
dequantization fails.

### Result

| target | status | evidence |
|---|---|---|
| Qwen2.5-3B-Instruct-GPTQ-Int4 | **PASS** | BER **0.0**, 10,256/10,256 bits, corr 0.9903, ratio 0.1402, 36/36 layers |
| Qwen2.5-3B-Instruct-AWQ | **PASS** | BER **0.0**, 10,256/10,256 bits, corr 0.9941 (raw 0.9890), ratio 0.1093, **35/36 layers** (layer 2 excluded by name in `metrics.layers_excluded`) |

**The claim this supports, stated narrowly:** a payload round-trips at
BER 0.0 through three 4-bit checkpoint formats — NF4, GPTQ, AWQ — each read
by a format-specific dequantizer and each verified against its own
FP16/bf16 reference *before* any residual is computed.

What that does **not** say: one model (Qwen2.5-3B-Instruct), one payload
size (10,256 bits), one module type (`mlp.down_proj`), and a **clean**
channel — no noise, no patch, no adversary. Robustness and detectability
were measured for NF4's sign embedding only (§7, §12), never for GPTQ or
AWQ. "NES works beyond NF4" as an unqualified sentence is broader than the
evidence; this is the version that survives.

---

## 17. Final state

**Coverage: 35 PASS, 6 FAIL, 0 NOT_RUN, 0 ERROR.**
Every registry model now has cells, and no cell is left unrun. 9/9
consistency checks pass, the test suite runs clean (38 tests, OK), and
`nes-llm/claim_audit.py` re-derives 64 MEASURED claims from the
artifacts on disk rather than from this prose — its failures, when it
has any, are the findings below.

Of the 41 cells: 33 sit on the 7-model NF4 grid, 2 are the quantized
checkpoints outside it (Exp9's GPTQ and AWQ targets), and 6 are FAIL.
"7 models covered, 0 errors" describes only the NF4 grid — the NF4
path (`bitsandbytes`, `nf4`, group 64) is what those 33 cells measure.
The two non-NF4 cells are separate evidence from separate dequantizers
and are listed separately below; neither is an NF4 result, and no NF4
cell was reused to produce them.

```
model                                      exp1   exp2   exp3   exp6   exp7
Qwen/Qwen2.5-3B                            PASS   FAIL   PASS   PASS   PASS
Qwen/Qwen2.5-7B                            PASS   FAIL   PASS   PASS   PASS
TinyLlama/TinyLlama-1.1B-Chat-v1.0         PASS   FAIL   PASS   PASS   PASS
google/gemma-2-9b                          PASS   PASS   PASS   PASS   PASS
meta-llama/Llama-3.1-8B                    PASS   FAIL   PASS   PASS   PASS
microsoft/Phi-3-mini-4k-instruct           PASS   PASS   PASS   PASS   PASS
mistralai/Mistral-7B-v0.3                  PASS   PASS   PASS   PASS   PASS
```

Plus, outside the NF4 grid:

| | result |
|---|---|
| exp8 cross-model | FAIL (driven by the neural detector) |
| exp9 GPTQ | **PASS, BER 0.0**, dequant corr 0.9903, 36/36 layers |
| exp9 AWQ | **PASS, BER 0.0**, dequant corr 0.9941 (raw 0.9890), 35/36 layers |
| exp10/exp11/exp12 strategies | LWE grid width passes both gates on **5 of 5** measured models |

### Claim audit findings

Four claims did not survive the final audit, and **all four are now
closed** — three by rewording the claim, one by producing the artifact
that should have existed. They are recorded rather than quietly
corrected, because a document that only ever gets righter is not an
audit. The audit is `nes-llm/claim_audit.py`: 64 checks, re-derived from
`results/*.json`, **64/64 passing at the time of writing**.

**1. exp12 coverage — closed by re-running it.** §7 and its commit
message state *5 of 5 measured models pass both gates*, with a six-row
table. `results/exp12_lwe_cross_model.json` held **four entries, three
measured**: TinyLlama SKIPPED (0-layer cache), then Qwen2.5-7B,
Llama-3.1-8B and gemma-2-9b. Qwen2.5-3B and Mistral-7B were not in the
file at all — not SKIPPED, absent — and no archived version of that
artifact ever contained them, although `cache_status` reports both
caches complete. Two independent records disagreed and only one was an
artifact.

The claim was never contradicted: every model that *was* in the file
passed both gates at BER 0.0000 / detector 50.00%, which is also what
the 5-row table reported. It was a coverage gap, not a wrong number.
exp12 was re-run across all six cached models with no `--models`
filter, and the artifact now records **five measured, one skipped** at
the same BER and detector accuracy. **5/5 is citable again** — because
the file says so, not because the table did.

**2. exp2's count — closed by re-running the calibration.** §4.1 said
*"1/7 pass … Six fail"* and both documents said exp2 fails on **6 of 7
models**. The manifest says **4 of 7 fail** and **3 pass**: gemma-2-9b
at 88.1% of layers above threshold, Phi-3-mini at 100%, Mistral-7B at
81.2%; Qwen2.5-3B at 11.1%, Qwen2.5-7B at 75%, TinyLlama at 0%,
Llama-3.1-8B at 68.8%. Neither earlier figure came from the cells.

There were two figures because there are two measurements.
`exp2_criterion_calibration.json` reads the legacy
`residual_profile_*.json` files — the suite's models plus `gemma-2-2b`,
eight profiles — and the copy on disk had been written between 11:22
and 11:34 on Oct 2, before gemma-2-9b (11:47) and Llama-3.1-8B (11:34)
were re-run. Five of its seven rows matched today's profiles exactly;
those two did not, by an order of magnitude (0.000382 vs 0.003605,
0.000879 vs 0.022642). Its `finding` string compounded this: it said
*"Zero of 7 …"* while `models_passing` in the same object said `1`,
because the word was hard-coded rather than counted.

Both halves were fixed rather than edited around. The generator now
counts and labels the FP4 comparison for what it is — one model, five
probed layers. The artifact was regenerated from the profiles as they
stand and reports **3 of 8**, agreeing with exp2 on **every** shared
model, gemma-2-9b and Llama-3.1-8B included — the pair that exposed the
staleness. The two `claim_audit.py` checks that held this open pass.

**3. exp6 stated as "BER 0 at σ=0.001" — closed.** Reworded in three
places in `RESEARCH_PLAN.md`. Six of seven models are exactly 0.0;
Mistral-7B is **0.00115** at σ=0.001 and **0.01074** at σ=0.002. Both
numbers are comfortably inside the 0.02 / 0.10 gates, so every cell
still reads PASS and no gate moved. What was wrong was the claim, not
the result: "BER 0" turned a passing measurement into an absolute, and
the seventh model was the one that made it false. The audit now checks
the gate *and* the 6-of-7 count.

**4. "7 models covered, 0 errors" — closed.** Retired as misleading.
Those 33 cells are the **NF4** grid (`bitsandbytes`, nf4, group 64). The
GPTQ and AWQ cells sit on separate checkpoints with separate
dequantizers; no NF4 cell was reused to produce them. The current figure
is 9 model ids, 35 PASS / 6 FAIL / 0 NOT_RUN, with the format split
stated wherever the number appears.

The audit's lesson is the session's own: the number that looks finished
is the one worth re-reading. "5 of 5" was written from console output,
and console output is not an artifact. "BER 0" was written from the six
models that were zero, and an average is not a maximum. exp2's "6 of 7"
was written from a diagnostic that measured a different set of files
than the cells it was standing in for. All four have the same shape — a
claim widened one word past its evidence, or lifted from a source that
was never the record — so `claim_audit.py` checks the counts as well as
the values, and names the models behind them.

### The failures, and why none were "fixed"

| failure | verdict |
|---|---|
| exp2 (4 of 7 models) | criterion is quantization-format dependent (FP4 passes, NF4 fails); threshold left unchanged — a research decision, not a code fix |
| exp7 neural, sign embedding | 70.5% vs 55%; proved structural across a 100× alpha and 5× gamma sweep. LWE at grid width 0.010 reaches 50.00% and passes on **5 of 5** measured models |
| exp8 | FAIL is correct: it aggregates and inherits the neural FAIL |

### Open, deliberately not done

- **Three AWQ modules excluded from Exp9's residual set.**
  `model.layers.2.mlp.down_proj` (corr 0.9337), `model.layers.32` and
  `model.layers.33` `self_attn.v_proj` (0.9211 / 0.9147) fail the
  verification gate after the absorbed scale is removed. They are named
  in `metrics.layers_excluded` rather than silently embedded into, and
  the two `v_proj` ones are outside Exp9's module (`mlp.down_proj`)
  entirely. Investigating them further was timeboxed and stopped — see
  §16.
- **Robustness and detectability for non-NF4 formats.** Exp9 measures a
  *clean* channel: no noise, no patch, no adversary, one payload size
  (10,256 bits), one module type (`mlp.down_proj`). Every σ-sweep and
  every detector number in §7/§12 belongs to NF4 sign embedding.
- **Cross-scheme detector.** Every detectability number uses a detector
  trained against the same scheme it tests. A detector trained on sign
  and tested on LWE is the stronger experiment and has not been run.
- **LWE naming.** There is no lattice and no SIS/LWE instance, so the
  post-quantum claim in `lwe_strategy.py`'s docstring is unsupported.
  Flagged, not renamed — renaming a claimed contribution is the
  authors' call.
- **`bits_embedded` shortfall.** 48,256 embedded against 50,000
  requested, a consistent ~3.5% gap. Not chased.
- **Two residual implementations.** `loader.py` (3-value return) and
  `model_loader.py` (dict return) still coexist; legacy `scripts/exp*.py`
  import the former.
- **Neural strategy.** A memory-safe `train_sampled()` is written but
  never run, so the concern that it converges to the same sign-based
  solution (§7.5) is untested.

### A note on the session's shape

Six of the bugs found here produced plausible-looking wrong answers
rather than crashes: BER never measured, Exp9 measuring NF4 under a GPTQ
label, a GPTQ nibble axis that transposed silently, an alpha that was a
no-op, a residual profiler biased toward small values, and two studies
that scored exactly 50% and looked like security wins. The verification
gates that now exist — real bit comparison, dequantization correlation,
`INVALID STUDY` on an all-chance sweep, cover-free round trip — each came
directly from one of those.

## 18. Recent updates — the audit became a script

Written after §17, because the work that followed it was not more
measurement but more *reading*: every MEASURED claim in
`RESEARCH_PLAN.md` was checked against the file it claims to come from,
and four did not match. Nothing that was measured turned out wrong.
Four sentences were.

### 18.1 `claim_audit.py`

`nes-llm/claim_audit.py` makes this section executable: 64 checks, each
re-deriving one MEASURED claim from `results/*.json`. It exits non-zero
on any claim it cannot verify — a number it cannot find is UNVERIFIED,
never assumed true — and it reads its gate values from
`experiment_registry.THRESHOLDS`, so moving a threshold breaks the audit
as well as the experiment.

It checks **counts** as well as values: how many models were measured,
which layers were compared, which entries are in the file. That is the
finding rather than a refinement of it. All four failures were counts
that had drifted from their source while every value underneath stayed
correct — which is precisely why reading the numbers again found nothing
and re-deriving them did.

### 18.2 The four findings

Full detail is in §17; this is the index.

| # | claim as written | what the artifact said | closed by |
|---|---|---|---|
| 1 | "5 of 5 measured models pass both gates" (§7, and its commit message) | 4 entries, 3 measured — Qwen2.5-3B and Mistral-7B absent, not skipped | re-running exp12 across all 6 cached models: 5 measured, 1 skipped, BER 0.0000 / detector 50.00% on every measured row |
| 2 | exp2 "fails on 6 of 7 models"; §4.1 "1/7 pass … Six fail" | manifest: 4 of 7 fail, 3 pass. The calibration said `models_passing: 1` while its own `finding` said "Zero of 7" | reworded to the cells; the generator now counts, and the artifact was regenerated — 3 of 8, agreeing with exp2 on every shared model |
| 3 | "Sign robustness BER 0 at σ=0.001" (three places) | 6 of 7 are exactly 0.0; Mistral-7B is 0.00115 and 0.01074 — both inside the 0.02 / 0.10 gates | reworded; no cell changed and no gate moved |
| 4 | "7 models covered, 0 errors" | those 33 cells are the NF4 grid; GPTQ and AWQ are separate checkpoints with separate dequantizers | reworded to 9 model ids, 35/6/0, format split stated wherever the number appears |

Findings 1 and 2 were closed by **producing the artifact that should
have existed**. Findings 3 and 4 were closed by narrowing a claim to
what was already measured. None was closed by changing a result:
exp7_neural still fails at 70.5%, exp8 still inherits it, exp2 still
fails on 4 models, and every threshold is unchanged — `claim_audit.py`
asserts that last point itself.

The shape is the same in all four: a claim one word wider than its
evidence. "5 of 5" written from console output; "6 of 7" lifted from a
diagnostic that measured a different set of files; "BER 0" an average
standing in for a maximum; "7 models" a grid count wearing a suite's
name.

### 18.3 Commits

| hash | what |
|---|---|
| `3cbe536` | AWQ nibble order from AutoAWQ source, absorbed-scale gate, thresholds moved into the registry |
| `838e7c9` | exp9 verifies every layer and names the ones that fail |
| `5263fab` | results regenerated: 35 PASS / 6 FAIL / 0 NOT_RUN |
| `7132f5e` | §16/§17 written, `RESEARCH_PLAN` restructured into Phases A–D, claim-audit section added |
| `6b8f5b8` | `claim_audit.py` created |
| `e88c473` | finding 2 (exp2's counts) corrected in both documents |
| `f69b929` | the suite's two import errors repaired → 38 tests OK |
| `34a41e4` | `nes-llm/README.md` filled (it was tracked and empty) |
| `4d149e8` | 19 tracked `.pyc` files untracked and ignored |
| `a688eba` | verification commands and state recorded in the documents |
| `62083cc` | findings 1 and 2 closed by re-running exp12 and the calibration |

### 18.4 What was repaired along the way

- **The test suite could not finish.** `unittest discover` reported
  `FAILED (errors=2)` on every run, and neither error was an assertion.
  `test_integration` imported pytest — not installed in this
  environment — and never used it. `test_keyed_embedding` contained no
  tests at all: a scratch script whose last statement decoded a
  *wrong-key* stream, which raises, so importing the file was an error
  rather than a run. Its intent is now three assertions, and a
  wrong-key decode is treated as non-recovery rather than pinned to a
  particular exception. 38 tests, OK.
- **`nes-llm/README.md` was tracked and zero bytes.** The package's
  entry point was a blank page while the instructions lived in a 32KB
  handoff written for a previous agent. It now carries layout, how to
  run, the verification commands, and an experiment table with every
  gate and every state, FAILs included.
- **19 `.pyc` files were in the index.** A committed byte-code file
  records which source produced it, so it goes stale the moment the
  source does and disagrees with it afterwards — the same failure mode
  as the four findings, in a file nobody reads. Untracked and ignored.

### 18.5 What verification now runs

```bash
cd nes-llm
python run_nes_experiments.py --audit   # cell states   35 PASS / 6 FAIL / 0 NOT_RUN
python check_consistency.py             # cross-artifact 9/9
python claim_audit.py                   # MEASURED claims 64/64
python -m unittest discover -s tests -p 'test_*.py'   # 38 tests, OK
```

The generated report carries the same three commands in its §9, so
anyone holding the report can re-derive it rather than trust it.

### 18.6 State at this commit

35 PASS / 6 FAIL / 0 NOT_RUN / 0 ERROR; 64/64 claims verified; 9/9
consistency checks; 38 tests. All four findings closed, and the warning
marker is now absent from both documents — which is what makes it worth
keeping as a marker rather than deleting: its presence in either
document means a claim is open. And **no result, threshold or verdict
changed by any of it**. The audit altered claims about the work, not the
work, which is the only outcome that should have been possible and the
one worth stating: an audit that starts "improving" numbers is a
conflict of interest with a progress bar.

---

## 19. Phase B begins — W4.2 keyless recovery (exp13)

First item of the suggested order (`RESEARCH_PLAN` §7 #1). §3 W4.2 asks
whether an attacker could recover the LWE grid scale from the weights
alone and read bits without the key, and calls it *"a potential break of
the security property and cheap to test."* It is a break.

### 19.1 What was built

`nes-llm/src/experiments/exp13_keyless_recovery.py` — a standalone
module in the exp12 mould (artifact + claim-audit checks, not a manifest
cell, so the coverage numbers are untouched). The gate lives where
ground rule 2 says it must, `THRESHOLDS["exp13"]`:

- `min_keyless_ber ≥ 0.5` — an attacker holding only the released model
  must be at chance reading the bitstream, and
- `min_width_search_relative_error ≥ 0.01` — the scale must not be
  locatable to within 1% from the weights alone.

Both must hold; either failing records FAIL. Attacker model is
Kerckhoffs: the released model and the public source, nothing else — no
key, no cover, no carrier map, no payload parameters. One scope
statement that the artifact carries and any citation must too: the
payload is AES-256-GCM ciphertext (the embedder encrypts before
embedding), so *message* confidentiality is not under test and is not
claimed broken. What fails is the strategy's own claim that the hidden
channel is key-gated.

### 19.2 The measurement

| tier | holds | result |
|---|---|---|
| shipped path | — | width **0.010, one distinct value** over 36 layers × 6 keys; `keyed_branch_active: false` — HMAC is computed and discarded |
| designed control | — | forcing `grid_width=None`: **0/36 layers** key-dependent, spread 0.0 — the `min_magnitude` floor dominates the key term everywhere |
| key invariance | any key | 6 keys decode identical streams, max pairwise **BER 0.0** |
| **phase attacker** | weights + public constant | **precision 1.0, recall 1.0, stream BER 0.0 over all 10,256 bits**; clean control yields **0** candidate positions at 1e-5 / 1e-6 / 1e-7 |
| Kerckhoffs re-run | + public QACI pipeline + payload size | precision ≈ 0.65, 0–4 of 36 layer allocations match, BER ≈ 0.49 (chance) — shifted allocations misalign the stream |
| width search | weights only | spike on layer 0 (1 layer scanned); argmax **ties {0.002, 0.010} at 153/153**; `5w` partial (68 — carriers whose interval index ≡ 2 mod 5), `w/2` and `2w` score 0; detection survives only ±1e-7 (±1e-6 partial, ±1e-5 dead) while **decoding tolerates ±1% at BER 0**, and the tied `w/5` decodes at BER 0.0 |

Gate: **FAIL** — `min_keyless_ber` failed (0.0 against 0.5). The width
condition "passes" at 0.8 only because the argmax is `w/5`, a
parity-equivalent subdivision; the artifact records it as a tie
(`true_width_tied_for_best: true`), not as concealment. Reading the 0.8
alone would say the search missed, when search, detection and decode
all land on the same lattice.

All three security claims in the strategy docstring
(`lwe_strategy.py:13-17`) are recorded **REFUTED**, each with its
falsification criterion in the artifact: the spacing (public constant,
keyed branch unreachable, 0/36 in the designed formula), the parity
mapping (six keys, identical bits), the positions (precision/recall 1.0
from the weights alone).

**What this does and does not break.** The LWE channel is readable by
anyone with the model and the source. The message stays AES ciphertext —
nothing here touches AES. The production default (sign) is unaffected;
this is specific to LWE. And the fix direction is visible in the same
numbers: a genuinely secret width would deny the phase attacker its
lattice (detection dies at 1e-5), so keying the width would matter —
the defect is that the shipped path never keys it.

### 19.3 Two side findings

- **exp11/exp12's width-forcing hooks are dead.** Measured:
  `patched(0.002) → 0.01`, `patched(0.05) → 0.01` — the constructor
  substitutes `DEFAULT_GRID_WIDTH` before the patched `min_magnitude`
  can reach `_derive_interval_width`. exp11's own artifact predates the
  constant (its rows genuinely vary: BER 0.5857 → 0.0127 → 0.0 → 0.0,
  detector 0.706 at 0.05), so the published frontier stands as
  measured. But a future re-run with `--grid-width` other than 0.010
  would silently measure the default. exp12's re-run this session used
  0.010 and is unaffected.
- **Re-running the public pipeline is the weaker attack.** The
  embedding changes layer statistics enough to shift the Hamilton
  allocation (matching on 4/36 layers in one run, 0/36 in another),
  which misaligns the concatenated stream to chance. The physical
  center signature needs no allocation at all — which is why the phase
  tier, not the "knows everything public" tier, is the one that reads
  the channel.

### 19.4 Development runs — recorded, not hidden

Three runs, because two of them taught something:

1. The width search reported "no spike found" after scanning all 36
   layers. A bug, not a measurement: the break condition required
   `median > 0`, but background hits at this tolerance are exactly 0,
   so `3 × median` was never reachable. Fixed to
   `best > max(3·median, median + 5)`.
2. The search then found the spike but reported argmax 0.002 against a
   true 0.010 with no explanation — which reads as a failed search
   unless the tie scores are on the record. Added
   `argmax_candidates`, `true_width_tied_for_best` and
   `scores_at_published_widths`, plus `w/5` and `5w` points in the
   decode curve.
3. Final.

The Kerckhoffs tier varies run to run — a fresh AES key per run means
different ciphertext bits, hence different stego layout (precision
0.645 → 0.642 → 0.653, BER 0.487 → 0.498 → 0.494). `claim_audit`
range-checks it as "chance" (0.4–0.6) instead of pinning a digit;
everything else in the artifact is deterministic and pinned.

### 19.5 Verification

```bash
cd nes-llm
../.venv/bin/python claim_audit.py                    # 64/64
../.venv/bin/python check_consistency.py              # 9/9
../.venv/bin/python -m unittest discover -s tests -p 'test_*.py'   # 38 OK
../.venv/bin/python -m src.experiments.exp13_keyless_recovery      # re-runs
```

State: 64/64 claims, 9/9
consistency, 38 tests, manifest unchanged at 35 PASS / 6 FAIL /
0 NOT_RUN / 0 ERROR — exp13 sits outside the manifest grid, and its
FAIL is a gate verdict recorded in the artifact and the claim audit,
not a seventh manifest cell. The six manifest FAILs are still exp2 ×4,
exp7_neural and exp8.

## 20. W3.2 — the blind-patch adversary (exp14)

Suggested order item 2 (`RESEARCH_PLAN` §7). §3 W3.2 asks what happens
when the detector stops being handed the answer: both exp7_neural patch
classes are cut *at carrier positions*, and §0's load-bearing
"Not established" #3 says so in as many words. Carriers are ~0.001% of
positions, so a blind adversary's patches should almost never contain
signal — "could still catch an aggregate distribution shift" was the
plan's hypothesis.

### 20.1 What was built

`nes-llm/src/experiments/exp14_blind_patch_detector.py` — standalone,
same mould as exp13 (artifact + claim-audit checks, not a manifest
cell). Gate in `THRESHOLDS["exp14"]`: `max_blind_detector_accuracy ≤
0.55`, the *same* self-chosen 55% number exp7_neural already uses —
reused, not invented. Only the blind arm is gated.

Two arms cut from the same 12 embeddings (48 patch pairs each, 576 per
arm — above the suite's 400-pair floor):

- **blind** — patch position drawn uniformly at random, independent of
  the carrier set. The adversary knows nothing about placement;
- **control** — exp7's rule, an aligned patch containing a carrier.
  Positive control: if this arm does not clearly exceed the gate, the
  blind number carries no information and the artifact says so.

Everything else is exp7's code *imported unchanged* — `Detector`,
`ResidualDataset`, `evaluate`, `split_by_sample_id`, 30 epochs, batch
32, lr 1e-4, seed 42 — so the only variable between the arms, and
against exp7 itself, is where patches are cut. One deliberate
tightening: the split is **by embedding** (`sample_id` = embed index,
9 train / 3 test), so no stego tensor appears on both sides of
train/test. exp7 split by pair while every pair came from the same
model's structure.

### 20.2 The measurement

| arm | placement | accuracy | detail |
|---|---|---|---|
| **blind** | uniform random | **0.5000** | confusion matrix **tp=0, fp=0, fn=144, tn=144** — a constant all-clean predictor; the detector learned nothing |
| control | carrier-centered (exp7 rule) | **0.6875** | 101/144 stego caught — clears the 55% gate and sits near exp7's 70.5% |
| blind exposure | — | **4/576** positions contained any carrier (0.69%) | even those 4 did not produce a single true positive |

Gate: **PASS** (0.5000 ≤ 0.55). Split: 432 train / 144 test pairs per
arm. Message length varied by one byte per embedding (total_bits
9984 → 10072) so the Hamilton allocation shifts and the control arm is
not carved from one fixed carrier set.

**Reading.** The control clears the gate, so the pipeline is validated
and the blind result is informative: exp7's 70.5% FAIL is
**carrier-conditioned**. An adversary without carrier positions lands on
exactly chance — not "barely above", but a collapsed constant
predictor. §0's "the detector is handed the answer" is now measured:
handing it the answer is what made the difference (50.0% vs 68.75%,
same embeddings, same detector, same training loop).

### 20.3 Side finding — carriers cluster

The control patches average **~45 carriers each** (per-embedding sums
1673–2829 over 48 patches), while uniform positions average ~0.05.
QACI selects by magnitude, so carriers concentrate in a few
high-magnitude aligned blocks rather than spreading uniformly. That
explains both numbers at once: a blind patch hits one of those blocks
4 times in 576, and a carrier-centered patch is not "one carrier plus
4095 cover" but a block dense with them — which is exactly why the
carrier-centered detector works so well and why the blind one never
sees the signal.

### 20.4 Verification

One run, exit 0 — no development runs this time; the seeds make both
arms and the split reproducible.

```bash
cd nes-llm
../.venv/bin/python claim_audit.py                    # 64/64
../.venv/bin/python check_consistency.py              # 9/9
../.venv/bin/python -m unittest discover -s tests -p 'test_*.py'   # 38 OK
../.venv/bin/python -m src.experiments.exp14_blind_patch_detector  # re-runs
```

State: 64/64 claims, 9/9 consistency, 38 tests, manifest unchanged at
35 PASS / 6 FAIL / 0 NOT_RUN / 0 ERROR — exp14, like exp13, sits
outside the manifest grid, and its PASS is a gate verdict in the
artifact and the claim audit, not a 36th cell.

Scope note for any citation: this measures the **production sign
strategy on one model** (Qwen2.5-3B), and only patch-placement
sensitivity. It does not license "undetectable" — exp7_neural's 70.5%
FAIL stands for an adversary that knows placement, and W3.3/W3.4
(stronger and multi-model detectors) are untouched.

## 21. W2 — LWE fidelity, three-way PPL (exp15)

Suggested order item 3 (`RESEARCH_PLAN` §7). §3 W2 called fidelity
"the biggest hole": *LWE perplexity is unmeasured anywhere.
Undetectability is worthless if the model is damaged.* The prior was
a perturbation table (LWE moves weights 9.4× less than sign) with
"expect is not a result" attached. This is the result.

### 21.1 What was built

`nes-llm/src/experiments/exp15_lwe_fidelity.py` — standalone, exp13/14
mould. It runs **exp5's three-way protocol unchanged** — NF4 baseline,
reconstruction control (`W_NF4 + R_original`), embedded
(`W_NF4 + R_embedded`); only control → embedded is attributable to the
payload — through exp5's own `FidelityValidator`, wikitext-2, 200
texts, max_length 512, batch 4, payload 50,000 bits, message
`"A"*6000`. The gate is `THRESHOLDS["exp15"].max_ppl_degradation_pct
= 2.0` — exp5's threshold, reused, not a new number. One process per
model (the memory rule), one artifact each.

Two things exp5 did not have: a **sign re-verification arm** on
Qwen2.5-3B (the plan explicitly asks to check the recorded number
reproduces), and a per-model artifact rather than a manifest cell.

### 21.2 The measurement

| model | baseline | control | LWE embedded | **LWE Δ vs control** | gate 2% |
|---|---|---|---|---|---|
| Qwen2.5-3B | 12.4707 | 11.3494 | 11.3502 | **+0.0077%** | **PASS** |
| gemma-2-2b | 17.4481 | 16.5286 | 16.5369 | **+0.0501%** | **PASS** |

Reconstruction alone moves PPL by **−8.99%** (Qwen) and **−5.27%**
(gemma) relative to the NF4 baseline — reported separately, never
folded into the payload's number, exactly as exp5's attribution rule
requires. The absolute delta vs baseline is therefore ≈ −8.98% /
−5.22%, and reading *that* as "embedding damage" would be the
mis-attribution the three-way protocol exists to prevent.

**The prior was right, and now it is measured:** LWE's
embedding-specific cost is 0.008–0.050% — one to two orders of
magnitude under the 2% gate, on two architectures (qwen, gemma).

### 21.3 Sign re-verification — reproduced exactly where it matters

Arms 1 and 2 re-derived exp5's recorded values **to all printed
digits**: baseline 12.4707 (recorded 12.4707), control 11.3494
(recorded 11.3494). The protocol itself is reproducible.

Arm 3 (sign embedded) differs: re-run **+0.0529%** vs recorded
**−0.0053%** — a 0.058-point gap. Both are orders of magnitude under
the 2% gate, so the *verdict* reproduces; the digit does not. The
mechanism is already on the record from exp13 (§19.4): a fresh AES
key per run means different ciphertext bits, hence different sign
values at the ~48k carriers, hence PPL at the 0.05% level. The
artifact stores both numbers and their point difference
(`delta_point_difference: 0.0582`) rather than quoting whichever one
looks better. `claim_audit` pins baseline/control exactly and
range-checks the sign delta (< 0.1 points) instead of pinning a digit
it cannot pin honestly.

### 21.4 Side finding — Phi-3-mini no longer loads

First choice for the second model was Phi-3-mini (in the grid, cache
complete). The model pair **no longer constructs under transformers
5.16.1**: the checkpoint's remote modeling code reads
`config.rope_scaling["type"]` in `_init_rope` and the checkpoint's
`rope_scaling` dict has no `"type"` → `KeyError` at construction,
before any of this experiment's code runs. Patching the config (adding
`"type"` or nulling `rope_scaling`) would change RoPE behaviour and
silently corrupt the very PPL being measured, so it was not done —
the loader was left alone and gemma-2-2b (complete 26-layer cache,
constructs cleanly) took the second slot instead.

This is an environment fact with a blast radius beyond exp15: **any
`--force` re-run of a Phi-3 manifest cell would now crash the same
way.** Recorded here rather than worked around; a fix belongs in an
environment change (transformers pin or checkpoint config), not in an
experiment.

### 21.5 Verification

```bash
cd nes-llm
../.venv/bin/python claim_audit.py                    # 64/64 (5 new)
../.venv/bin/python check_consistency.py              # 9/9
../.venv/bin/python -m unittest discover -s tests -p 'test_*.py'   # 38 OK
../.venv/bin/python -m src.experiments.exp15_lwe_fidelity \
    --model Qwen/Qwen2.5-3B                           # re-runs
```

State: 64/64 claims, 9/9 consistency, 38 tests, manifest unchanged at
35 PASS / 6 FAIL / 0 NOT_RUN / 0 ERROR — exp15 is the third standalone
gate outside the manifest grid. Scope: LWE's *fidelity* is now
established on two models; its key-gating remains refuted (exp13) and
its detectability numbers (exp10–12) unchanged. W2's two deliverables
— "PPL for LWE on ≥2 models" and "re-verify sign's number
reproduces" — are both closed.

## 22. W3.1 — cross-scheme detector (exp16)

Suggested order item 4 (`RESEARCH_PLAN` §7). §0 "Not established" #2
said every detectability number uses a detector trained against the
scheme it tests, and §3 calls cross-scheme *"the claim's main
weakness."* This is the measurement.

### 22.1 What was built

`nes-llm/src/experiments/exp16_cross_scheme_detector.py` — standalone,
same mould as exp13–15. Two datasets, four evaluations:

| train | test | role |
|---|---|---|
| sign | sign | within-scheme control |
| sign | LWE | **cross** |
| LWE | LWE | within-scheme control |
| LWE | sign | **cross** |

Design decisions worth stating:

- **Placement is carrier-centred in both schemes**, held fixed —
  exp14 proved blind placement is chance, so placement cannot be the
  variable. Only the scheme varies. That isolates transfer.
- 6 embeddings per scheme × 72 pairs = **432 pairs per scheme** (≥
  the 400-pair floor), split **by embedding** (4 train / 2 test) so
  no stego tensor is on both sides.
- Detector, epochs, batch, lr, seed imported from exp7's module —
  identical training loop to exp7/exp14.
- Gate `THRESHOLDS["exp16"].max_cross_scheme_detector_accuracy =
  0.55` — exp7's number, reused. Pre-registered: **both** cross
  directions must be at/below it; the within-scheme controls carry
  no gate but the artifact records whether they cleared the line
  (`controls_valid`), and the reading rule was pre-stated in the
  module docstring: cross numbers are not read without a valid
  control.

### 22.2 The measurement

| train → test | accuracy | confusion | reading |
|---|---|---|---|
| sign → sign | **62.85%** | 85/96/48/59 | control **clears** the 55% line — pipeline validated |
| **sign → LWE** | **50.00%** | tp=29, fp=29 — no signal | **no transfer; informative** (its control passed) |
| LWE → LWE | **50.00%** | tp=0, fp=0 — learned nothing | control **collapsed** |
| **LWE → sign** | **50.00%** | same collapsed model | **NOT established** |

Gate: **PASS** as pre-registered (both cross directions 0.5000 ≤
0.55). The artifact also records `controls_valid: false`, because the
pre-registered strong condition — *both* within-scheme controls
clearing the line — did not hold.

### 22.3 The reading — directional, and the collapsed control is not a bug

The collapse of `lwe_to_lwe` is **not a broken pipeline**: it
reproduces exp12's independently measured LWE detector accuracy of
**50.00%** (§1, `exp12_lwe_cross_model.json`). LWE's modifications are
9.4× smaller than sign's (§3 W2's prior table), and this detector
cannot learn them even in its own scheme — that is a *detectability*
fact about LWE, already on the record, not an exp16 failure.

So the conclusion splits:

- **sign → LWE: established.** The sign detector demonstrably works
  (62.85% on its own scheme) and scores chance on LWE. Structure
  learned from sign does not transfer.
- **LWE → sign: not established.** The LWE-trained detector never
  learned LWE, so its 50% on sign says nothing about transfer. What
  it does say is consistent with §22.2's note: there is nothing in
  LWE's patches *for this detector to learn* — which is why a
  LWE-trained generic detector is empty-handed either way.

The honest one-sentence summary for citation: *a sign-trained
neural detector does not transfer to LWE (62.85% → 50.00%, control
validated); the reverse direction is uninformative because LWE is not
detectable by this detector even in its own scheme, matching exp12's
50.00%.* Not "cross-scheme established in both directions" — the
artifact's own `controls_valid: false` forbids that reading, and
`claim_audit` pins the asymmetry so it cannot be quietly rounded off
into a stronger claim.

What would have been reported if the numbers had come out the other
way: cross > 55% in either direction would have meant the schemes
share detectable structure and a generic detector catches both — a
worse result for the stealth claim, recorded the same way. The gate
was written to be able to fail.

### 22.4 Verification

```bash
cd nes-llm
../.venv/bin/python claim_audit.py                    # 64/64 (4 new)
../.venv/bin/python check_consistency.py              # 9/9
../.venv/bin/python -m unittest discover -s tests -p 'test_*.py'   # 38 OK
../.venv/bin/python -m src.experiments.exp16_cross_scheme_detector  # re-runs
```

State: 64/64 claims, 9/9 consistency, 38 tests, manifest unchanged at
35 PASS / 6 FAIL / 0 NOT_RUN / 0 ERROR — exp16 is the fourth
standalone gate outside the manifest grid. Scope: one transfer
direction measured and negative (no transfer), one uninformative by
the control's own collapse; Not-established #2 is annotated, not
deleted.

## 23. W1.1 — the quantization-aware strategies (exp17)

Suggested order item 5 (`RESEARCH_PLAN` §7 #5). §2 W1.1: two
strategies on disk, never run — `QuantizationStrategy` and
`NF4QuantizationStrategy` implement the per-tensor ABC, not
production's dict contract, so *"needs an adapter in
strategy_registry.py."*

### 23.1 What was built

Two registry entries and one adapter, in
`nes-llm/src/embedding/strategy_registry.py`:

- **`qae` (READY)** — `QaeDictAdapter`: per layer it delegates to the
  ABC's own `embed(residual_tensor, positions, bits)` so the
  strategy's margin logic (0.25 × layer std) runs *unmodified*; the
  adapter only does what `BaseEmbedder` does for every other strategy
  (sorted-layer walk, global bit-stream slicing,
  `EmbeddingResult` wrapping). Nothing is re-implemented, so nothing
  can drift.
- **`nf4_qae` (BLOCKED)** — registered with a raising factory and the
  full diagnosis in `StrategySpec.notes`. The reference residual
  comes from `ReferenceBuilder.build(fp16_weight, nf4_weight)` — one
  extra NF4 quantize/dequantize cycle over the *absolute* weights —
  but `strategy.embed` receives only `(residuals, bits,
  selector_indices)`, `EmbeddingConfig` carries no weights or model
  id, and a grep confirmed no caller passes `IntelligentEmbedder
  .embed`'s optional `fp16_weights`/`quantized_weights`. The
  reference cannot be rebuilt from cached residuals. Wiring it
  honestly means extending the shared contract for *every* strategy —
  an author decision, recorded instead of bolted onto one experiment.
  The probe's recorded `RuntimeError` *is* the measurement of that
  status.

The probes (`probe_extraction`) catch exceptions per strategy, and
`test_every_spec_has_notes` only checks notes, so a raising factory
degrades into data rather than breaking anything.

### 23.2 The measurement — exp17

`nes-llm/src/experiments/exp17_qae_round_trip.py` runs **exp3's exact
production path** (same `DecryptPipeline`, same honest BER against the
transmitted bit sequence rather than a stats field) with
`embedding_strategy="qae"`, plus a no-cover probe for both registered
strategies. Gate `THRESHOLDS["exp17"].max_ber = 0.0` — exp3's own
number, reused, with exp3's decrypt-and-match conditions.

| quantity | value |
|---|---|
| round trip | **BER 0.0 over 48,256 bits**, 0 errors; decrypt OK, message matches → **PASS** |
| no-cover probe, qae | embed ✓, extract-without-cover ✓, BER 0.0, `structurally_usable: true` |
| no-cover probe, nf4_qae | `embed_ok: false`, error = the BLOCKED diagnosis verbatim |
| values changed | 24,076 of 811,597,824 (2.97e-05) — carriers whose sign already matched the bit are written unchanged |
| mean \|Δ\| over changed | 0.0355 (max 0.160) |

**Structural reading, stated before anyone over-reads it:** the
adapter round-trips, and the encoding class is sign-family — the
class writes `+max(|r|, 0.25·std)` / `−max(|r|, 0.25·std)`, so
QAE-V1 forces a sign flip exactly as sign does (`forces_sign_flip:
True` in the registry). The *"stays inside the NF4 bucket"* property
the plan attributes to QAE lives in `NF4QuantizationStrategy` — the
class that is blocked. So exp17 establishes **wiring and round trip**,
not a stealth result: no PPL, robustness or detectability number was
produced for qae, and the plan's trap note stands — if a later
comparison has QAE win, the honest statement is *"QAE is well-matched
to NF4"*, not a stealth claim about other schemes.

### 23.3 Verification

```bash
cd nes-llm
../.venv/bin/python claim_audit.py                    # 64/64 (4 new)
../.venv/bin/python check_consistency.py              # 9/9
../.venv/bin/python -m unittest discover -s tests -p 'test_*.py'   # 38 OK
../.venv/bin/python -m src.experiments.exp17_qae_round_trip        # re-runs
```

State: 64/64 claims, 9/9 consistency, 38 tests, manifest unchanged at
35 PASS / 6 FAIL / 0 NOT_RUN / 0 ERROR — exp17 is the fifth standalone
gate outside the manifest grid. What W1.1 promised ("adds two
strategies cheaply") is delivered as: one wired and measured, one
registered with its blocker diagnosed — which is the honest maximum
the current contract allows.

## 24. W1.3 — strategy × model matrix, first pass (exp18)

Suggested order item 6 (`RESEARCH_PLAN` §7). §2 W1.3 is the
*"breadth deliverable"*: all viable strategies × models, one table,
four axes — extractability, BER, robustness, detectability — with the
plan's own cost control: first pass on 3 models, one per size class,
one model per process.

### 24.1 What was built

`nes-llm/src/experiments/exp18_strategy_model_matrix.py`. The design
decision that matters: **the four axes are exp10's own three
measurement functions**, parametrised by model (`family`/`num_layers`
kwargs added, defaulting to exp10's Qwen constants so exp12's
positional imports are byte-compatible). The protocol — 400 pairs / 20
embeds / 30 epochs detector, exp6's σ grid with 3 trials, exp10's
round trip — is unchanged, which makes the table *an extension of
exp10/exp12*, not a parallel dialect with its own dialect drift.

- **Models:** gemma-2-2b (26L, small), Qwen2.5-3B (36L, mid),
  Llama-3.1-8B (32L, large). The plan's small-class pick TinyLlama
  has an incomplete cache → SKIPPED under the standing rule;
  gemma-2-2b (the exp15 second model) takes the slot.
- **Strategies:** the four READY ones — sign, magnitude_aware, lwe,
  qae. Excluded **by name** in every artifact: `neural`
  (NEEDS_TRAINING — W1.2, never trained) and `nf4_qae` (BLOCKED —
  exp17's contract diagnosis).
- **Gates:** `THRESHOLDS["exp18"]` — four reused numbers, none new:
  exp3's 0.0 round trip, exp6's 0.02 @ σ=0.001 and 0.10 @ σ=0.002,
  exp7's 0.55 detector. Verdicts are **per cell**; a cell failing an
  axis is data, not an experiment error.

### 24.2 The matrix — 12 cells

Detector accuracy (the separating axis); gate 0.55, ✓ = under gate:

| model | sign | magnitude_aware | lwe | qae |
|---|---|---|---|---|
| gemma-2-2b (26L) | 0.781 ✗ | 0.781 ✗ | **0.500 ✓** | 0.838 ✗ |
| Qwen2.5-3B (36L) | 0.750 ✗ | 0.675 ✗ | **0.500 ✓** | 0.725 ✗ |
| Llama-3.1-8B (32L) | 0.606 ✗ | 0.594 ✗ | **0.500 ✓** | 0.594 ✗ |
| **wins** | 0/3 | 0/3 | **3/3** | 0/3 |

The other two axes do not separate anything:

- **Round trip: 12/12 at BER 0.0** (own extractor, no cover).
- **Robustness: 12/12 gates pass** — σ=0.001 is 0.0 in every cell;
  σ=0.002 ranges 0.0–0.0127 (LWE is the highest at ~0.0127, 8× under
  the 0.10 gate). At the un-gated σ=0.005, LWE pays for its smaller
  perturbation (BER 0.31–0.32 vs sign-family ≤0.09 on gemma) — a
  trade-off recorded, not gated.

### 24.3 The reading

1. **Only detectability discriminates.** Every strategy round-trips
   and survives the noise gates on every model — the four-axis table
   collapses to one axis. The strategy choice is a stealth choice.
2. **LWE = exactly 0.50 on all three models** — the third
   independent reproduction of the chance-level reading (exp10's
   Qwen, exp12's 5-model sweep, now exp18's three). The detector
   gradient (gemma 0.78–0.84 strongest, Llama 0.59–0.61 weakest)
   applies only to the sign family; LWE sits on chance
   model-independently.
3. **qae never wins anywhere, and is the *worst*-detected strategy
   on the two smaller models** (0.838 gemma, 0.725 Qwen). exp17's
   structural reading predicted this: qae is sign-family with a
   margin floor, and the floor *amplifies* the sign signal at weak
   carriers rather than hiding it. The plan's W5.2 premise
   (*"QAE encode + LWE read-out … both mechanisms avoid sign
   flips"*) is therefore half-false as written — QAE-V1 does force
   sign flips. Any W5.2 build must state that, or use a mechanism
   that does not (nf4_qae, which is blocked).
4. **Pipeline flag:** `usable_in_production_pipeline` is false for
   LWE on all three models — `DecryptPipeline` is hardcoded to the
   sign extractor (exp10's recorded note: a wiring gap, not a scheme
   defect; its own extractor round-trips at BER 0.0).

**Side finding, recorded not fixed:** exp10's artifact rows are
post-Phase-3 (`lwe wins=True`) but its stored `conclusion` string is
pre-fix stale (*"LWE needs the cover"*) and contradicts the rows
beneath it. Artifacts are records, so it was not edited;
`claim_audit` pins the rows and reports
`conclusion_stale=True` so nobody cites the conclusion field.

**Scope:** first pass — 3 of the 7-model grid; widening remains.
`neural` remains untrained (W1.2). Cell verdicts only; the detector
number is strategy-specific (exp10's comparability note carries).

### 24.4 Verification

```bash
cd nes-llm
../.venv/bin/python claim_audit.py                    # 64/64 (6 new)
../.venv/bin/python check_consistency.py              # 9/9
../.venv/bin/python -m unittest discover -s tests -p 'test_*.py'   # 40 OK
../.venv/bin/python -m src.experiments.exp18_strategy_model_matrix \
    --model Qwen/Qwen2.5-3B                           # re-runs one model
```

State: 64/64 claims, 9/9 consistency, 40 tests (two adapter tests
added with the exp10 guard), manifest unchanged at 35 PASS / 6 FAIL /
0 NOT_RUN / 0 ERROR — exp18 is the sixth standalone artifact outside
the manifest grid. W1.3's first pass closes: the breadth deliverable
now has a table, and the table has one axis.

## 25. W5.1 — adaptive routing as designed (exp19)

Suggested order item 7a (`RESEARCH_PLAN` §7). W5.1 was
*"tests someone else's design and gives a baseline for anything
better"* — `AdaptiveStrategy` estimates σ from first-order residual
differences and routes: σ < 0.0005 → lwe, σ < 0.003 → neural, else
sign. The plan called it cheap; it was (one embed per model, plus
two fallbacks for the one failure).

### 25.1 What was built

- **`adaptive` joined `REGISTRY`** (seventh strategy): lazy factory,
  `needs_trained_model: true` for the neural branch,
  `forces_sign_flip` **branch-dependent** and said so in the notes
  (sign flips, lwe/neural do not — the artifact records which branch
  fired), plus the lwe-branch caveat: the design builds `LWEStrategy`
  with a fresh `os.urandom(32)` key per run, so adaptive-lwe is
  self-consistent but **not bit-identical** to the registry's
  zero-key lwe. Recorded, not "fixed" — that is the design's own
  constructor.
- **`AdaptiveRoutedExtractor`** in `extractors.py`: only the strategy
  knows which branch fired, so the decoder is fetched from it at call
  time (`get_extractor`), forwarding the cover when a caller has one
  and defaulting to the stego tensor — what an extractor really
  holds, which `LWEStrategy.extract` documents as equivalent for
  sparse payloads (Phase 3's no-cover derivation).
- **`THRESHOLDS["exp19"]`** = exp3's `max_ber: 0.0`, reused. The
  **routing choice is measurement, not gate**; a route to an
  unavailable branch is recorded as the design's own failure.
- **Two defects found by the first run, fixed in the module, not
  patched around:** `IntelligentEmbedder`'s `EmbedResult` wrapper
  drops the inner `EmbeddingResult.metadata` where the design parked
  `estimated_sigma` (recomputed via the deterministic
  `estimate_noise` — same value the routing used, recorded in
  `method.sigma_recompute`), and `routing.thresholds` was left
  empty (now read from the strategy's class constants). All three
  artifacts were re-run from the final code so none is a hybrid of
  two versions.

### 25.2 The routing — three models, three branches

| model | σ_est | routed branch | route available | round trip BER | gate |
|---|---|---|---|---|---|
| gemma-2-2b (26L) | **0.000448** | **lwe** | yes | `adaptive->lwe`: **0.0** | PASS |
| Qwen2.5-3B (36L) | **0.001554** | **neural** | **no** — `EmbeddingError` (no trained model) | fallbacks: `sign` **0.0**, `lwe` **0.0** | PASS |
| Llama-3.1-8B (32L) | **0.007736** | **sign** | yes | `adaptive->sign`: **0.0** (pipeline recovered) | PASS |

Four round trips ran, four at exp3's 0.0. The failure cell is the
interesting one: the middle bracket routes to a branch the design
ships without (it says: train a model, pass `neural_model_path`),
so on Qwen **the design as written does not embed at all**. The
`EmbeddingError` text is kept verbatim in the artifact; the
experiment then round-tripped the design's available branches as
its baseline — the design itself has no automatic fallback, it
raises.

### 25.3 The reading

1. **The router discriminates on real data.** Three models fell into
   three different brackets — the thresholds are not decorative,
   and a 17× spread in estimated residual noise (0.000448 → 0.007736)
   exists across these three families. Any "one scheme for all
   models" claim has to beat a router that would pick differently
   per model.
2. **Design-as-written fails closed on the middle bracket.** The
   failure is the design's own (recorded, never patched), and the
   fallback branches — both available, both 0.0 — show what the
   design *would* do if it had one.
3. **σ-estimator agrees with intuition about size:** the smallest,
   quietest residuals route to fidelity (lwe), the noisiest to
   robustness (sign). No contradiction to chase.
4. **Scope:** routing + round trip only. Detectability of the routed
   primitives is exp18's axis (lwe 0.50, sign-family 0.59–0.84 —
   so the router picks *detectable* branches on two of three models
   when it picks sign; stealth is not in this design's objective).
   No detector, no perplexity, no new gate.

### 25.4 Verification

```bash
cd nes-llm
../.venv/bin/python claim_audit.py                    # 71/71 (7 new)
../.venv/bin/python check_consistency.py              # 9/9
../.venv/bin/python -m unittest discover -s tests -p 'test_*.py'   # 40 OK
../.venv/bin/python -m src.experiments.exp19_adaptive_routing \
    --model Qwen/Qwen2.5-3B                           # re-runs one model
```

State: 71/71 claims (the 7 new pin the gate, branch-recomputes-from-σ
consistency with the three-way split, Qwen's failure as recorded,
gate-status recomputation, protocol pins, pipeline-flag
consistency), 9/9 consistency, 40 tests, manifest unchanged at
35 PASS / 6 FAIL / 0 NOT_RUN / 0 ERROR — exp19 is the seventh
standalone artifact outside the manifest grid. W5.1 closes; W5.3
(sign/parity split) is the highest-value remaining W5 item.

## 26. W5.3 — the sign/parity split dial (exp20)

Suggested order item 7, the plan's *"most interesting scientifically"*
(`RESEARCH_PLAN` §4 W5.3): *"Parity on a fraction of carriers, sign on
the rest. Makes the stealth-vs-robustness trade-off an explicit dial
rather than a per-scheme guess."*

### 26.1 What was built

- **`SplitStrategy`** (eighth registry strategy): `split_fraction`
  of carriers carry LWE-style parity, the rest production sign —
  both mechanisms **delegated** (`LWEStrategy.embed`,
  `SignEmbeddingStrategy.embed`) on disjoint positions, neither
  re-implemented. The dial is `EmbeddingConfig.split_fraction`.
- **Keyless partition:** parity iff `blake2b("nes-split-v1:layer:"
  "position")` falls below the fraction — public by design, since
  exp13 established the grid width is public too. Membership is a
  per-position predicate, not a rank over the embedder's selector
  set, so the extractor (holding only written carriers) reproduces
  it exactly. Bit layout is parity-first; extraction decodes parity
  carriers with `LWEStrategy.extract` (stego-std grid, no cover) +
  sign carriers with `SignExtractor`, concatenated in the same
  order.
- **exp10's four measurement functions** gained an optional
  `config_overrides` kwarg (default `None` — exp10/exp12/exp18
  byte-compatible), so the five cells vary ONLY the fraction:
  round trip, exp6's σ-grid, exp7's 400-pair detector — protocol
  unchanged, comparable to exp10/exp18 directly.
- **Gate:** `THRESHOLDS["exp20"]` = exp18's four reused numbers.
  Per-cell verdicts (exp18's rule); the SHAPE across fractions is
  the measurement.

**The label bug the first run caught (recorded, not smoothed):**
`split_fraction` is the plan's *parity* share — f=0.0 leaves the
parity set empty, so 0.0 is pure **sign** — but the module
docstrings, `ANCHOR_FRACTIONS`, and the registry notes labeled the
endpoints backwards. The artifact convicted itself: f=0.0 ran
`DecryptPipeline` successfully (a sign-only decoder) and f=1.0's
σ=0.002 BER 0.01267550702028081 and detector 0.50 were exp18's
**lwe**-cell numbers while the anchors compared them to `sign`.
The mechanism code and unit tests had it right (the test variables
are `pure_sign` at 0.0); prose was the wrong side. Labels were
fixed and the artifact **regenerated by a full re-run — never
edited in place**. Two facts verified for the record: under aligned
state, split(f=0.0) stego is **bit-identical** to direct `sign`
(same carriers, same tensors), and the full production path gives
identical carriers with transmitted bits differing only because
`IntelligentEmbedder` generates a **fresh AES key per embed**
(exp15's recorded digit-variation phenomenon) — so cross-run
endpoint deltas are payload variance plus exp10's strategy-tagged
detector messages (`PHASE2_split_` vs `PHASE2_sign_`), not
mechanism drift.

### 26.2 The dial — five fractions, one model

Qwen2.5-3B first pass; all five round trips BER 0.0; all five pass
σ=0.001 (0.0 everywhere). Detector gate 0.55:

| parity share | BER @σ0.002 | BER @σ0.005 | detector | wins |
|---|---|---|---|---|
| 0.00 (pure sign) | 0.0 | 0.0001 | 0.7875 ✗ | — |
| 0.25 | 0.0034 | 0.0810 | 0.6938 ✗ | — |
| 0.50 | 0.0067 | 0.1595 | 0.7000 ✗ | — |
| 0.75 | 0.0095 | 0.2376 | 0.5875 ✗ | — |
| 1.00 (pure parity) | 0.0127 | 0.3175 | **0.5000 ✓** | **✓** |

Endpoints, anchored to exp18's committed cells: **pure parity
reproduces exp18's `lwe` cell exactly** (detector delta 0.0;
σ0.002/σ0.005 bit-for-bit), pure sign lands +0.0375 from exp18's
`sign` cell (fresh-key/variance band, as verified above).

### 26.3 The reading

1. **The trade-off is a dial.** Robustness falls monotonically
   with parity share (0 → 0.0127 → 0.3175 at σ 0.002/0.005);
   detectability falls with it (0.7875 → 0.50). Stealth is bought
   with noise-fragility, continuously — the per-scheme guess
   ("sign or LWE?") was choosing a point on this curve without
   seeing it.
2. **Only pure parity wins every gate** — and only because LWE's
   detector is invariantly 0.50 (fourth reproduction: exp10, exp12,
   exp16/18, now through the split's own path). The sign signal
   contaminates detectability up to a 0.75 parity share (0.5875,
   still over gate): stealth does not emerge gradually — it
   arrives late.
3. **DecryptPipeline consumability** follows the pure-sign end
   only (f=0.0 True, f>0 False): one sign-readable bitstream among
   five cells — exp10's wiring note, not a scheme defect.
4. **Mid-dial cells fail detectability, per cell, never as an
   experiment error** (exp18's rule): four "✗" rows are the
   measurement, and the win row is the endpoint.

**Scope:** one model (first pass); the 7-model widening and W7's
full frontier (mean perturbation × detector × BER markers) remain.
`neural`/`nf4_qae` are not involved here at all.

### 26.4 Verification

```bash
cd nes-llm
../.venv/bin/python claim_audit.py                    # 80/80 (9 new)
../.venv/bin/python check_consistency.py              # 9/9
../.venv/bin/python -m unittest discover -s tests -p 'test_*.py'   # 42 OK
../.venv/bin/python -m src.experiments.exp20_split_dial \
    --model Qwen/Qwen2.5-3B                           # re-runs the dial
```

State: 80/80 claims (9 new: artifact, gate = exp18's four numbers,
five cells in order, verdict recomputation, round trips at 0.0,
protocol pins, anchor deltas recomputed from exp18's artifact,
exact dial vectors, both trade-off directions recomputable),
9/9 consistency, 42 tests (two split tests: round trip at three
fractions + endpoint purity + cover non-mutation), manifest
unchanged at 35 PASS / 6 FAIL / 0 NOT_RUN / 0 ERROR — exp20 is the
eighth standalone artifact outside the manifest grid. W5.3 closes;
W5.2's premise (QAE/LWE both avoid sign flips) remains half-false
from exp17 and is next in the W5 list.

## 27. W5.2 — QAE encode + LWE read-out (exp21)

Suggested order item 7 continues (`RESEARCH_PLAN` §4 W5.2): *"QAE
encode + LWE read-out. Quantization-aware placement, parity decode.
Plausible: both mechanisms avoid sign flips."* The premise was already
recorded half-false by exp17 — QAE-V1 **forces sign flips**
(`+max(|v|, margin)` / `-max(|v|, margin)`, sign-family with a margin
floor) — so the honest reading of the item is the question it
actually asks: **can the LWE parity read-out decode a QAE-encoded
stream at all?**

### 27.1 What was built

- **One production-path embed, three readings of the same stego** so
  the read-out pairing is the only variable: the matched control
  (qae's own `SignExtractor`), the raw LWE parity read-out
  (`floor(v/0.010)%2`, `DEFAULT_GRID_WIDTH`, stego statistics), and
  a public cell-parity correction of the raw reading.
- **Gate:** `THRESHOLDS["exp21"]` — both readings at exp3's 0.0,
  both reused. The control at 0.0 is what makes a non-zero interop
  BER attributable to the pairing rather than a broken embed; the
  verdict is the **raw** interop's, and the correction is recorded
  as structure with an explicit *"not a gate rescue"* role.

### 27.2 The numbers

| reading | BER | errors / compared | gate 0.0 |
|---|---|---|---|
| matched control (qae's extractor) | **0.0** | 0 / 10,256 | ✓ |
| raw LWE parity read-out | **0.5433** | 5,572 / 10,256 | ✗ |
| public cell-parity correction | **0.0** | 0 / 10,256 | (structure) |

**Verdict: FAIL** — the plan's *"plausible"* claim fails as written,
recorded with its number (exp13's pattern: the property that should
hold does not, and the gate stays at 0.0).

### 27.3 The reading

1. **A pre-registered prediction missed, and the miss is data.**
   Before the run the module predicted the *complement* (BER near
   1.0): typical residuals (std ≈ 0.002) sit below one grid width
   (w = 0.010), where parity mirrors sign exactly. Measured: 0.5433,
   near chance. Correction verified against the cache — QACI selects
   the **top-magnitude tail**, and the sampled top-tail carriers are
   *all* above one grid width (min 0.013 > 0.010, median 0.021), so
   no complement regime exists for the values that actually get
   written. Prediction, miss, and correction are all recorded.
2. **The structure the numbers expose.** For any non-multiple of w,
   `parity(v) = sign(v) ⊕ cell-parity(|v|)` — an algebraic identity,
   not a property of qae. Read-out C applies exactly that
   (`raw ⊕ floor(|stego|/0.010)%2 ⊕ 1`, stego magnitudes only) and
   returns **0.0**: the identity is *measured*, not asserted.
3. **What this says about the plan item.** The LWE read-out of a qae
   stream is the sign bit XOR a per-carrier constant any extractor
   can recompute from the stego it already holds — so a "QAE encode +
   LWE read-out" hybrid would be **sign reading with a public
   relabeling**, not a second independent channel. The premise
   *"both mechanisms avoid sign flips"* dissolves on both halves:
   qae flips signs (exp17), and parity read-out of *any* value
   carries the sign bit in re-encodable form (here). The raw
   combination still fails the round trip — 0.5433 vs 0.0 gate.
4. **Run-to-run.** The raw interop BER reproduced bit-for-bit across
   runs (5,572/10,256 both times): the error count equals the
   odd-cell fraction of the carrier set, which is payload-independent
   (fresh AES keys change the stream, not the QACI-selected
   positions). exp15's fresh-key variance does not move this number.

### 27.4 Verification

```bash
cd nes-llm
../.venv/bin/python claim_audit.py                    # 87/87 (7 new)
../.venv/bin/python check_consistency.py              # 9/9
../.venv/bin/python -m unittest discover -s tests -p 'test_*.py'   # 42 OK
../.venv/bin/python -m src.experiments.exp21_qae_lwe_readout \
    --model Qwen/Qwen2.5-3B                           # re-runs the pair
```

State: 87/87 claims (7 new: artifact, gate = exp3's 0.0 × 2,
matched control, raw interop exact number + verdict recompute,
corrected exactly 0.0 with its not-a-rescue role, premise quoted
verbatim and half-false against exp17, protocol pins), 9/9
consistency, 42 tests, manifest unchanged at 35 PASS / 6 FAIL /
0 NOT_RUN / 0 ERROR — exp21 is the ninth standalone artifact outside
the manifest grid. W5.2 closes with a FAIL that is the finding.
W5.4 (per-layer strategy selection) is the last W5 item.

## 28. W5.4 — per-layer LWE grid width (exp22)

Suggested order item 7 closes (`RESEARCH_PLAN` §4 W5.4): *"Per-layer
strategy selection. Different grid width per layer, keyed by layer
noise. Layers differ: Qwen2.5-7B spans 0.0012–0.0130, Phi-3 spans
0.0026–0.0031."*

### 28.1 Scoping, before any code

Two measurements decided the design:

1. **Is there a dial?** Qwen2.5-3B's per-layer residual std spans
   **0.00114–0.00262 — 2.30×**, median 0.00241 (the plan's ~10× quote
   is the 7B). A real but modest dial: two quiet outliers below a
   tight pack.
2. **Can embed and extract agree on a width?** Embed sees *original*
   layer std; the extractor sees *stego* std. Measured directly: an
   LWE embed moves per-layer std by at most **0.0153%**, and zero
   4-decimal buckets flip — so `round(std, 4)` coarsening makes the
   two views bucket-identical on this model (the gate's BER 0.0
   would catch any edge case).

### 28.2 What was built

- `EmbeddingConfig.lwe_width_rule` — `"global"` (default: the shipped
  absolute 0.010, byte-compatible with exp10/11/12) or `"per_layer"`:
  `w_l = clip(4.0 × round(std_l, 4), 0.005, 0.020)` — proportional to
  the layer's own noise (equalizing grid-to-noise ratio: a global
  grid is 8.77σ wide on the quietest layer, 3.81σ on the noisiest),
  clipped to exp11's measured window, scale 4.0 putting the median
  layer at ~0.0096 ≈ the global default.
- **exp22**: cells = width rules, exp10's three axes each via
  `config_overrides`, `THRESHOLDS["exp22"]` = exp10's four reused
  numbers (0.0 / 0.02 / 0.10 / 0.55), per-cell verdicts (exp18's
  rule), deltas against the global control as the measurement; the
  global cell anchored to exp18's committed lwe cell (recorded,
  never gated); the widths table recorded so the artifact shows what
  the extractor computes.

### 28.3 Run 1: the design's own failure, diagnosed not smoothed

The global control reproduced exp18's lwe cell **bit-for-bit**
(σ0.002 = 0.01267550702028081, detector 0.50 — lwe's invariant, now
measured in six separate experiments) and won. per_layer round-tripped
at **0.0** and hid (detector 0.50) — then **collapsed under noise**:
**0.0223 at σ0.001** (gate 0.02, fails narrowly) and **0.5747 at
σ0.002** (near chance, against global's 0.0127).

Pre-registration honesty: the prediction said per_layer "can only
match or trail global's" robustness — right direction — but attributed
it to quiet layers sitting at the window floor. The actual cause was
verified numerically and it is different:

**The extractor sizes the grid from the tensor it receives, and the
robustness measurement hands it the *noisy* tensor.** Noise inflates
std through `√(std² + σ²)` (quiet layer 0.00114 → 0.00230 at
σ=0.001), and at 4-decimal precision **36/36 layers bucket
differently at every σ tested** — the extractor's grid drifts wider
than the embedder's (0.0096 → 0.0104 at σ0.001; 0.0096 → 0.0124 at
σ0.002), and carriers at high cell indices decode with the wrong
parity. Every number coheres: σ0 (no inflation) = exact 0.0;
mismatch grows with σ, BER tracks it. Embedding drift (0.0153%,
§28.1) was never the problem — **magnitude-keying is unbuildable at
extract time**: the statistic moves under exactly the perturbation
the gate measures.

### 28.4 `layer_rank`: the keying noise cannot move

Run 1's diagnosis is preserved in the artifact as
`noise_bucket_flips` (recomputed analytically from its own recorded
stds), and it suggested the fix: `√(std² + σ²)` is **strictly
monotone**, so the *rank order* of layer noise is preserved exactly
under any σ. Widths keyed to rank are therefore identical at embed
and extract by construction, while still running different widths
per layer: a fixed ladder `[0.005, 0.010]` by rank (ties on layer
id; constants, so no endpoint depends on either side's view).

| rule | round trip | σ0.001 | σ0.002 | σ0.005 | detector | wins |
|---|---|---|---|---|---|---|
| global | **0.0** | 0.0 | 0.0127 | 0.3175 | 0.50 | **yes** |
| per_layer | **0.0** | 0.0226 ✗ | 0.5763 ✗ | 0.6606 | 0.50 | no |
| layer_rank | **0.0** | 0.0015 ✓ | 0.0736 ✓ | 0.4269 | 0.50 | **yes** |

Deltas vs global: layer_rank +0.0015/+0.0609 (robustness), detector
0.0; per_layer +0.0226/+0.5636, detector 0.0.

### 28.5 The reading

1. **The plan's literal design fails, with its cause in numbers.**
   Widths keyed to *measured* layer noise cannot survive the
   robustness gate, because the extractor's measurement of layer
   noise is contaminated by the attack being measured. exp22
   records the failure mode (36/36 bucket flips per σ, recomputable
   from the artifact) rather than a verdict without a mechanism.
2. **The salvageable half passes.** Rank-keyed heterogeneity
   round-trips at 0.0 and clears both robustness gates — agreement
   by construction, tested against the same protocol.
3. **But heterogeneity does not move the frontier.** The lwe
   detector is **width-blind** — 0.50 on all three cells, seventh+
   reproduction of an invariant now spanning exp10/12/16/18/20 and
   exp22 — and the ladder's sub-default median width *costs*
   robustness (σ0.002: 0.0127 → 0.0736). The shipped **global
   0.010 stays the best point of the three** on this model: W5.4's
   answer is "the dial exists and is buildable, and it does not
   help" — a complete negative result, not a half-tested one.
4. **Control discipline held throughout.** The global cell equals
   exp18's lwe cell exactly (detector delta 0.0, curve bit-identical)
   in both runs; per_cell verdicts [True, False, True] recompute from
   the rows; unknown width rules fail loudly at build time.

### 28.6 Verification

```bash
cd nes-llm
../.venv/bin/python claim_audit.py                    # 98/98 (11 new)
../.venv/bin/python check_consistency.py              # 9/9
../.venv/bin/python -m unittest discover -s tests -p 'test_*.py'   # 46 OK
../.venv/bin/python -m src.experiments.exp22_layer_widths \
    --model Qwen/Qwen2.5-3B                           # 3 cells, ~22 min
```

State: 98/98 claims (11 new: artifact, gate = exp10's four reused,
three rules + overrides, widths recomputed from the artifact's own
stds for all three rules, 3 verdicts, 3 round trips, protocol pins,
anchor with recomputed delta, both deltas vs control, the 36/36
diagnosis recomputed analytically, and the pinned result table),
9/9 consistency, 46 tests (4 new: global default byte-compatible,
per-layer rule with a cross-instance round trip, bucket stability
under the measured drift band, rank agreement under
`√(std²+σ²)` distortion + loud failure for unknown rules), manifest
unchanged at 35 PASS / 6 FAIL / 0 NOT_RUN / 0 ERROR — exp22 is the
tenth standalone artifact outside the manifest grid. **All of §7
item 7 (W5) is now closed: 7a/exp19, W5.3/exp20, W5.2/exp21,
W5.4/exp22.** Next in the suggested order: W6 model surgery, whose
environment probe already recorded the blockers — `peft`, `gptqmodel`
and `trl` absent (`bitsandbytes` present, NF4 reachable), so W6.3's
GPTQ leg and the LoRA-typed paths are blocked as designed, not
patched.

## 29. W6 — model surgery survival (exp23)

Suggested order item 8 (`RESEARCH_PLAN` §4 W6): *"Model surgery (LoRA
merge, fine-tune, re-quantize, prune, merge) determines viability."*
W6.1/W6.3-NF4/W6.4/W6.5 are measured here; W6.2 and W6.3's
GPTQ/AWQ legs are blocked with probes, recorded not patched.

### 29.1 What was built, and the crash the layout mismatch caused

**exp23**: one production-path sign embed (payload 10k → 10,256
bits) over Qwen2.5-3B's cached residuals, then nine surgery cells
over the SAME embed — control, LoRA-shaped deltas (rank 8, RMS
1.0e-3 / 1.0e-2 of RMS(W)), magnitude prune 10/30%, NF4 re-quant
(bitsandbytes' own kernels, blocksize 64), and task-vector merge with
Qwen2.5-3B-Instruct at t = 0.01/0.05/0.5. Per cell
`r' = embedded_r + (W' − W_stego)`, extracted by the production
strategy; gate `THRESHOLDS["exp23"]` = exp3's 0.0 twice, per-cell
verdicts (exp18's rule). Surgery scope is `mlp.down_proj` only — the
payload's scope: residuals *are* down_proj residuals by definition, so
other modules cannot touch the payload by construction.

Run 1 crashed at W_stego construction, on layer 0, before any cell
ran and before any artifact was written. Cause: the residual world is
**flat** (`extract_residuals` stores `(fp16_w − dequant).flatten()`,
the cache matches it, carriers index that layout) while `_down_projs`
returns the **matrix** — a `(2048, 11008)` minus `(22544384,)`
broadcast. Fixed at the two points where the worlds meet, using the
bridge the pipeline itself uses: a numel-guarded reshape at the build
(a blind reshape of a mismatched matrix is a ground-rule violation)
and `flatten` at `measure_cell`'s single residual-space add. No
measurement changed; nothing was edited after the fact.

### 29.2 The run

Control triple **0.0 / 0.0 / 0.000e+00** — direct extraction, weight
path, and `cache_vs_pair` (a fresh pair residual equals the cached one
exactly, so embed and weights read the same residual view).

| cell | BER | errors/10,256 | rms Δ/W | carriers displaced | gate |
|---|---|---|---|---|---|
| control | 0.0 | 0 | 0 | 0 | ✓ |
| lora_0.001 | 0.0 | 0 | 1.0e-3 | 10,256 | ✓ |
| lora_0.01 | 0.0 | 0 | 1.0e-2 | 10,256 | ✓ |
| prune_10 | 0.0 | 0 | 0.0206 | **0** | ✓ |
| prune_30 | 0.0 | 0 | 0.1094 | **0** | ✓ |
| nf4_requant | **0.3768** | 3,864 | 0.0949 | 9,536 | ✗ |
| merge_0.01 | 0.0 | 0 | 1.34e-4 | 7,839 | ✓ |
| merge_0.05 | 0.0 | 0 | 6.72e-4 | 7,839 | ✓ |
| merge_0.5 | **0.2418** | 2,480 | 6.72e-3 | 7,839 | ✗ |

not_run: 3 blockers with runtime probes — W6.2 fine-tune
(`peft=False, trl=False`; 26 GB cannot host an honest run), W6.3 GPTQ
(`gptqmodel=False, auto_gptq=False`), W6.3 AWQ (`AwqQuantizer`
ImportError under transformers 5.16.1) — recorded, not patched.

### 29.3 The reading

1. **W6.1 and W6.4 hold outright.** LoRA-sized linear adds round-trip
   at 0.0 even though they displace *every* carrier — the delta at
   carriers (rms 2.4e-5 / 2.4e-4) never reaches the sign decoder's
   margin. Pruning 10% and 30% of the matrix displaces **zero**
   carriers (rms at carriers exactly 0.0): the payload sits outside
   the smallest-|W| mass. That is a measured placement fact, not
   luck — and it is why prune's pass needs its own column in the
   table rather than a bare 0.0.
2. **W6.3's NF4 leg fails, and its failure is graceful, not total.**
   Re-quantizing the whole stego weight displaces 9,536 carriers and
   flips 3,864 bits (0.3768) — re-quant noise (rms at carriers 0.022)
   swamps the residual scale. Against the plan's total-vs-graceful
   question: 0.3768 < 0.5, so the stream is degraded far past the
   gate but still carries signal; it is dead by the gate, alive as a
   channel.
3. **W6.5 survives merging to t=0.05 and degrades smoothly past it.**
   Merge deltas scale linearly with t (rms ratios 1 : 5 : 10 — the
   task vector measured, not asserted); BER stays 0.0 through t=0.05,
   then 0.2418 at t=0.5 — better than chance at half-replacement, so
   the payload decays continuously as the stego is overwritten.
4. **Pre-registration held.** "control 0.0" ✓, "nf4_requant
   plausibly heavy loss" ✓ (0.3768), "merge dies as t grows" ✓
   (0 → 0 → 0.2418), prune recorded open and answered with its
   mechanism. Nothing was rewritten to match the numbers; the gate
   never moved.

### 29.4 Verification

```bash
cd nes-llm
../.venv/bin/python claim_audit.py                    # 111/111 (13 new)
../.venv/bin/python check_consistency.py              # 9/9
../.venv/bin/python -m unittest discover -s tests -p 'test_*.py'   # 66 OK
../.venv/bin/python -m src.experiments.exp23_model_surgery \
    --model Qwen/Qwen2.5-3B                           # 9 cells, ~10 min
```

State: 111/111 claims (13 new: artifact, gate = exp3's 0.0 twice,
control triple exact, nine cells in order, every BER recomputed from
its own error counts, the seven survivors + both failure numbers
pinned, prune's zero-carrier explanation, LoRA RMS scales, merge
linearity, protocol pins, three named blockers, pre-registration
intact, reproducibility), 9/9 consistency, 66 tests (exp23's 4
surgery-primitive tests among them; exp24's 13 frontier tests and 3
index-sampler pins also landed in the suite this stretch), manifest
unchanged at 35 PASS / 6 FAIL / 0 NOT_RUN / 0 ERROR — exp23 is the
eleventh standalone artifact outside the manifest grid. **W6 measured
status: W6.1 ✓, W6.4 ✓, W6.5 ✓ to t=0.05 (✗ at 0.5, gracefully),
W6.3-NF4 ✗ at 0.3768 (graceful), W6.2 + GPTQ + AWQ NOT_RUN and
named.** Next: exp24 (W7), then W1.4 execution and W8.

## 30. W7 — the Pareto frontier (exp24)

Suggested order item 9 (`RESEARCH_PLAN` §4 W7): *"Plot every strategy ×
every parameter: x = mean perturbation magnitude, y = detector
accuracy, marker = BER@σ0.001"* — the strongest publishable framing.

### 30.1 The scoping decision: x had to be measured here

No committed artifact records mean |Δ| **together** with detector
accuracy. exp2/exp17 have magnitudes without detectors; every
detector-bearing artifact (exp10/11/12/18/20/22, exp7's study) records
changed-value *counts* (`signal_density`), never magnitudes. A count is
not the plan's x-axis, so substituting one would have been a different
claim wearing the same label. exp24 therefore **measures x itself** —
one embed per unique (model, strategy, config) group, originals
snapshotted before the embed — while y and marker are *cited* from the
sources at delta 0.0. That split is the whole design: measured where
the programme has no number, exact where it does.

**exp24**: 35 points over 9 frontier sources (exp10, exp11, exp12,
exp7's parameter study, exp18's three matrix files, exp20, exp22) plus
exp14/exp16 cited under `related`. 5 strategies (sign, magnitude_aware,
lwe, qae, split) × 5 models. Parameter reconstruction follows each
source's own mechanism: exp11/exp12 keep their alpha=1.0 +
min_magnitude=w/2 patch, exp20 its split_fraction, exp22 its
width_rule, exp7's recorded alpha maps to `min_magnitude` (its
module's own mapping, never `EmbeddingConfig.alpha`), everything else
the shipped default. Gate: `THRESHOLDS["exp24"]` — `max_source_delta`
0.0, citation integrity rather than a scientific threshold.

### 30.2 The run

x spans **0.003727 → 0.432320**, y spans **0.5 → 0.7875** (16 points
sit at the detector floor). The frontier is **one point**:

| | x = mean \|Δ\| | y = detector | marker = BER@σ0.001 |
|---|---|---|---|
| **`exp22:…:lwe:layer_rank`** | **0.00372693** | **0.50** | **0.00146256** |

It dominates all 34 other points — minimal on x *and* at the y floor
simultaneously, so no point can trade against it. Runner-up is
exp22's own per_layer (x 0.004788, y 0.50); the exp11/exp12 LWE cells
cluster at x ≈ 0.00498, y 0.50. The split dial traces a clean
monotone path (sf 0 → 1: x 0.047074 → 0.004992, y 0.7875 → 0.50) and
still ends dominated by layer_rank at sf=1.0. Sign-family points sit
10–100× right (x 0.046956 → 0.432320).

**10 exclusions, all with reasons** (never dropped): exp10's neural
cell (NEEDS_TRAINING), exp12's tinyllama (cache incomplete), and
gemma-2-2b's 8 cells (4 group-level: model absent from
`experiment_registry`; 4 magnitude follow-ons). **6 omitted source
groups** carry why they cannot form y (exp2/exp4/exp6/exp17/exp19 +
a catch-all naming exp1/3/5/8/9/13/15/21/23). exp14's blind/control
reading and exp16's cross-scheme matrix are cited under `related` —
same embedding, different question — and exp2/exp17's magnitudes
under `prior_magnitude_citations`, each with its protocol recorded as
cited-not-used.

### 30.3 The reading

1. **The frontier is a corner, not a curve.** The plan imagined
   points trading x against y; measured, one point is best on both
   axes at once, so the "frontier" is a winner. That is a stronger
   claim than a trade-off: no detector-accuracy/magnitude compromise
   exists among committed results — exp22's rank-keyed width is
   simply the smallest, least detectable embed the programme has
   measured.
2. **LWE's placement is why.** Every y=0.5 point in the set is LWE
   (or pure parity), and they cluster at x ≈ 0.004–0.005 while
   sign-family embeds start at 0.047 — an order of magnitude more
   perturbation for *worse* detectability. The frontier restates
   exp18's finding (LWE wins on detectability) in the plan's own
   coordinates.
3. **The dial families converge but do not win.** The split dial's
   own endpoint (sf=1.0, which *is* pure LWE) reaches y=0.5 at
   x=0.004992 — still dominated by layer_rank's 0.003727, i.e. the
   per-layer width rule buys 25% less perturbation than the dial's
   best setting at identical detectability.
4. **Citation integrity held under test.** All 35 y values, all 35
   markers (6 nulls preserved), and the frontier recompute exactly;
   tampering one y by +0.01 in a scratch copy failed the audit on the
   spot, so the gate can actually fail.

### 30.4 Verification

```bash
cd nes-llm
../.venv/bin/python claim_audit.py                    # 127/127 (16 new)
../.venv/bin/python check_consistency.py              # 9/9
../.venv/bin/python -m unittest discover -s tests -p 'test_*.py'   # OK
../.venv/bin/python -m src.experiments.exp24_pareto_frontier       # 35 pts, ~1 min
```

State: 127/127 claims (16 new: artifact, gate = citation integrity at
0.0, counts, every y recomputed at delta 0.0, markers with 6 nulls
preserved, round-trip BERs, frontier recomputation, the frontier
point's exact triple, why it is one point, x measured for all 35,
config reconstruction per source mechanism, 10 exclusions, 6 omitted
groups, related citations, prior citations, protocol pins), 9/9
consistency, tests green — manifest unchanged at 35 PASS / 6 FAIL /
0 NOT_RUN / 0 ERROR — exp24 is the twelfth standalone artifact
outside the manifest grid. **W7 measured status: frontier = 1 point
(exp22 layer_rank, x 0.00372693 / y 0.50 / marker 0.00146256),
34/34 points dominated, exclusions + omissions recorded not
dropped.** Next: W1.4 execution, then W8.

## 31. W1.4 — consolidation executed (19 files + 2 loaders gone)

The worked decision in `RESEARCH_PLAN` §4 W1.4, executed in two
commits with the full check suite run after each.

### 31.1 The 14 zero-risk files (commit `5635ecc`)

Re-grepped the import graph live before deleting: every importer of
the 14 files lay inside the deleted set, no committed `results/*.json`
came from the seven `src/evaluation/*` leaves, `claim_audit` /
`check_consistency` cited none of them, and the console entry point
(`nes=src.cli:main`) reached neither cluster. Deleted: the two
`real_residual_embedder` demos; `residual_embedder` (+`_v2`, `_qcae`)
and `embedder.py` — four divergent reimplementations of
`ResidualEmbedder`; their eight consumers (`src/main.py` + the seven
evaluation leaves). 2,370 lines. `keyed_residual_embedder` and the
live stack untouched.

### 31.2 The cache-build port — rebuild-compare before any loader deletion (commit `50ee50b`)

The recorded rule: *never delete the only path that can rebuild the
evidence.* `loader.py` line 553 was that path (it saved each layer as
a side effect of `extract_residuals`). Port:

- `model_loader.extract_residuals` gained an optional `cache=`
  keyword (default `None` → byte-for-byte the old behavior for all
  eight existing callers). When given, computed layers save through
  `cache_manager.save_layer` (numel-validated, detached to CPU,
  atomic rename, metadata after first save) and validated layers load
  instead of recomputing. Build and read now share one residual
  definition — they cannot drift.
- `scripts/cache_model.py` rewritten onto that path, plus a
  `--verify-against` mode that diffs a rebuilt cache against an
  existing one tensor by tensor.

**The gate for deleting `loader.py`:** a fresh rebuild of
Qwen/Qwen2.5-3B (both models loaded, all 36 layers re-dequantized)
into a *scratch* cache root, compared against the committed cache —
**108/108 tensors (36 layers × residual/fp16_weight/nf4_dequantized)
identical at delta 0.0**. Evidence cache untouched; scratch removed
after the compare.

Then deleted `scripts/exp1_probe`, `exp2_residual_fingerprint`,
`exp3_clean_ber`, `exp4_capacity_curve` (their only remaining importer
was `loader.py` itself; the manifest's `exp1–4` are
`src.experiments.experiments.*` and were never touched) and
`src/model/loader.py` — after confirming its four importers were
exactly those scripts.

### 31.3 Verification

```bash
cd nes-llm
../.venv/bin/python -m unittest discover -s tests -p 'test_*.py'   # 66 OK
../.venv/bin/python claim_audit.py                    # 127/127
../.venv/bin/python check_consistency.py              # 9/9
../.venv/bin/python -m scripts.cache_model --verify-against <cache root>
```

State after both commits: 66 tests, 127/127 claims, 9/9 consistency —
green after *every* batch, not just at the end. `src.model.loader`
raises `ModuleNotFoundError`; stale `.pyc` swept. **W1.4 status:
executed.** Cumulative deletions: 19 source files + 2,370 + 1,423
lines. Next: W8 (recipient CLI + delta integrity metadata), then the
misuse re-run.
