# NES Research & Productization Plan

Continuation document. Everything below is either **measured** (backed by an
artifact), **read** (established by reading code, not behaviour), or **planned**.

Read `RESEARCH_LOG.md` first for how we got here and which mistakes to not repeat.

---

## 0. Where we actually are

### Measured

| claim | evidence | file |
|---|---|---|
| Sign embedding round-trips at BER 0 | 48,256 bits compared, 0 errors | `results/exp3_*.json` |
| Sign capacity ≥ 10M bits at BER 0 | 4 sizes, all BER 0 | `results/exp4_*.json` |
| Sign robustness BER 0 at σ=0.001 | gate σ=0.001/0.002 both pass | `results/exp6_*.json` |
| Sign fidelity 0.0053% PPL delta | 3-way protocol, Qwen3B only | `results/exp5_*.json` |
| Sign is detected at 70.5% | carrier-centered MLP | `results/exp7_neural_*.json` |
| Sign detection is structural | α×100, γ×5, payload×10 → 69.4–74.4% | `results/exp7_neural_parameter_study.json` |
| LWE grid encoding is extractable without the cover | BER 0.5036 → 0.0000 | `results/exp10_*.json` |
| LWE passes both gates on 5/5 models | BER 0.0000, detector 50.00% | `results/exp12_lwe_cross_model.json` |
| GPTQ works, BER 0.0 | dequant corr 0.9903 vs FP16 | `results/exp9_formats.json` |
| 7 models covered, 0 errors | 33 PASS / 6 FAIL / 2 NOT_RUN | `results/experiment_manifest.json` |

### Not established — and load-bearing

1. **LWE perplexity is unmeasured on any model.** Only sign's was measured.
2. **Every detectability number uses a detector trained against the same scheme it
   tests.** Cross-scheme is untested.
3. **Clean and stego patches are both cut at carrier positions.** The detector is
   handed the carrier locations.
4. **The 55% detectability gate is self-chosen.** It is not a security property.
5. **LWE is LWE-*inspired*.** Key-derived grid and parity encoding; no lattice, no
   matrix A, no SIS/LWE instance. The post-quantum claim in `lwe_strategy.py` is
   unsupported.
6. **exp2's threshold fails on 6 of 7 models** because it tracks the quantization
   format, not the model. Unresolved and an author decision.

---

## 1. Workstream A — complete the strategy set

### A1. Quantization-aware strategies (never run)

`QuantizationStrategy` and `NF4QuantizationStrategy` encode bits **relative to the
NF4 centroid** — move the value but keep it inside its quantization bucket. That is
a *third* mechanism, distinct from both sign-flip and parity, and structurally it
should have near-zero PPL impact because the dequantized value barely moves.

Both implement the per-tensor `EmbeddingStrategy` ABC
(`embed(residual_tensor, bits, positions)`), not the dict interface production uses.
Needs an adapter in `strategy_registry.py`.

**Trap to avoid:** QAE is quantization-*aware*. It uses knowledge of NF4 that LWE
and sign do not. If QAE wins, the honest statement is "QAE is well-matched to NF4",
not "LWE is stealthy". Any comparison must state this.

### A2. Neural strategy (written, never run)

`train_sampled()` is implemented but untested. Blocked on `AdaptiveStrategy`, which
delegates to it.

**Known design concern [read]:** the objective rewards matching the value
distribution's mean and std. A sign flip satisfies both trivially, and nothing
penalises the sign leak. The encoder may converge to the same sign-based solution.
Running it answers this either way.

### A3. Full strategy × model matrix

All viable strategies × 7 models, one table, four axes: extractability, BER,
robustness (BER @ σ=0.001/0.002), detectability.

**Cost control:** detectability is ~10 min/model/strategy. Full 7×5 ≈ 6 h of pure
compute that will thrash 26 GB RAM. **First pass on 3 models** (one per size class:
TinyLlama 1.1B, Qwen2.5-3B, Llama-3.1-8B), then widen. Run one model per process.

### A4. Dead or duplicate code

`real_residual_embedder.py` / `_v2.py` have no embed/extract. `residual_embedder.py`,
`_v2`, `_qcae`, `embedder.py` each reimplement `ResidualEmbedder` differently.
`loader.py` (3-value return) and `model_loader.py` (dict return) remain two
divergent residual implementations, and legacy `scripts/exp*.py` import the former.
**Decide: consolidate or delete.** Do not leave four embedders and two loaders.

---

## 2. Workstream B — fidelity is the biggest hole

**LWE perplexity is unmeasured anywhere.** Undetectability is worthless if the model
is damaged.

Encouraging prior, measured:

| scheme | mean perturbation | vs residual scale |
|---|---|---|
| sign | 0.0471 | 25.6× |
| LWE | 0.0050 | **2.7×** |

LWE perturbs **9.4× less** than sign. Expect excellent PPL. Expect is not a result.

**Blocker:** the 3-way PPL protocol stalled 20+ min on MPS
(`_local_scalar_dense_mps` behind 3.2 GB of weight uploads). Needs:

- FP16 reference on CPU, NF4 on MPS, or
- chunked layer-by-layer reconstruction

Then: PPL for LWE on ≥2 models, and re-verify sign's number reproduces.

---

## 3. Workstream C — hybrids

Increasing ambition. Each is a separate artifact; none replaces an existing result.

- **C1 — `adaptive_strategy` as designed.** Noise-threshold routing to LWE/neural/sign.
  Cheap. Tests someone else's design and gives a baseline for anything better.
- **C2 — QAE encode + LWE read-out.** Quantization-aware placement, parity decode.
  Plausible: both mechanisms avoid sign flips.
- **C3 — sign/parity split.** Parity on a fraction of carriers, sign on the rest.
  Makes the stealth-vs-robustness trade-off an explicit dial rather than a
  per-scheme guess. **Most interesting scientifically.**
- **C4 — per-layer strategy selection.** Different grid width per layer, keyed by
  layer noise. Layers differ: Qwen2.5-7B spans 0.0012–0.0130, Phi-3 spans
  0.0026–0.0031.

---

## 4. Workstream D — adversaries

The detectability claim is only as good as the adversary.

| test | what it answers | cost |
|---|---|---|
| **D1 cross-scheme** | train on sign, test on LWE | moderate |
| **D2 blind patches** | adversary that doesn't know carrier positions | low |
| **D3 stronger detector** | wider/deeper, more epochs, more pairs | low |
| **D4 multi-model detector** | trained on model A, tested on B | high |
| **D5 active cleaner** | can an attacker remove the embedding? | high |

**D2 is the cheapest high-value item.** Both clean and stego patches are currently
cut *at carrier positions* — the detector is handed the answer. Carriers are ~0.001%
of positions, so a blind detector sees little per-patch signal, but could still
catch an aggregate distribution shift.

**D5 matters for the threat model.** If a defender can strip the payload on sight,
the mechanism is not durable even if it is undetectable.

---

## 5. Workstream E — model surgery

Does the embedding survive contact with the rest of the ML lifecycle? A carrier that
survives one cycle but dies on first fine-tune is not viable.

- **E1** survive LoRA merge
- **E2** survive 1–10k fine-tuning steps
- **E3** survive re-quantization (NF4 → GPTQ → back)
- **E4** survive pruning / distillation
- **E5** survive model merging

Each is: embed → surgery → extract → BER. Expect degradation; the question is
whether it is total or graceful.

---

## 6. Workstream F — security engineering

Things a research prototype has no answer for:

- **F1 key entropy & extraction.** AES-256 key from `os.urandom`. What happens if
  the key is weak, reused, or partially leaked? Is there a KDF?
- **F2 keyless detection.** Could an attacker find the key by searching grid widths?
  LWE's grid width is derived from `HMAC(key, layer_id)` — is the *scale* recoverable
  from the weights alone? If an attacker recovers the scale they may not need the key
  to read bits, which breaks confidentiality.
- **F3 multiple payloads.** What if two payloads share a model? Collision and
  crosstalk behaviour is unknown.
- **F4 capacity limits.** What is the true maximum, and what breaks first —
  QACI allocation, PER, or detectability?
- **F5 malformed input.** Recovery from a partially-written payload.

**F2 is the one I'd do first.** It is a potential break of the security property and
it is cheap to test.

---

## 7. Workstream G — the Pareto frontier

Stop treating stealth and robustness as separate gates. Map the frontier:

- x-axis: mean perturbation magnitude
- y-axis: detector accuracy
- marker: BER @ σ=0.001

Plot every strategy × every parameter. The publishable claim is the *shape* of this
frontier and where sign and LWE sit on it — not two isolated PASS/FAIL cells. The
grid-width sweep already hints at it: 0.002 undetectable but fragile, 0.05 robust but
detected, 0.005–0.020 both.

---

## 8. Workstream H — productization

### H1. Delta-only distribution — the key reframing

The embedding changes 10,256 of ~811M values (0.0013%). So the product is **not** a
modified base model. It is a delta, `W_stego − W_clean`, shipped like a LoRA adapter:

- ~0.2% of model size instead of 100%
- auditable: the recipient can inspect exactly what changed
- no modified base model enters any registry
- recipient reconstructs locally; nothing is uploaded

This also makes the security story much cleaner — you distribute a *patch*, not a
weaponised checkpoint.

### H2. Recipient-side tool

A CLI/library that takes base model + delta + key → recovers the payload. Needs to be
usable by someone who did not build it.

### H3. Statistical guards on the delta itself

If the delta is the deliverable, it should carry its own integrity metadata: hash,
carrier count, payload length. Prevents silent corruption.

---

## 9. Risk register

| risk | severity | note |
|---|---|---|
| **Smuggling past safety scanning** | **high** | A hidden payload extracted by the model at inference is a direct route around output filters. Any product must be able to answer "can this be abused?" before shipping. |
| **Evading model fingerprinting / provenance** | **high** | Weights-distribution changes are used for provenance in some regimes. Hidden data in weights conflicts with that. |
| **IP / licensing** | **high** | Redistributing derived weights may violate base-model licences. Unreviewed. |
| Regulatory (EU AI Act-style obligations) | medium–high | Needs a compliance answer before distribution. |
| Dual-use by others | medium | The capability exists regardless of what we ship. |

**This is a research programme, and legitimate as such.** It becomes a product only
after an explicit misuse assessment. Flagging once, here, so it is a decision rather
than a surprise.

---

## 10. What I would not do

- **Add more models.** 5 families, 22–42 layers. Low marginal value versus D/E/F.
- **Build a product before D1/D2/E1.** Publishing a stealth claim that a
  cross-scheme or blind detector defeats would be the worst outcome available.
- **Rename `LWE-Inspired` unilaterally.** It is a claimed contribution; renaming it
  is the authors' call, not a code fix.
- **Touch the 0.002 threshold.** An author decision, recorded not applied.

---

## 11. How to resume

```bash
cd nes-llm

# one command, any subset
../.venv/bin/python run_nes_experiments.py --models Qwen/Qwen2.5-3B
../.venv/bin/python run_nes_experiments.py --models Qwen/Qwen2.5-3B --exp exp3 exp6

# audit, verify, report
../.venv/bin/python run_nes_experiments.py --audit
../.venv/bin/python check_consistency.py
../.venv/bin/python tests/test_strategy_registry.py
../.venv/bin/python tests/test_quantization_adapters.py

# diagnostics
../.venv/bin/python -m src.model.verify_residual_cache <model> <family> <n_layers>
../.venv/bin/python -m src.experiments.exp12_lwe_cross_model --models <id>
../.venv/bin/python -m src.experiments.exp2_criterion_calibration
../.venv/bin/python src/steganalysis/exp7_neural_parameter_study.py
```

Machine: 26 GB RAM, ~30 GB MPS ceiling, 208 GB free, 7 models + GPTQ/AWQ cached.
Residuals 1.0–8.6 GB; run **one model per process**.

### Ground rules that must survive

1. A completed cell (PASS **or** FAIL) is skipped unless `--force`. Never rerun a
   FAIL until it passes.
2. Thresholds live in `experiment_registry.THRESHOLDS`. No experiment edits its own.
3. A missing artifact is reported missing. Never defaulted to PASS.
4. `EXP` results are written to `RESULTS_DIR`. Two modules got this wrong
   (`nes-llm/results` vs repo-root `results/`); use `paths.RESULTS_DIR`.
5. Any experiment that trains a detector needs ≥400 pairs. At 120 every variant
   scores exactly 50% and the study is void.
6. Verify a dequantizer against the FP16 reference before using it (corr > 0.95).
7. Never reshape a mismatched matrix — transpose or raise.

---

## 12. Suggested order

Interchangeable, but this sequence front-loads the cheapest results that could
invalidate later work.

| # | Item | Why here |
|---|---|---|
| 1 | **F2** key/scale recoverability | cheap; potential break of the security property |
| 2 | **D2** blind-patch adversary | cheapest test of the central claim |
| 3 | **B** LWE perplexity | biggest hole; invalidates the strategy choice if it fails |
| 4 | **D1** cross-scheme detector | the claim's main weakness |
| 5 | **A1** QAE adapter + round trip | adds two strategies cheaply |
| 6 | **A3** strategy × model matrix (3 models) | the breadth deliverable |
| 7 | **C** hybrids | most interesting science |
| 8 | **E** model surgery | determines viability |
| 9 | **G** Pareto frontier | the strongest publishable framing |
| 10 | **A4** consolidation | cleanup; do before H |
| 11 | **H** delta productization | only after 1–8 hold |
| — | misuse assessment | gate before any distribution |