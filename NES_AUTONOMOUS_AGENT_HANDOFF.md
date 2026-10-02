# NES Multi-Model Research --- Autonomous Agent Handoff & Completion Plan

**Project:** Neural Embedding Steganography (NES)\
**Purpose:** Autonomous coding/research-agent handoff for finishing,
validating, and automating the NES multi-model research pipeline.

> **Source basis:** `NES_MultiModel_Experiment_Guide.pdf`, the
> repository structure and experiment artifacts discussed during the
> research work, and observed experiment results. Where the guide and
> current implementation differ, this document records the difference
> rather than silently replacing one with the other.

------------------------------------------------------------------------

# 1. Research objective

NES embeds encrypted information into **quantization residuals of
language-model weights**.

Core flow:

``` text
FP16 model
   ↓
NF4 quantization
   ↓
NF4 dequantization
   ↓
R = W_FP16 - W_NF4_dequantized
   ↓
QACI carrier selection
   ↓
AES-256-GCM encrypted payload bits
   ↓
sign-based embedding into selected residuals
   ↓
embedded weights/residuals
   ↓
extraction + decryption
   ↓
BER / fidelity / robustness / security measurements
```

The original system was centered on TinyLlama/NF4. The multi-model
expansion tests whether the method generalises across several
open-source LLM families and whether it remains usable under other
quantization formats.

The final goal is **one reproducible pipeline** that can run the
complete experiment suite without one experiment-specific fix breaking
another experiment.

------------------------------------------------------------------------

# 2. Non-negotiable engineering rule

## Never distort shared production code for one experiment

If one model or experiment needs a special fix:

-   do not patch shared production code merely to make that experiment
    pass;
-   create an experiment-specific adapter/runner/helper;
-   preserve the common residual, QACI, embedding, encryption,
    extraction, and validation path;
-   reuse validated artifacts when appropriate;
-   keep the final pipeline capable of running every experiment/model.

This rule exists because the project is being completed iteratively and
shared-code changes can silently break earlier experiments.

A good architecture is:

``` text
                         Common NES Pipeline
                                │
             ┌──────────────────┼──────────────────┐
             │                  │                  │
          Exp 1–3            Exp 4–7              Exp 8
       core correctness   experiment eval      aggregation
             │                  │                  │
             └──────────────────┼──────────────────┘
                                │
                               Exp 9
                         GPTQ / AWQ extension
```

------------------------------------------------------------------------

# 3. Repository structure

The outer repository is approximately:

``` text
nes-research/
├── NES_MultiModel_Experiment_Guide.pdf
├── results/
├── artifacts/
└── nes-llm/
    ├── src/
    ├── scripts/
    ├── results/
    ├── artifacts/
    ├── models/
    └── notebooks/
```

The actual Python project root is:

``` text
nes-research/nes-llm/
```

Important source areas:

``` text
src/
├── core/
│   └── types.py
├── model/
│   ├── model_loader.py
│   ├── registry.py
│   ├── exp8_real_cross_model_table.py
│   └── exp8_result_adapter.py
├── carrier_intelligence/
│   ├── qaci_pipeline.py
│   ├── layer_profiler.py
│   ├── carrier_scheduler.py
│   ├── carrier_selector.py
│   ├── feature_extractor.py
│   ├── normalizer.py
│   └── quality_score.py
├── embedding/
│   ├── intelligent_embedder.py
│   ├── sign_strategy_v2.py
│   └── ...
├── extraction/
│   └── decrypt_pipeline.py
├── evaluation/
│   ├── fidelity_validator.py
│   ├── robustness_validator.py
│   └── exp5_model_builder.py
└── steganalysis/
    ├── security_validator.py
    ├── exp7_security.py
    ├── exp7_neural_detector.py
    ├── exp7_neural_detector_save_runner.py
    ├── build_exp7_detector_dataset.py
    ├── neural_detector.py
    └── kl_detector.py
```

Experiment scripts are under `scripts/`. Structured outputs belong under
`results/`; large binary datasets belong under `artifacts/`.

------------------------------------------------------------------------

# 4. Core architecture

## 4.1 Model loading

Primary component:

``` text
src/model/model_loader.py
```

It loads an NF4 model, an FP16 reference model, and the tokenizer.

NF4 uses BitsAndBytes with 4-bit NF4 quantization, double quantization,
and FP16 compute.

Important Apple-Silicon/MPS fix already established:

-   packed NF4 weights and full quantization state are kept on CPU for
    dequantization;
-   giant dequantized tensors must not be moved to MPS unnecessarily;
-   this avoids errors such as `RuntimeError: Invalid buffer size`
    caused by enormous MPS allocations.

Do not undo this optimization.

## 4.2 Residual extraction

The intended API is:

``` python
extract_residuals(nf4_model, fp16_model, family) -> dict
```

The output is:

``` text
layer_id -> flattened residual tensor
```

with:

``` text
R_i = W_FP16_i - dequantize(W_NF4_i)
```

The current validated implementation returns the residual dictionary. Do
not reintroduce an obsolete three-value unpacking pattern.

------------------------------------------------------------------------

# 5. Target models

Current real Exp8 target list:

  Model                                  Family      Expected layers
  -------------------------------------- --------- -----------------
  `meta-llama/Llama-3.1-8B`              llama                    32
  `mistralai/Mistral-7B-v0.3`            mistral                  32
  `google/gemma-2-9b`                    gemma                    42
  `Qwen/Qwen2.5-7B`                      qwen                     28
  `Qwen/Qwen2.5-3B`                      qwen                     36
  `TinyLlama/TinyLlama-1.1B-Chat-v1.0`   llama                    22
  `microsoft/Phi-3-mini-4k-instruct`     phi3                     32

The real orchestrator records both expected and actual layer counts.
Never silently assume the hardcoded value is correct.

Dense architectures currently exercised successfully include Llama,
Mistral, Gemma, Qwen and Phi. MoE/Falcon support is not equivalent
merely because registry entries exist; it needs true validation.

------------------------------------------------------------------------

# 6. Model registry

The guide requires architecture-specific layer resolution instead of
scattering architecture assumptions throughout the code.

Typical dense path:

``` text
model.layers[i].mlp
```

Falcon differs:

``` text
transformer.h[i].mlp
```

MoE models such as Mixtral require expert-aware handling.

The agent must distinguish **implemented registry entries** from
**experimentally validated architecture support**.

------------------------------------------------------------------------

# 7. QACI --- Quality-Aware Carrier Intelligence

QACI decides where payload bits should be embedded in the residuals.

Conceptual flow:

``` text
Residuals
  ↓
LayerProfiler
  ↓
features / quality statistics
  ↓
QualityScore
  ↓
Carrier selection
  ↓
CarrierScheduler
  ↓
integer bits-per-layer allocation
```

Important components include `qaci_pipeline.py`, `layer_profiler.py`,
`carrier_scheduler.py`, `carrier_selector.py`, `feature_extractor.py`,
`normalizer.py`, and `quality_score.py`.

### QACI status

QACI is an implemented and tested subsystem.

Validated pieces include:

-   fractional layer-position bias rather than assuming 32 layers;
-   Hamilton/largest-remainder style integer allocation;
-   quality-score ranking;
-   residual-driven carrier selection.

A scheduler test using quality weights `[0.2, 0.8, 0.5, 0.1]` and
capacity 100 produced `[2, 74, 23, 1]`.

The current default gamma is `2.5`. The guide recommends family-specific
tuning where residual profiles justify it.

------------------------------------------------------------------------

# 8. Embedding and cryptography

`IntelligentEmbedder` is primarily wired to the sign-based embedding
strategy.

Conceptually:

``` text
bit 1 → positive residual
bit 0 → negative residual
```

with a minimum-magnitude guard for near-zero values.

AES-256-GCM is implemented using:

-   32-byte key,
-   fresh 12-byte nonce,
-   authenticated encryption.

Important terminology limitation:

The current LWE component is **LWE-inspired**, using an HMAC-SHA256
keyed interval/grid mechanism. It should not be described as a complete
mathematically validated LWE cryptosystem unless replaced and
independently validated.

QRNG infrastructure exists, but a working production QRNG provider has
not been established.

------------------------------------------------------------------------

# 9. Experiment order

The guide's required sequence is:

``` text
1. Required code/model adaptation
2. Exp1 Architecture Probe
3. Exp2 Residual Fingerprint
4. Exp3 Clean BER
5. Exp4 Capacity Curve
6. Exp5 Fidelity
7. Exp6 Robustness
8. Exp7 Security
9. Exp8 Cross-model Table
10. Exp9 GPTQ/AWQ
```

The agent may reuse a valid existing artifact instead of rerunning an
experiment, but must preserve dependency order when an experiment
genuinely needs to be executed.

------------------------------------------------------------------------

# 10. Experiment 1 --- Architecture Probe

**Goal:** confirm that a model loads, residuals can be extracted, the
correct modules are resolved, and QACI accepts the resulting residual
dictionary.

**Gate:** residual extraction succeeds and QACI runs.

This is a structural validation, not a security/fidelity claim.

**Status:** validated for the dense models exercised so far.

------------------------------------------------------------------------

# 11. Experiment 2 --- Residual Fingerprint

**Goal:** profile residual distributions per layer/model.

Expected artifact:

``` text
residual_profile_<model>.json
```

Important fields:

-   mean magnitude,
-   standard deviation,
-   maximum magnitude,
-   entropy,
-   quality score,
-   position bias,
-   adjusted quality.

Guide criterion:

``` text
mean mag_mean > 0.002
for at least 80% of layers
```

Profiles should be preserved permanently because they inform later
alpha/gamma/payload decisions and the paper's architecture section.

**Status:** profiles have been generated for models, including Gemma;
exact full target-model artifact coverage must be audited rather than
assumed.

------------------------------------------------------------------------

# 12. Experiment 3 --- Clean BER

**Goal:** fundamental correctness of embed → extract → decrypt.

Required:

``` text
BER = 0
recovered message == original message
```

Previously validated clean BER=0 results include:

-   Llama-3.1-8B
-   Mistral-7B-v0.3
-   Gemma-2-9B
-   Gemma-2-2B
-   Qwen2.5-7B
-   Qwen2.5-3B

**Status:** complete for the tested models.

------------------------------------------------------------------------

# 13. Experiment 4 --- Capacity Curve

**Goal:** determine how much payload can be embedded at BER=0.

The guide asks for at least 500,000 bits and the maximum tested capacity
per model.

Current Qwen2.5-3B evidence:

``` text
5,000,000 bits   → BER 0
10,000,000 bits  → BER 0
1,000,000,000    → not completed/tested
```

Therefore the correct claim is:

> **Maximum tested payload at BER=0 = 10 Mb.**

Do not call this the absolute maximum capacity.

**Status:** partial; Qwen has been tested through 10M, but a complete
per-model capacity table still needs to be assembled.

------------------------------------------------------------------------

# 14. Experiment 5 --- Fidelity

**Goal:** measure model-quality degradation after embedding.

Primary gate:

``` text
PPL degradation < 2%
```

The correct real-PPL protocol distinguishes:

1.  NF4 baseline PPL;
2.  reconstruction-control PPL;
3.  embedded-model PPL.

This matters because NF4 → reconstructed FP16-like weights can change
PPL independently of embedding.

### Completed Qwen2.5-3B real result

``` text
NF4 baseline PPL           12.4707
reconstruction control     11.3494
embedded PPL               11.3488
embedding-specific delta   -0.0052866%
absolute degradation       0.0052866%
threshold                  2%
```

The approximately 9% NF4-to-reconstruction difference must **not** be
attributed to embedding.

The current Exp8 adapter consumes this already-completed Qwen result
rather than rerunning the expensive evaluation.

**Status:** Qwen complete/pass; full multi-model fidelity coverage
remains incomplete.

------------------------------------------------------------------------

# 15. Experiment 6 --- Robustness

**Goal:** characterize BER under Gaussian perturbation.

Sigma grid:

``` text
0
0.0005
0.001
0.002
0.005
0.010
0.020
```

Primary gates:

``` text
BER @ sigma=0.001 < 0.02
BER @ sigma=0.002 < 0.10
```

Recent Qwen2.5-3B curve:

``` text
0.0     → 0
0.0005  → 0
0.001   → 0
0.002   → 0
0.005   → ~3.9e-05
0.010   → ~0.0190
0.020   → ~0.1300
```

Exact non-gate values can vary slightly across stochastic reruns.

**Status:** Qwen gate validated; complete multi-model artifact coverage
still needs auditing/completion.

------------------------------------------------------------------------

# 16. Experiment 7 --- Security

Exp7 must distinguish **statistical security** from **neural-adversary
security**.

## 16.1 Statistical security

Measurements:

-   KL divergence,
-   sign bias,
-   mean shift,
-   standard-deviation shift,
-   simple statistical detector accuracy.

Gate:

``` text
KL < 0.05
statistical detector < 0.55
```

Recent Qwen2.5-3B real Exp8 statistical result:

``` text
KL ≈ 8.885e-05
sign bias ≈ 0.01475
mean shift ≈ 3.009e-07
std shift ≈ 2.794e-09
statistical detector ≈ 50.27%
```

This passes the statistical gate.

## 16.2 Neural detector

The current real neural-detector dataset uses:

-   Qwen2.5-3B,
-   500 clean samples,
-   500 stego samples,
-   4096-value patches,
-   10,000-bit payload,
-   paired clean/stego examples,
-   split by `sample_id` to keep pairs together.

Detector architecture:

``` text
4096 → 256 → 64 → 1
```

with ReLU, dropout, BCE-with-logits, Adam.

Saved result:

``` text
accuracy = 70.5%
TN = 73
FP = 27
FN = 32
TP = 68
threshold = 55%
status = FAIL
```

This is a **real completed experiment with a FAIL gate**, not a missing
experiment.

Do not weaken the threshold or modify the detector merely to produce
PASS.

### Important interpretation

A paired audit found:

``` text
500 pairs
2,048,000 total values
12,623 changed values
changed ratio ≈ 0.616%
mean absolute pair difference ≈ 0.000322455
```

The embedding is sparse and small, yet a carrier-centered neural
detector can exploit the signal.

The current dataset also reuses underlying Qwen residual structures
heavily, so it should be described as a **carrier-centered neural
steganalysis stress test**, not automatically as a universal adversarial
detector result.

A future fairer adversarial experiment can be added as a separate
experiment-specific file.

------------------------------------------------------------------------

# 17. Experiment 8 --- Real Cross-Model Table

The guide's original Exp8 example uses `NESBenchmark`, but that
benchmark uses synthetic residuals and is therefore not authoritative
for the final real-model table.

The project therefore has:

``` text
src/model/exp8_real_cross_model_table.py
```

This orchestrator intentionally does not modify:

-   `NESBenchmark`,
-   the production embedding path,
-   model loading,
-   earlier experiment scripts.

Its gates are:

``` text
G2 QACI allocation
G3 clean BER
G4 real PPL
G5 robustness
G6 statistical security
G6 neural detector
```

Missing artifacts are reported as `NOT_RUN`; they are never converted to
PASS.

### Current Qwen2.5-3B state

Latest observed table:

``` text
G2 QACI          PASS
G3 BER           PASS
G4 PPL           PASS
G5 robustness    PASS
G6 statistical  PASS
G6 neural        NOT_RUN
overall          INCOMPLETE
```

This is an orchestration/artifact-discovery issue because the real Exp7
neural detector result already exists.

After fixing `exp8_result_adapter.py` to locate the saved result,
expected state is:

``` text
G6 neural = FAIL
accuracy  = 70.5%
overall    = FAIL
```

A completed failing experiment must be represented as FAIL, not
INCOMPLETE.

------------------------------------------------------------------------

# 18. Exp8 result adapter

File:

``` text
src/model/exp8_result_adapter.py
```

Responsibilities:

-   provide completed Exp5 results;
-   locate saved Exp7 neural-detector results;
-   prevent expensive experiments from being silently rerun;
-   keep Exp8-specific compatibility logic isolated.

It must retain both:

``` python
get_exp5_result()
find_neural_detector_result()
```

Do not replace the adapter with only the neural lookup function.

The neural result currently exists at:

``` text
nes-llm/results/exp7_neural_detector_results.json
```

Path discovery should be based on `Path(__file__).resolve()` so running
from `nes-research/` versus `nes-research/nes-llm/` does not change
artifact discovery.

------------------------------------------------------------------------

# 19. Experiment 9 --- GPTQ/AWQ

**Status: not completed.**

Goal: test whether NES works beyond NF4.

Formats:

``` text
GPTQ
AWQ
```

Target clean result:

``` text
BER = 0
```

The dequantization path must be format-specific. GPTQ may expose
`qweight`, `qzeros`, `scales`, and `g_idx`; AWQ has its own
representation.

Do not pretend GPTQ/AWQ are NF4.

The guide suggests AWQ may require a lower alpha, around `0.10–0.15`,
but this must be experimentally validated rather than blindly applied.

Recommended isolation:

``` text
src/model/quantization/
    nf4.py
    gptq.py
    awq.py
```

or equivalent clean adapters.

------------------------------------------------------------------------

# 20. Current completion matrix

  -----------------------------------------------------------------------
  Area                    Status                  Meaning
  ----------------------- ----------------------- -----------------------
  NF4 loading             COMPLETE                Real loading works

  FP16 loading            COMPLETE                Real reference loading
                                                  works

  MPS-safe dequantization COMPLETE                CPU-safe dequant path
                                                  established

  Residual extraction     COMPLETE                Real residual
                                                  dictionary validated

  Dense model support     PARTIAL/VALIDATED       Several dense families
                                                  tested

  Universal architecture  INCOMPLETE              MoE/Falcon/etc. require
  support                                         true validation

  QACI                    COMPLETE                Core
                                                  pipeline/components
                                                  tested

  Layer position          COMPLETE                No fixed-32 assumption
  normalization                                   in active path

  Carrier scheduling      COMPLETE                Integer allocation
                                                  tested

  Sign embedding          COMPLETE                Clean recovery
                                                  validated

  AES-256-GCM             COMPLETE                Authenticated
                                                  encryption implemented

  LWE                     PARTIAL                 LWE-inspired, not full
                                                  validated LWE crypto

  QRNG                    INCOMPLETE              Provider not
                                                  established

  Exp1                    COMPLETE/VALIDATED      Structural path works
                                                  on tested models

  Exp2                    PARTIAL/GENERATED       Profiles exist; full
                                                  coverage must be
                                                  audited

  Exp3                    COMPLETE for tested     BER=0 obtained
                          models                  

  Exp4                    PARTIAL                 Qwen through 10M
                                                  tested; not absolute
                                                  maximum

  Exp5                    PARTIAL                 Qwen real PPL
                                                  complete/pass; other
                                                  models remain

  Exp6                    PARTIAL                 Qwen gate complete;
                                                  multi-model artifact
                                                  coverage remains

  Exp7 statistical        PARTIAL                 Qwen passes

  Exp7 neural             COMPLETE + FAIL         70.5% detector accuracy

  Exp8 real orchestrator  IMPLEMENTED             Real aggregation path
                                                  exists

  Exp8 Qwen row           INCOMPLETE currently    Neural artifact lookup
                                                  still needs to be
                                                  consumed

  Exp8 all-model table    INCOMPLETE              Full model coverage not
                                                  yet assembled

  Exp9 GPTQ/AWQ           NOT DONE                Must implement/validate

  Final automation        NOT DONE                Immediate engineering
                                                  goal
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 21. Complete vs pass vs incomplete

The agent must keep these states separate.

### COMPLETE + PASS

Experiment ran and met its gate.

### COMPLETE + FAIL

Experiment ran and failed its gate. This is valid scientific evidence.

Example:

``` text
Exp7 neural detector = 70.5% > 55% → FAIL
```

### NOT_RUN / INCOMPLETE

Required experiment or artifact is genuinely missing/unavailable.

### IMPLEMENTED ONLY

Code exists, but experimental evidence is insufficient.

Never infer PASS from implementation alone.

------------------------------------------------------------------------

# 22. Remaining work --- exact order

## Step 1 --- Fix Exp8 neural result discovery

Change only:

``` text
src/model/exp8_result_adapter.py
```

Make it locate:

``` text
nes-llm/results/exp7_neural_detector_results.json
```

Expected:

``` text
neural status = FAIL
accuracy = 0.705
overall = FAIL
```

Do not retrain the detector just to fix this.

## Step 2 --- Build an experiment manifest

Create:

``` text
results/experiment_manifest.json
```

Record at least:

``` text
experiment
model_id
family
status
gate_status
artifact_path
configuration
source
notes
```

## Step 3 --- Audit all existing artifacts

Search:

``` text
results/
artifacts/
```

Create a matrix:

``` text
model × experiment
```

Classify every cell as:

``` text
PASS
FAIL
NOT_RUN
MISSING_ARTIFACT
IMPLEMENTED_ONLY
```

Do not rerun valid existing work unnecessarily.

## Step 4 --- Standardize structured experiment outputs

Every experiment should save JSON with:

``` text
experiment
model_id
family
configuration
metrics
thresholds
status
timestamp
source/artifact
```

Terminal output must not be the only record.

## Step 5 --- Complete Exp8 for all target models

Assemble the final table for:

``` text
Llama-3.1-8B
Mistral-7B-v0.3
Gemma-2-9B
Qwen2.5-7B
Qwen2.5-3B
TinyLlama
Phi-3-mini
```

Reuse valid previous experiment artifacts whenever possible.

## Step 6 --- Evaluate neural security honestly

Preserve the current 70.5% result. If a more representative adversarial
detector is needed, implement it as a separate experiment-specific
protocol with a separate artifact.

## Step 7 --- Complete Exp5 multi-model fidelity

For each model save:

``` text
NF4 baseline PPL
reconstruction-control PPL
embedded PPL
embedding-specific PPL delta
```

## Step 8 --- Complete Exp6 multi-model robustness

Save the complete sigma→BER curve and the required operating-point
gates.

## Step 9 --- Complete Exp7 statistical security per model

Save:

``` text
KL
sign bias
mean shift
std shift
statistical detector accuracy
```

Keep statistical and neural results separate.

## Step 10 --- Implement Exp9 GPTQ/AWQ

Use format-specific quantization adapters and validate clean BER.

## Step 11 --- Build the final automation runner

Desired one-command interface:

``` bash
python run_nes_experiments.py
```

or:

``` bash
python -m src.experiments.run_all
```

It should inspect, validate, skip completed work, execute missing
dependencies, save artifacts, aggregate results, and generate the final
report.

------------------------------------------------------------------------

# 23. Recommended automation architecture

``` text
run_all.py
│
├── environment_check.py
├── manifest.py
├── experiment_registry.py
├── dependency_graph.py
├── artifact_manager.py
│
├── experiments/
│   ├── exp1_architecture.py
│   ├── exp2_fingerprint.py
│   ├── exp3_clean_ber.py
│   ├── exp4_capacity.py
│   ├── exp5_fidelity.py
│   ├── exp6_robustness.py
│   ├── exp7_security.py
│   ├── exp8_cross_model.py
│   └── exp9_gptq_awq.py
│
└── reporting/
    ├── aggregate_results.py
    ├── generate_tables.py
    └── generate_summary.py
```

Avoid a single enormous script containing every experiment's
implementation.

------------------------------------------------------------------------

# 24. Dependency graph

``` text
Model loader
    ↓
Residual extraction
    ├────────→ Exp1
    └────────→ Exp2
                 ↓
              QACI tuning
                 ↓
                Exp3
          ┌──────┼──────┐
          ↓      ↓      ↓
        Exp4   Exp5   Exp6
          └──────┼──────┘
                 ↓
                Exp7
                 ↓
                Exp8
                 ↓
                Exp9
```

Exp8 should aggregate outputs from earlier experiments instead of
silently inventing independent versions of them.

------------------------------------------------------------------------

# 25. Automation safety rules

The autonomous agent MUST:

1.  Never overwrite a completed result without preserving the previous
    artifact.
2.  Never modify shared production code solely to satisfy one
    experiment.
3.  Never turn `NOT_RUN` into `PASS`.
4.  Never turn an actual `FAIL` into `PASS`.
5.  Never call the largest tested capacity the absolute maximum without
    a justified stopping criterion.
6.  Never describe the current LWE-inspired mechanism as full LWE
    cryptography.
7.  Never claim universal architecture support merely because registry
    entries exist.
8.  Save structured outputs for every expensive experiment.
9.  Preserve failures and create separate diagnostic experiments when
    necessary.
10. Before changing shared code, establish that the change is genuinely
    cross-experiment.
11. After shared-code changes, rerun a small baseline test before
    large-model runs.
12. Never silently change research thresholds.
13. Never fabricate missing model results.
14. Never silently delete a failing model from the cross-model table.
15. Keep experiment-specific adapters separate from common production
    logic.

------------------------------------------------------------------------

# 26. MPS / Apple Silicon constraints

The project has been run on Apple Silicon/MPS.

Known failure:

``` text
RuntimeError: Invalid buffer size
```

when enormous dequantized tensors are moved to MPS.

Safe approach:

``` text
packed NF4 + quantization state → CPU
dequantize → CPU
move only manageable tensors when necessary
```

The automation should detect:

``` text
CUDA
MPS
CPU
```

and choose a safe path.

------------------------------------------------------------------------

# 27. Reproducibility requirements

Every experiment should record:

``` text
random seed
model ID
family
layer count
payload size
message/payload configuration
alpha
gamma
noise sigma
number of trials
dataset
learning rate
epochs
batch size
device
relevant software versions
```

This is especially important for stochastic Exp6 and Exp7.

------------------------------------------------------------------------

# 28. Artifact naming convention

Prefer:

``` text
results/
    exp1_<model>.json
    exp2_<model>.json
    exp3_<model>.json
    exp4_<model>.json
    exp5_<model>.json
    exp6_<model>.json
    exp7_<model>.json
    exp7_neural_detector_<model>.json
    exp8_cross_model.json
    exp9_<format>_<model>.json
    experiment_manifest.json
```

Large binary artifacts belong under `artifacts/`.

Avoid ambiguous names such as `final.json`, `new_results.json`, or
`test2.json`.

------------------------------------------------------------------------

# 29. Final deliverables

At completion the repository should contain:

## Core implementation

-   model loader;
-   residual extraction;
-   architecture registry;
-   QACI;
-   embedding;
-   extraction/decryption;
-   cryptographic components;
-   evaluation validators.

## Experimental evidence

-   Exp1 results;
-   residual profiles;
-   Exp3 BER results;
-   capacity results;
-   fidelity results;
-   robustness curves;
-   statistical security results;
-   neural detector results;
-   final cross-model table;
-   GPTQ/AWQ results.

## Reporting

At minimum:

``` text
cross_model_table_real.json
cross_model_table_real.csv
experiment_manifest.json
final_research_summary.md
```

------------------------------------------------------------------------

# 30. Final autonomous workflow

``` text
PHASE A — UNDERSTAND
    ↓
inspect repository
    ↓
read guide
    ↓
locate artifacts
    ↓
read manifest

PHASE B — AUDIT
    ↓
model × experiment matrix
    ↓
PASS / FAIL / NOT_RUN / IMPLEMENTED_ONLY

PHASE C — VALIDATE CORE
    ↓
environment
    ↓
model loader
    ↓
residual extraction
    ↓
QACI
    ↓
embedding/extraction

PHASE D — COMPLETE MISSING EVIDENCE
    ↓
Exp1 → Exp2 → Exp3 → Exp4 → Exp5 → Exp6 → Exp7 → Exp8 → Exp9

PHASE E — AGGREGATE
    ↓
manifest
    ↓
JSON
    ↓
CSV
    ↓
final research summary

PHASE F — CONSISTENCY CHECK
    ↓
no missing required artifacts
no fabricated PASS
no stale results accidentally used
all configurations recorded
all failures preserved
```

------------------------------------------------------------------------

# 31. Immediate priority for the takeover agent

Do this first:

``` text
1. Inspect repository and guide.
2. Locate every existing result/artifact.
3. Inspect the current Exp8 adapter.
4. Fix only the Exp8 neural-result lookup if necessary.
5. Run Exp8 and confirm Qwen neural = FAIL 70.5%, overall = FAIL.
6. Build an experiment/model manifest.
7. Identify genuinely missing cells.
8. Complete only those cells.
9. Finish Exp8 aggregation.
10. Implement Exp9 GPTQ/AWQ.
11. Build the final one-command runner.
12. Generate the final report/table.
```

Do **not** start by rewriting the NES core.

------------------------------------------------------------------------

# 32. Final instruction to the agent

Treat this repository as an existing research system, not a blank
project.

The correct behaviour is:

``` text
AUDIT FIRST
    ↓
REUSE VALIDATED WORK
    ↓
ISOLATE EXPERIMENT-SPECIFIC FIXES
    ↓
COMPLETE MISSING EVIDENCE
    ↓
PRESERVE FAILURES
    ↓
AUTOMATE
    ↓
AGGREGATE
    ↓
REPORT
```

The final result should be a coherent, reproducible NES pipeline capable
of running the complete multi-model experiment suite without one
experiment's fix breaking another experiment.
