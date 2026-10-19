# Prior-Art Review — NF4 Anchor-Preserving Weight Payload Embedding

Search conducted: 2026-10-16/17 (overnight run). Reviewer: automated research agent.
Scope: determine whether the specific `interior_nearest` contribution is already
described or implemented.

**Independence note.** This review was run as Workstream B, independent of
Workstream A. No literature finding was used to choose experimental parameters,
and no experimental result was used to justify a novelty claim. Conclusions here
are provisional research judgements, not legal opinions or guarantees of
acceptance.

## 1. The contribution under test

> A bit-payload embedding method for NF4-quantized LLM weights that uses
> interior-code carriers and a nearest-same-parity mapping, preserves block
> endpoint codes, and achieves exact extraction after repeated
> dequantize/requantize transformations.

Key mechanism elements to match against prior work:

1. Message carrier = **quantized 4-bit code indices** (not float weights).
2. Carrier **eligibility = interior codes only** (NF4 endpoints 0 and 15 excluded).
3. Embedding op = **move carrier to nearest code of the target parity**.
4. **Endpoint/anchor preservation** — never rewrite block endpoint codes.
5. **Exact recovery after repeated requantization** (not just one pass).

## 2. Search strategy

Sources queried: arXiv, ACL Anthology, Semantic Scholar, Google Scholar (via
aggregators), OpenReview, IEEE/ACM landing pages, publisher sites, GitHub.

Topic groups searched:
1. Neural-network weight steganography / model-weight payload embedding.
2. LLM weight watermarking and quantization-based watermarking.
3. Quantization-robust watermarking and exact message recovery.
4. NF4 / bitsandbytes / QLoRA / packed 4-bit and nested quantization.
5. Quantized-code parity, codebook-index embedding, invariant carriers, anchor/endpoint-preserving mappings.
6. Robustness under repeated requantization.
7. Model steganography attacks and defenses.

Terminology note: `anchor preservation` and `interior_nearest` are **not**
standard terms and returned no direct matches. The review therefore compares the
underlying operations (parity-index embedding, quantization index modulation,
endpoint avoidance) rather than names.

## 3. Sources read (primary text, not abstracts only)

Full text or substantial algorithm sections were read for:

- Li et al., *Watermarking LLMs with Weight Quantization* (Findings of EMNLP 2023) — abstract + method (quantization-watermark, int8 hiding).
- Dubin, *Disarming Steganography Attacks Inside Neural Network Models* (arXiv 2309.03071) — full text.
- Guo et al., *Invariant-based Robust Weights Watermark for Large Language Models* (arXiv 2507.08288) — full text incl. method and appendices.
- Luan et al., *Robust and Efficient Watermarking of LLMs Using Error Correction Codes* (PoPETs 2025) — full text.
- Zhang & Koushanfar, *EmMark* (DAC 2024, arXiv 2402.17938) — full text.
- Chen & Wornell, *Quantization Index Modulation* (IEEE Trans. Inf. Theory 2001) — via secondary description + follow-ups; primary PDF located.
- Altun, Sharma & Bocko, *Set Theoretic QIM Watermarking* (ICASSP 2006) — full text.
- *Reversible Quantization Index Modulation for Static DNN Watermarking* (arXiv 2305.17879) — method section.
- *Towards Watermarking of Open-Source LLMs* (arXiv 2502.10525, ICLR 2025 workshop) — full text.
- *TransMark: Lossless high-capacity watermarking for LLMs via neuron permutation invariances* (Elsevier / ScienceDirect, 2026) — abstract + highlights.
- Kumaresam et al., *Neural Network Model Watermarking Using Quantized Least Significant Bits and Pseudorandom Error-Correcting Codes* (Technical Disclosure Commons, defensive publication, 2026) — abstract.
- HiddenLayer / Snyk tensor-steganography write-ups — implementation-level descriptions of float-LSB weight embedding.

## 4. Mechanism-by-mechanism comparison

| Source | What is embedded | Carrier representation | NF4/bnb-specific? | Endpoint/anchor preserved? | Nearest-same-parity interior op? | Exact after requantize? | Requant cycles tested | Capacity / detection / utility | Code | Closest overlap | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Li et al. 2023** (EMNLP-F) | Ownership watermark / trigger | Full-precision weights, planted during quantization; hidden when int8 | No (int8, fp32) | n/a | No | No — watermark works in fp32, hidden at int8 | Single quantization | Watermark, not payload; no repeated requantization | Yes (Quantize-Watermark) | Quantization-based watermarking; **different mechanism** (training-time gradient planting, not code-index parity) | High |
| **Dubin 2023** | Defense against hidden malware | Float-weight LSBs (attack side) | No | n/a | No | n/a (defense) | n/a | Defense; 100% prevention via quantization/noise | No | Confirms float-LSB weight stego exists; **not** a code-index method | High |
| **Guo et al. 2025** (2507.08288) | Per-user key / ownership value | **Model invariants** (QK^T linear constraints) in embedding layer | No | n/a | No | No — score-based (ranking) extraction, not exact bits | Quantization attack tested (4/8-bit), not repeated requant | 100% detection (single-user); 100-bit/layer | No | Robust weight watermark; **different carrier** (invariants, not codes); extraction is statistical | High |
| **Luan et al. 2025** (PoPETs) | Model identifier (codeword) | **Weight permutations** (FIT) | No | n/a | No | Yes for identifier, via GRS ECC | Quantization tested (AutoGPTQ 2–4-bit), single pass | Identifier; 6 models; PPL preserved | No | Uses ECC for robustness; **carrier is permutation, not code parity** | High |
| **EmMark (Zhang & Koushafar 2024)** | Signature (±1) | **Quantized weight parameters** (selected by score) | INT4/INT8 via AWQ/LLM.int8, not NF4 | Not stated (selects endpoints out via scoring) | No — additive ±1 on chosen weights | Extraction rate 100% under overwrite, but **not designed for requantization** | Not tested | 100-bit/layer; PPL preserved | Yes (EmMark) | Closest in **carrier domain** (quantized weights) but embedding is additive, not parity/interior | High |
| **Chen & Wornell 2001 (QIM)** | Arbitrary message bits | Quantization indices of a host signal | No (general) | n/a | **Conceptually related** — embeds by quantizing to a lattice keyed by the bit | Robust to some distortions, not requantization of 4-bit codes | n/a | Foundational theory | n/a | The **nearest-parity idea is a QIM-family concept**; our method is a code-index instance | High |
| **Altun et al. 2006 (QIM)** | Message bits | Quantized statistics (mean) | No | n/a | QIM lattice keyed by bit | Robust to JPEG compression | n/a | 1000 bits | n/a | Confirms QIM parity-index embedding is long-established in image stego | High |
| **R-QIM DNN watermarking (2305.17879)** | Watermark bits | DNN weight samples, QIM quantizer | No | n/a | **Yes — nearest point in Λ0/Λ1 by bit** | Not requantization-focused | n/a | Reversible watermarking | n/a | **Very close conceptual match to the parity-index step** applied to DNN weights | High |
| **Towards Watermarking of OSM (2502.10525)** | Various | Various weight-editing schemes | Mentions NF4 as a quantization method | n/a | No | Finds no scheme durable; quantization not a challenge for *their* watermarks | Quantization as one modification | Survey/eval | No | Establishes "durability" requirement; no code-index parity method | High |
| **TransMark (2026)** | Bits via neuron permutation | **Neuron ordering** in FFN | No | n/a | No | Survives quantization via ECC + permutation | Quantization, noise, pruning, finetune | High capacity; lossless (no param change) | No | High-capacity lossless; **carrier is permutation, not code parity** | Medium (abstract + highlights only) |
| **Kumaresam et al. 2026** (defensive pub.) | ECC-coded owner signature | **Quantized LSBs** of weight params | No | n/a | No | Robust to perturbation/finetune via ECC | Not stated | Ownership; architecture-agnostic | No | Quantized-LSB carrier + ECC; **not** interior/parity | Medium (abstract only) |
| **Snyk / HiddenLayer** | Arbitrary payload/malware | Float-weight LSBs | No | n/a | No | No (breaks on requantization) | n/a | Up to MBs | Yes (blog code) | Float-LSB; **not** 4-bit code-index | High (implementation) |

## 5. Novelty decision

### 5.1 Established overlap (clearly predates our method)
- **Embedding payload bits by modulating quantized indices** is the established
  QIM paradigm (Chen & Wornell 2001; Altun et al. 2006), and has already been
  applied to DNN weights (R-QIM, arXiv 2305.17879). The specific idea of moving
  a sample to the nearest lattice point selected by the bit is QIM's core.
- **Embedding in quantized LLM weight parameters** post-training is established
  by EmMark (DAC 2024), which selects quantized weights by a score and perturbs
  them to carry a signature.
- **Robustness of weight watermarks to quantization** is widely reported
  (EmMark; Luan et al. 2025; OSM survey 2502.10525), though usually for a
  *single* quantization pass and usually as a property of ECC/permutation, not
  of code-index parity.
- **Payload embedding in neural-network weights** broadly (float-LSB stego) is
  well established (Dubin 2023; Snyk/HiddenLayer).

### 5.2 Potential differentiator
The candidate differentiator is **not** "embedding bits in quantized weights"
(QIM + EmMark cover that) and **not** "robustness to quantization" (covered).
The narrow candidate is the combination of:

1. operating on the **NF4 4-bit code indices of a packed, doubly-quantized
   (nested) checkpoint** as the carrier domain, and
2. restricting carriers to **interior codes 1–14** while **never rewriting block
   endpoint codes 0/15**, using a **nearest-same-parity** mapping so that the
   carrier set is invariant under requantization, and
3. reporting **exact (bit-perfect) recovery across repeated
   dequantize/requantize cycles** as the explicit robustness target, with the
   endpoint-preservation effect isolated causally (our Arm E shows the operation,
   not the mask, is operative).

Within the reviewed scope, no source was found that (a) targets NF4 packed code
indices specifically, (b) frames endpoint preservation as the robustness
mechanism, and (c) demonstrates exact recovery over repeated requantization.
This is a plausible, narrow differentiator.

### 5.3 Unresolved novelty risks
- **Terminology risk.** "Anchor preservation" and "endpoint-preserving mapping"
  are not searched terms; equivalent operations may exist under other names
  (e.g., "bin-edge avoidance", "codebook boundary protection"). Not resolved.
- **QIM equivalence risk.** A reviewer may characterize our method as a
  straightforward QIM instance on NF4 codes. Distinguishing it requires arguing
  why interior/endpoint preservation is non-obvious for nested NF4 specifically.
  The causal evidence (Arm E) helps but is on one model/tensor.
- **EmMark proximity.** EmMark already selects quantized weights and avoids the
  extreme quantization level; the delta between "select weights with large |W|"
  and "restrict to interior codes" needs a careful, honest statement.
- **Coverage gaps.** IEEE Xplore / ACM DL full texts were partly inaccessible;
  TransMark and the defensive publication were read only at abstract level.
  Non-English literature not searched.

### 5.4 Decision

**Potentially novel, but important comparisons remain unresolved.**

A defensible *narrow* differentiator appears to exist within the reviewed scope
(NF4 code-index carriers + interior-only/endpoint-preserving + exact recovery
over repeated requantization). However, because (i) the underlying parity-index
embedding is a known QIM concept, (ii) EmMark is close in carrier domain, and
(iii) two sources were only partially accessible, the claim cannot yet be stated
as "first" or "unprecedented." Resolving §5.3 (full-text EmMark/TransMark read,
terminology sweep) is required before any strong novelty statement.

**No claim of "first" or "unprecedented" is made.**

---

# Addendum A - Closest-prior-art completion (2026-10-17)

Continuation of section 5. Full method sections were read for the closest matches.
Access level is labelled per source.

## A.1 New sources read (primary text / method sections)

| Source | Access level | Relevance |
|---|---|---|
| R-QIM, Reversible Quantization Index Modulation for Static DNN Watermarking (arXiv 2305.17879) | Full text read (method + simulations) | Closest conceptual match: nearest-lattice-point QIM embedding applied directly to DNN weight samples |
| FreMark, Frequency-Domain Watermark Embedding in Quantized LLMs (DSPP 2025, Springer LNCS 16177) | Abstract + extended summary (full text paywalled) | DCT-domain watermark in quantized LLMs with explicit quantization-boundary clipping |
| EmMark (DAC 2024) | Full text (previously) | Quantized-weight carrier domain |
| Li et al. 2023 (EMNLP-Findings) | Full text (previously) | Quantization watermark |
| Chen & Wornell 2001 (QIM) | Foundational; method via R-QIM restatement | Parity/lattice embedding theory |
| Altun et al. 2006 (QIM) | Full text (previously) | QIM parity-index embedding |
| Luan et al. 2025 (PoPETs) | Full text (previously) | Weight permutation + ECC, quantization tested |
| Guo et al. 2025 (2507.08288) | Full text (previously) | Invariant-based robust weight watermark |
| TransMark (Elsevier, 2026) | Abstract + highlights | Neuron-permutation high-capacity watermark |

## A.2 Structured comparison against interior_nearest

| Criterion | R-QIM (2305.17879) | FreMark (2025) | EmMark (2024) | Li 2023 | Luan 2025 | interior_nearest (ours) |
|---|---|---|---|---|---|---|
| Carrier domain | Float DNN weight samples | DCT coefficients of weight matrices | Quantized weight params (INT4/8) | Full-precision weights (int8 channel) | Weight permutations (FIT) | NF4 4-bit packed code indices |
| Quantizer / representation | 1-D scalar lattice, step Delta | DCT + clipping to quant bounds | AWQ / LLM.int8 | Uniform int8 | n/a (permutation) | NF4 16-level codebook, blocksize 64, nested |
| Embedding operation | Move sample to nearest point of Lambda_m (parity lattice) | Perturb high-freq DCT coeffs during training | Add +/-1 to selected weights | Gradient-based planting | Permute dims | Move carrier to nearest code of target parity |
| Endpoints / code states excluded | No | Clips to quant boundaries (adherence, not exclusion) | Excludes min/max quant level in scoring | n/a | n/a | Yes - codes 0 and 15 never rewritten; carriers restricted to 1-14 |
| Carrier set stable through requant? | Not addressed | Claims robustness to quantization via clipping | Not designed for requantization | Watermark hidden at int8 by design | ECC recovers identifier | Yes - eligibility invariant under requantization (measured) |
| Extraction | Min-distance decoder | Frequency-domain extraction | Score reproduction + compare | Compare fp vs quant | Linear assignment + GRS decode | Carrier reconstruction from artifact alone; bit parity read |
| Robustness transforms tested | Additive noise | Quantization, training perturbations | Overwriting, re-watermark, forging | Quantization | Quantization (2-4 bit), pruning, finetuning | Repeated dequantize/requantize cycles 0-3 (and 0-5 stress) |
| Model families | MLP, VGG (vision) | LLMs (quantized) | OPT, LLaMA-2 | LLaMA | Llama2/3, 6 models | TinyLlama, Qwen2.5, StableLM-2 |
| Detectability evaluation | No | Not reported | No | No | No | Yes (grouped, 16 features, 3 detectors) |
| Utility evaluation | Train loss / accuracy | PPL | PPL / zero-shot | Task metrics | PPL / HellaSwag | Paired PPL (this run) |
| Exact overlap with interior_nearest | Nearest-lattice QIM step on weights (concept) | Quantized-LLM carrier + boundary handling | Quantized-weight carrier domain | Quantization + weight carrier | ECC robustness framing | - |
| Material difference | Reversible float-lattice, not 4-bit code indices; no endpoint preservation; no requantization claim | DCT domain, training-time, no code-parity, no endpoint exclusion | Additive on selected weights, not parity/interior | Planted before quantization, watermark hidden at int8 | Permutation carrier, no code indices | Code-index parity + interior-only + endpoint preservation + requant-cycle exactness |
| Unresolved question | Whether nearest same-parity code is materially distinct from QIM parity lattice | Full text inaccessible; may contain boundary-exclusion logic | Whether scoring already implies endpoint avoidance | none material | none material | - |

## A.3 Terminology sweep results

- NF4 codebook-index embedding - no direct prior method found.
- interior-code / endpoint-preserving quantization embedding - no direct match; FreMark quantization-boundary adherence is the nearest phrase but is a training-time DCT clipping constraint, not a code-index carrier restriction.
- quantized-weight steganography - returns float-LSB weight stego (Snyk/HiddenLayer) and QIM-family methods.
- nearest same-parity quantization mapping - maps to QIM nearest-lattice-point rule (Chen & Wornell; R-QIM).
- stable-carrier embedding under repeated quantize/dequantize - no prior method that explicitly targets repeated-requantization carrier stability.

## A.4 Revised novelty judgement

**Unresolved, leaning narrow-and-potentially-novel.**

- The parity-index embedding step is not novel: it is QIM (Chen & Wornell 2001) and is already applied to DNN weight samples by R-QIM (2305.17879). A reviewer could characterize interior_nearest as a QIM instance on NF4 codes.
- The combination claimed - NF4 packed code-index carriers + interior-only codes 1-14 + never rewriting block endpoints 0/15 + measured exact recovery across repeated dequantize/requantize cycles - was not found in any reviewed source. FreMark is the closest on quantized-LLM + boundary handling but is DCT-domain and training-time.
- The defensible differentiator is therefore narrower than a new embedding method: it is the specific NF4 code-index formulation with endpoint preservation, whose empirical contribution is exact, idempotent payload recovery under repeated NF4 requantization.

**Novelty risk that remains:** full FreMark text and full EmMark scoring details were not fully read; a QIM-equivalence framing could subsume the parity step. These must be resolved before any strong novelty sentence is written.

**No claim of first or unprecedented is made.**

---

# Addendum B — FreMark elevated overlap risk (2026-10-18)

The DSPP 2025 extended summary (abstract-level access; full text paywalled)
states FreMark's core mechanism verbatim:

> "The key innovation lies in our interval-constrained embedding mechanism,
> which ensures that watermarked parameters remain within quantization-invariant
> intervals, guaranteeing watermark persistence even after quantization."
> — Chen, P., Ding, Z., Wu, Y., Ye, P., *FreMark: Frequency-Domain Watermark
> Embedding in Quantized LLMs*, DSPP 2025, LNCS 16177.

This is a **conceptual cousin** of endpoint/anchor preservation: both keep the
watermarked value inside a region that quantization maps back to itself, so the
mark survives re-quantization. This raises the overlap risk above the level
recorded in Addendum A.

Differences that remain material:
- Domain: FreMark embeds in DCT high-frequency coefficients; `interior_nearest`
  embeds in packed NF4 code indices.
- Timing: FreMark is training-time (periodic perturbations + clipping during
  training); `interior_nearest` is a post-training, artifact-level transform.
- Mechanism: FreMark uses interval-constrained perturbation; `interior_nearest`
  uses interior-code (1–14) nearest-same-parity mapping with explicit endpoint
  (0/15) exclusion.
- Claim scope: FreMark targets robustness to a single quantization step;
  `interior_nearest` is evaluated for byte-idempotent exact recovery across
  repeated dequantize/requantize cycles (measured 0–3, stress 0–5).
- FreMark reports no code-index carrier set, no detectability study, and no
  paired-PPL equivalence test.

**Novelty verdict with this evidence: UNRESOLVED.** The "stay inside the
quantization-invariant interval" idea is now known to predate this work in the
DCT domain. The defensible, still-unmatched element is the specific NF4
code-index formulation (interior-only codes + endpoint exclusion) with measured
idempotent exact recovery under repeated requantization. Full FreMark text is
required before any stronger statement. **No "first"/"unprecedented" claim.**

---

# Addendum C — Terminology sweep resolves toward overlap (2026-10-19)

Follow-up run to close §5.3 (terminology risk) and the QIM-equivalence risk.
Full detail and verbatim quotes: `PRIOR_ART_EVIDENCE_GATE_G_20261019.md`.

## C.1 The parity-neighbor code-index operation is a named published method (NID)

**Liu, Tian, Lu, Chen, "Neighbor-index-division steganography based on QIM
method for G.723.1 speech streams," J. Ambient Intell. Humaniz. Comput. 7(1):
139–147, 2016.**

NID labels each codeword index `i` by `(i − 1) mod k`, partitions the codebook
into sub-codebooks by that index label (not Euclidean distance), and moves a
carrier whose label disagrees with the payload bit to the **neighbor index of the
opposite sub-codebook**. This is the same operation as `interior_nearest`:
code-index carrier, parity/label partition, neighbor-of-opposite-subcodebook
replacement. It predates this work by ~10 years. The same family (CNV-QIM,
Xiao/Huang/Tang GLOBECOM 2008) uses the same "neighbor codeword of opposite
subcodebook" rule.

This **resolves the §5.3 terminology risk toward overlap**: the operation is not
unnamed; it is NID/CNV-QIM in the compressed-speech literature.

## C.2 Invariant-region framing is multiply established

- **HPQIM** (Li & Chang, Warwick TR): histogram-preserving QIM — embed while
  leaving the observable statistic unchanged.
- **Li et al. 2023** (already reviewed): keep fp weights inside the interval that
  int8 quantization maps to the same integer.
- **FreMark 2025** (Addendum B): interval-constrained embedding within
  quantization-invariant intervals.
- **DeepWatermark** (Kuribayashi et al., APSIPA 2020): QIM even/odd (parity)
  selection applied to sampled DNN weights.

"Preserve the quantization-invariant statistic so the mark survives" is thus
established in image, speech, and NN-weight domains before this work.

## C.3 What remains unmatched (narrow)

- NF4 **packed** code indices specifically (blocksize 64, nested) as the carrier.
- **Explicit endpoint (0/15) exclusion** as the robustness mechanism (NID/CNV/
  EmMark do not state endpoint exclusion).
- **Exact, byte-idempotent recovery over repeated dequantize/requantize cycles**
  as the stated robustness target.
- Detectability screening + paired-PPL equivalence reporting (per §5.1 review).

## C.4 Revised novelty judgement

**Narrow, and weaker than previously stated.** The embedding operation is a
published named technique (NID) with established invariant-region cousins. The
defensible claim is a **specific NF4 engineering formulation + an empirical
repeated-requantization exactness result**, not a new embedding principle. A
reviewer can reasonably characterize the method as "NID/QIM applied to NF4 packed
code indices with endpoint exclusion." No "first"/"unprecedented" claim.

FreMark full text (C.5 below) remains inaccessible; that gap is unchanged.

## C.5 Gate G status

**PARTIAL / not PASS.** Evidence is now *stronger that the novelty is narrow*.
Per protocol §3, expansion remains **locked**.

