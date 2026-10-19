# Prior-art evidence log — Gate G follow-up (2026-10-19)

Purpose: close the two open items from `PRIOR_ART_REVIEW.md` §5.3 / Addendum B
(FreMark full text; QIM-equivalence framing) and re-test the "terminology risk"
(equivalent algorithms under different names). All findings below are from
inspectable web searches/extractions run on 2026-10-19; verbatim mechanism quotes
are reproduced so the comparison can be checked.

Tools: tavily search + extract. Not a legal opinion. No claim of completeness.

---

## F1. NID — Neighbor-Index-Division steganography (PRIMARY FINDING)

**Citation.** J. Liu, H. Tian, J. Lu, Y. Chen, "Neighbor-index-division
steganography based on QIM method for G.723.1 speech streams," *Journal of
Ambient Intelligence and Humanized Computing*, vol. 7, no. 1, pp. 139–147, 2016.

**Verbatim mechanism (from the secondary source that cites and restates NID,
Semantic Scholar PDF 44d4/fb56…, and the Wiley AMR-WB paper):**

> "NID assumes that the codewords of neighbor indices (i.e., neighbor positions)
> in a codebook would be close together. Hence the codewords in a codebook can be
> easily separated into subcodebooks according to their indices instead of the
> Euclidean distance."

> "Neighbor Index Division Algorithm … 1. An integer k is selected for embedding
> capacity and label the i-th codeword with digit (i − 1) mod k, respectively.
> 2. All the codewords with same label into a sub codebook are collected and k
> different subcodebooks are produced. The binary secret message is modified into
> k-ary digits …"

> (CNV-QIM, the same family) "when the label of the codeword X, which is
> associated with the cover quantization index Ia, agrees with the secret message,
> Ia remains unchanged, or else it should be replaced with the quantization index
> of the neighbor codeword N(X) which belongs to the opposite subcodebook."

**Why this matters.** This is, operationally, the `interior_nearest` embedding
step: (i) carrier = the *quantized code index*; (ii) partition indices into
sub-codebooks *by index parity/label*; (iii) if the carrier's label disagrees
with the bit, move the carrier to the *neighbor index of the opposite
sub-codebook*. The only differences are domain (speech codec codebooks vs NF4
code indices) and the explicit endpoint (0/15) exclusion, which NID does not
state. NID is from **2016**, well before this work.

Status: the "terminology risk" flagged in §5.3 is now **resolved toward
overlap** — the parity-neighbor code-index operation is a named published
technique (NID/CNV-QIM), not an unnamed coincidence.

---

## F2. HPQIM — Histogram-Preserving QIM

**Citation.** C. Li, C.-C. Chang, "Histogram Preserving QIM-Based Watermarking"
(University of Warwick technical report; also the basis of histogram-preserving
QIM in the image domain).

**Verbatim:**
> "the proposed HPQIM-based watermarking scheme … preserves the statistical
> characteristics of the original histogram of the image after embedding …"

**Why this matters.** It is the "invariant carrier / embedding that leaves the
observable statistic unchanged" idea — the conceptual cousin of endpoint
preservation — established in the image-QIM literature. It reinforces that
"preserve the quantizer-invariant statistic so embedding survives/stealths" is
not new framing.

---

## F3. DeepWatermark — QIM applied to DNN weights (2020)

**Citation.** M. Kuribayashi, T. Tanaka, N. Funabiki, "DeepWatermark: Embedding
Watermark into DNN Model," APSIPA ASC 2020.

**Verbatim:**
> "we focus on the fully-connected layers and apply a quantization-based
> watermarking method to the weights sampled from the layers … In the QIM method,
> x is quantized into the nearest even/odd point from quantization grids
> determined by a quantization step δ. The selection of even and odd point is
> determined by w."

**Why this matters.** Even/odd (parity) selection on *sampled DNN weights* via
QIM, 2020 — earlier than R-QIM (2023) already in the review. Parity-on-weights is
thus multiply established.

---

## F4. Quantization-interval invariance for NN weights (2023, already in review)

**Citation.** Li et al., "Watermarking LLMs with Weight Quantization," EMNLP
Findings 2023 (already reviewed).

**Verbatim (from the project README):**
> "We achieve this by exploiting the property that 32-bit floating-point numbers
> in a certain interval can be mapped to the same int8 integer by the
> quantization process. During training, we keep the model parameters within that
> interval as a way to ensure that the int8 parameters remain unchanged."

**Why this matters.** "Keep the value inside the interval that quantization maps
back to itself" — i.e. quantization-invariant interval — applied to NN weights in
2023. Together with FreMark (2025) and F2, the invariant-region idea is
established in both image and NN-weight domains.

---

## F5. FreMark full text

Still paywalled (Springer LNCS 16177, DOI 10.1007/978-981-95-3185-1_14;
published 13 Oct 2025). Only the abstract/extended summary is accessible, as
recorded in Addendum B. **Not resolved.**

---

## Net effect on Gate G

| Element of the candidate differentiator | Prior art found | Verdict |
|---|---|---|
| Carrier = quantized code index | QIM (2001), NID (2016), EmMark (2024) | established |
| Parity/label partition of codebook indices | CNV-QIM, NID (2016) | established |
| Move carrier to neighbor of opposite sub-codebook | CNV-QIM, NID (2016) | established |
| Parity selection on DNN weights | DeepWatermark (2020), R-QIM (2023) | established |
| Keep value inside quantization-invariant region | Li 2023, FreMark 2025, HPQIM | established |
| **NF4 packed code indices specifically** | none found | unmatched |
| **Explicit endpoint (0/15) exclusion** | not stated in NID/CNV/EmMark | unmatched |
| **Exact idempotent recovery over repeated requantization** | none found | unmatched |
| **Detectability + paired-PPL equivalence reporting** | none found (per review) | unmatched |

**Conclusion (unchanged direction, strengthened evidence):** the *mechanism* is
now demonstrably a published, named technique (NID) with an established
invariant-region cousin (HPQIM/FreMark/Li-2023). The only still-unmatched pieces
are the NF4-specific carrier formulation, the explicit endpoint exclusion, and
the repeated-requantization exactness result — i.e. a **narrow engineering
formulation + empirical robustness claim**, not a new embedding principle.

Gate G therefore remains **PARTIAL / not PASS**. It is now *more* clearly a
narrow-novelty situation, not a broad-novelty one. Per protocol §3, expansion
stays **locked**.
