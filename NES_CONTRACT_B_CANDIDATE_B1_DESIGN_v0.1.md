# NES Contract B Design Note
## Candidate B1: Keyed, distortion-aware paired-code embedding in NF4 indices

**Date:** 2026-10-09  
**Status:** proposed design for a small feasibility prototype; not implemented or measured.  
**Purpose:** specify a concrete candidate before coding, consistent with the frozen artifact-only receiver contract.

---

## 1. Why this candidate

The existing NES sign/residual path changes a floating-point residual and recovers it using reference/residual context. The unresolved question is whether a payload can instead be recovered from the final serialized quantized artifact alone.

Candidate B1 embeds each payload bit in the parity of a quantized code index, but only in carefully selected pairs of neighboring codebook entries. The carrier position is chosen using a deterministic key-derived ordering, while eligibility and distortion cost are derived from properties that remain unchanged when the code index flips within its pair.

This is a testable candidate—not a novelty claim and not an assertion that the method is already robust or undetectable. A literature review must check related quantized-network steganography and code-index LSB methods before making a novelty claim.

## 2. Artifact and codebook representation

For a 4-bit NF4 code index \(q_i \in \{0,\ldots,15\}\), define its pair ID and within-pair bit:

\[
p_i = \left\lfloor \frac{q_i}{2} \right\rfloor,
\qquad
b_i = q_i \bmod 2.
\]

The two indices in pair \(p\) are \(2p\) and \(2p+1\). To write bit \(m_i\), preserve the pair ID and select:

\[
q_i' = 2p_i + m_i.
\]

Thus \(p_i\) remains invariant and \(q_i' \bmod 2=m_i\). The receiver can recover the bit by reading the final code index parity, while the pair identity remains stable for re-creating the carrier set.

**Important format requirement:** implement this against the actual serialized NF4 code representation and its official/runtime-compatible unpack/repack path. Do not assume a generic nibble layout. First establish bit-exact round-trip between the repository's format reader and a real saved/reloaded checkpoint. If this cannot be done without modifying unrelated production code, use an isolated experiment adapter and report the limitation.

## 3. Distortion-aware eligibility

For codebook values \(c_0,\ldots,c_{15}\), define the pair gap:

\[
g_p = |c_{2p+1} - c_{2p}|.
\]

If a quantization block has scale \(s_i\), a first-order code substitution cost is:

\[
d_i = |s_i| g_{p_i}.
\]

This is a local weight-space proxy, not a full model-sensitivity metric. Where the format uses nested/double-quantized scales, use the dequantized effective scale from the serialized artifact and record the exact calculation. Do not substitute a scale from the original FP16 checkpoint.

Define a pre-registered eligibility threshold \(d_i \leq \tau\) and/or a fixed eligible pair-ID set based on the NF4 codebook gaps. Select \(\tau\) on a calibration split or via a pilot grid, then freeze it before confirmatory measurements. Do not tune it on the final detector test set or after seeing payload recovery results.

### Stable carrier-set requirement

Eligibility must be invariant under the allowed bit write. Since \(q_i\) and \(q_i'\) share \(p_i\), the pair-gap term is invariant. The scale must also remain unchanged by the code substitution. Carrier selection must not depend on the current within-pair parity, because that parity is precisely what the payload changes.

## 4. Deterministic keyed allocation

Inputs: final artifact's code indices/scales, declared key \(K\), stable tensor names and shapes, target carrier count \(N\), frozen eligibility rule.

For every eligible coordinate \(i\), compute a deterministic pseudorandom rank:

\[
r_i = \operatorname{HMAC\!-\!SHA256}
(K,\ \text{model-format-id} \Vert \text{tensor-name} \Vert i).
\]

Sort eligible coordinates by \(r_i\), with a stable tie-breaker, and select the first \(N\). The same key, model-format identifier, tensor names/shapes, and final artifact reproduce the same set because eligibility uses pair ID and scale, both invariant under within-pair writes.

The embedding process must first enumerate and freeze the carrier set from the clean quantized artifact, then write bits by changing only the within-pair bit. The receiver recomputes eligibility and ranks from the stego artifact and must obtain the identical positions. This equality is a required test, not an assumption.

The key is a carrier-selection mechanism. Do not claim cryptographic secrecy from it alone; payload confidentiality and integrity require separate authenticated encryption, and detectability must be tested independently.

## 5. Payload and capacity

For the feasibility prototype:

- Use the frozen contract's 10,000-bit application payload: exactly 1,250 bytes, generated deterministically for reproducibility.
- Record any application-level framing, length header, nonce/IV, authentication tag, and encrypted-stream overhead separately.
- Compute the actual required carrier count from the final framed/encrypted stream. Do not call 10,000 application bits the total embedded stream if it is longer.
- If the current production encoder only supports text at its public interface, implement a local experiment adapter or test harness; do not quietly change shared production behavior.
- Capacity is the number of eligible carriers available under the frozen distortion rule, minus any explicitly specified protocol overhead. Report the actual number; do not infer capacity from parameter count alone.

## 6. Serialization and receiver isolation

The sender:
1. Load the exact Qwen2.5-3B checkpoint revision.
2. Quantize using the registered NF4 configuration.
3. Save the clean quantized artifact and compute its SHA-256.
4. Read actual serialized code indices and effective scales.
5. Determine eligible carriers and keyed ordering.
6. Embed the encrypted/framed stream by changing the within-pair bit only.
7. Save the stego artifact through the format-supported serialization path.
8. Record the artifact SHA-256, changed code count, carrier count, and weight-space distortion.

The receiver:
1. Load only the stego artifact, declared key, and frozen public protocol/configuration.
2. Do not load the original FP16 weights, residual cache, clean checkpoint, delta, or any sidecar containing carrier positions or payload bits.
3. Recreate eligible coordinates and keyed ordering from the stego artifact.
4. Verify the recomputed carrier positions exactly match the writer's committed test record during development; the final receiver itself must not depend on that record.
5. Read \(q_i \bmod 2\) from each selected carrier in the declared order.
6. Recover the framed encrypted stream, authenticate/decrypt it, and compare the application payload.

A run is invalid for Contract B if the extraction code accidentally reads the in-memory pre-serialization tensor or accesses the original model/residual cache.

## 7. Pilot sequence and stop/go criteria

### B1.1 — Representation round-trip (no payload)

- Select a small real NF4 tensor/layer from the repository's actual saved artifact.
- Decode code indices, repack them unchanged, save and reload.
- Require exact equality of all code indices and all required quantization metadata.
- Confirm the dequantized output is bit-exact where the runtime path promises it, or document a justified tolerance before embedding.
- Stop if the adapter cannot round-trip safely.

### B1.2 — Pair-write invariants

On a small isolated tensor, flip the within-pair bit for a deterministic set of eligible indices. Assert:
- all modified codes remain in 0..15;
- every changed code retains the same pair ID;
- every unselected code is unchanged;
- scales and quantization metadata are unchanged;
- candidate eligibility and keyed carrier positions recompute identically from the stego codes.

### B1.3 — Short payload end-to-end

Use a tiny fixed payload first. Serialize the stego artifact, terminate the writer process, start a clean reader process, and recover from the saved artifact only. Require exact bit recovery, authenticated decryption, and application-message equality.

### B1.4 — 10,000-bit pilot and controls

Only after B1.1–B1.3 pass:
- run the frozen 10,000-bit application payload;
- compare with clean quantized and sham/control artifacts;
- report eligible carrier count, changed codes, distortion, perplexity/reconstruction control, and initial detector performance;
- test the historical noise gates only after clean artifact-only recovery is established.

**Stop** if serialization/reload changes the carrier representation unexpectedly, carrier positions drift, or clean artifact-only recovery fails. Preserve the failed artifact and diagnose the failure rather than relaxing the contract.

## 8. Main scientific risks

1. **Utility:** even adjacent NF4 code values can have substantial gaps, especially near the extremes. Distortion-aware eligibility may sharply reduce available carriers.
2. **Capacity:** a strict cost threshold may leave fewer eligible carriers than the framed/encrypted payload needs.
3. **Detectability:** index parity may induce statistical or learned patterns even if the weight-space distortion is small. Chance-level results from the existing LWE-labelled scheme do not predict this candidate's detectability.
4. **Allocation stability:** any eligibility feature that changes after writing makes the receiver unable to reproduce the carrier set. Pair ID and scales are intended invariants; they must be tested.
5. **Artifact compatibility:** bitsandbytes NF4 state serialization, nested quantization metadata, and checkpoint loading must be handled by a format-aware adapter. A tensor-level proof is insufficient.
6. **Model utility:** small local codebook gaps are only a weight-space proxy; they do not guarantee small perplexity or task-performance impact.
7. **Security:** a keyed carrier schedule does not by itself prove that an attacker cannot infer positions or decode the channel. Wrong-key behavior, public attacks, and authenticated encryption need separate tests.
8. **Prior art:** code-index parity / quantized-weight LSB embedding may overlap existing literature. Novelty cannot be asserted until that literature is reviewed.

## 9. Required output from the first prototype

The prototype should produce a machine-readable JSON artifact with:
- model ID/revision and Git SHA;
- quantization format/configuration and library versions;
- codebook and packing adapter identifiers;
- clean/stego artifact hashes;
- total code count, eligible count, selected carrier count, changed-code count;
- carrier-set hash and independent writer/reader carrier-set equality;
- application payload bits, encrypted/framed bits, carrier bits;
- code-index BER and application-payload equality;
- changed-code fraction, code-index distortion and dequantized weight-space distortion;
- serialization/reload status;
- explicit receiver-input manifest;
- status (`PASS`, `FAIL`, `NOT_RUN`, `SKIPPED`, `ERROR`) and diagnostic reason.

## 10. Decision

**Candidate B1 is ready to prototype only after the real NF4 serialization adapter is understood.** It is not yet a demonstrated method, and its novelty is not established. First implement B1.1 in an experiment-local module with tests; do not modify the production residual/sign path. If the format round-trip is exact and the stable-carrier invariants hold, proceed to a short payload. Otherwise, stop and revise the representation choice before running a model-scale experiment.
