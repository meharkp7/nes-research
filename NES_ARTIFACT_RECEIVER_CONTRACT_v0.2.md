# NES Artifact and Receiver Contract
## Frozen experiment specification — v0.1 (2026-10-09)

**Status:** protocol specification draft for review; not an experimental result.  
**Paper scope:** one integrated paper on quantization-aware distributed steganography in LLM weights.  
**Primary pilot:** `Qwen/Qwen2.5-3B` (use the exact checkpoint revision recorded by the run).  
**Primary study:** seven named NF4 target models. GPTQ and AWQ remain separate format-specific studies.

---

## 1. Research question

Can a distributed embedding method improve recovery and the overall trade-off among capacity, model utility, detectability, distortion, and robustness when the receiver has only the final serialized quantized model artifact and the explicitly permitted decoding information?

The paper must distinguish two receiver contracts:

- **Contract A — residual/delta-assisted recovery:** the recipient has the exact clean base checkpoint, the payload-bearing delta/patch, and the protocol key. This is the established NES baseline.
- **Contract B — quantized-artifact-only recovery:** the recipient has the final serialized quantized artifact and only the key/public decoding information declared by the method. The recipient does not have the original FP16 weights, the original residual tensor, or a separate payload-bearing residual/delta sidecar. This is the main unresolved research arm.

Results from A and B must never be pooled into one success rate or described as the same capability.

## 2. Primary target models and format boundary

The seven-model primary NF4 study uses these target IDs:

1. `Qwen/Qwen2.5-3B`
2. `Qwen/Qwen2.5-7B`
3. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
4. `meta-llama/Llama-3.1-8B`
5. `google/gemma-2-9b`
6. `microsoft/Phi-3-mini-4k-instruct`
7. `mistralai/Mistral-7B-v0.3`

Use the repository's registered NF4 configuration and record the exact package versions, quantization parameters, model revision, and device/runtime. The research plan identifies the established seven-model grid as bitsandbytes NF4, group/block size 64. Do not silently change the configuration to rescue a failed cell.

A model that cannot be downloaded, loaded, or completed is `NOT_RUN`/`SKIPPED` with a reason—not a pass and not a failure of the embedding mechanism unless the embedding itself was actually tested and failed.

**GPTQ and AWQ are separate format-specific experiments.** They use their own adapters/dequantizers and verification gates. Their results must not be counted as NF4 model cells or treated as evidence of GPTQ/AWQ robustness unless the corresponding transformation experiment is actually run.

## 3. Payload accounting

**Nominal application payload:** 10,000 bits for the primary matched comparison.

Every artifact and result must report these quantities separately:

- application payload bits before encryption/framing;
- cryptographic nonce/IV, authentication tag, headers, framing, and other protocol overhead;
- total encrypted/framed bitstream length;
- total embedded carrier bits;
- number of eligible parameters and actual carrier count;
- recovered application payload bits and BER.

Do not call a 10,256-bit embedded stream a 10,000-bit application payload without accounting for the difference. The current `PayloadEncoder` adds a 32-bit length header, and the current `delta-export` CLI accepts UTF-8 text (including a text file), while its `--bits` setting is an embedding budget—not a transparent control for exactly 10,000 application-data bits. Therefore, the matched 10,000-bit pilot requires a deterministic 1,250-byte application payload and an experiment-specific harness/adapter that records each protocol stage. Do not silently change shared production code to achieve this. If the current production path cannot accept the fixed binary payload through a test adapter, record that as an interface limitation and separately report a reproducible text-payload baseline. Do not truncate, pad, or alter payload semantics after seeing the result. Predefine any required padding/framing rule.

Use the same application payload, encryption/framing rules, model revision, and evaluation split across matched arms wherever technically possible. Record any unavoidable mismatch as a protocol deviation.

## 4. Contract A — established residual/delta-assisted baseline

### Sender inputs

- Exact clean reference checkpoint and revision.
- Repository's NF4 quantization/dequantization path and configuration.
- Reference weights and corresponding dequantized NF4 weights needed to construct the residual.
- Payload and cryptographic key.
- Pinned carrier-selection/allocation and embedding implementation.

### Baseline representation

For reference weights \(W_{\mathrm{ref}}\), quantizer \(Q\), and matching dequantizer \(D\):

\[
R_{\mathrm{clean}} = W_{\mathrm{ref}} - D(Q(W_{\mathrm{ref}})).
\]

The established residual pipeline embeds the encrypted/framed bitstream into selected residual carriers and reconstructs a payload-bearing weight representation. The W8 delta distribution path represents the patch as:

\[
\Delta = W_{\mathrm{stego}} - W_{\mathrm{clean}}
       = R_{\mathrm{embed}} - R_{\mathrm{clean}},
\]

subject to the production patcher's documented arithmetic and verification checks.

This equation describes the existing residual/delta workflow; it does **not** imply that the final NF4 code tensor itself carries a recoverable payload. The current recipient implementation explicitly reloads the NF4/FP16 pair to derive clean residuals, then reconstructs residuals using the delta; this is more specific than the README shorthand “base model + delta + key.”

### Recipient inputs

- The exact compatible clean base checkpoint/revision and the matching NF4/FP16 pair used by the production residual loader.
- The integrity-checked NES delta/patch.
- The key supplied out of band.
- The pinned production extraction/decryption implementation.

### Recipient operation

1. Load the matching NF4/FP16 model pair and derive the clean residuals with the production loader.
2. Verify delta metadata/integrity and compatibility.
3. Reconstruct the embedded residuals by adding the stored delta at recorded carrier positions.
4. Extract the encrypted/framed bitstream through the production extraction pipeline.
5. Authenticate/decrypt with the declared key.
6. Compare recovered application payload with the transmitted application payload.

A failure at each stage must be logged separately: artifact integrity, reconstruction, bit extraction/BER, cryptographic authentication/decryption, and application-message equality.

### What Contract A can support

It can support claims about recovery using the base checkpoint plus delta and key under the tested configuration. It cannot support a claim that a receiver can recover the payload from the final quantized checkpoint alone.

## 5. Contract B — quantized-artifact-only candidate

### Sender inputs

- Exact source checkpoint/revision.
- The pinned quantization implementation and configuration.
- Application payload, cryptographic key, and the candidate method's declared public metadata.
- A pre-specified distributed allocation and encoding rule.

### Permitted output

One final serialized quantized model artifact, plus only those key/public protocol materials explicitly declared by the candidate method. The output may not depend on a separately distributed residual tensor, payload-bearing delta, original FP16 weights, or hidden reference copy.

### Receiver inputs

- The final serialized quantized artifact after it has been saved and reloaded.
- The declared key and public protocol metadata, if any.
- The pinned decoder and compatible format reader.

### Required receiver test

The decoder must recover from the saved-and-reloaded artifact, not from an in-memory pre-serialization tensor. The test process must not have access to the original FP16 weights, residual cache, clean reference weights, or a sidecar that contains the payload-bearing changes.

The method may alter a permitted part of the quantized representation, but the candidate must specify exactly which values/fields are carriers, how the allocation is reproduced, and how extraction works before implementation. A failed attempt to carry information through pre-quantization perturbations is a valid negative result; it must not be re-labelled as a test of direct quantized-code embedding.

### Contract B success definition

A clean pilot counts as successful only if all of the following are true:

1. The artifact serializes and reloads successfully using the specified format reader.
2. The receiver has only the permitted inputs listed above.
3. The intended application payload is recovered and authenticated correctly.
4. BER is computed against the actual transmitted embedded bitstream; application-message equality is checked separately.
5. The clean control and the embedded artifact use the same model revision and quantization configuration.
6. Artifact hash, configuration, seed, carrier count, payload accounting, and receiver-input manifest are recorded.

BER 0 on an in-memory tensor that is not recoverable from the final artifact does not satisfy Contract B.

## 6. Matched pilot arms

Run the initial pilot on Qwen2.5-3B before expanding to the full model grid.

| Arm | Artifact/receiver contract | Purpose |
|---|---|---|
| A1 | Established residual/delta-assisted NES baseline | Confirm the existing production path under the frozen protocol |
| B1 | Quantized-artifact-only candidate | Test the unresolved receiver contract |
| C1 | Clean quantized model, no payload | Quantization/reconstruction control |
| C2 | Matched sham/control artifact if required by the detector design | Control for artifacts unrelated to the payload encoding |

A1 is a baseline, not evidence for B1. If B1 is not yet implemented, first produce a design note that specifies carrier representation, writer, serialized artifact, reader, allocation reproduction, and expected failure modes. Do not code an underspecified candidate.

## 7. Frozen primary measurements

### Recovery and capacity

- BER on the actual transmitted bitstream, with `bits_compared` and `bit_errors`.
- Exact encrypted/framed stream recovery.
- Cryptographic authentication/decryption success.
- Exact application-payload equality.
- Application payload bits recovered, total embedded bits, carrier count, and eligible parameter count.

Report BER and decryption/message recovery as separate outcomes.

### Utility and distortion

- Perplexity with the repository's established three-way control protocol: clean/reference, quantization/reconstruction control, and embedded result.
- Embedding-specific PPL change relative to the correct control, not merely relative to the original full-precision model.
- Weight-space distortion and relevant distribution summaries.
- Model revision, evaluation data, tokenizer, and evaluation configuration.

The historical 2% PPL threshold may be retained as an internal screening gate, but report actual values and controls.

### Detectability

- Statistical and neural detector results separately.
- Balanced accuracy, confusion matrix, sample counts, split strategy, and uncertainty intervals where estimable.
- State whether the detector knows carrier locations or is blind to them.
- Include valid clean controls and check for constant/all-at-chance predictors.
- Report the historical 55% accuracy threshold as a project screening gate only—not as a security guarantee or proof of undetectability.

Do not infer universal undetectability from one detector at chance.

### Robustness

Run clean recovery first. Then test each named transformation as a separate condition and report BER per condition:

- Gaussian perturbation at \(\sigma=0.001\) and \(\sigma=0.002\), using the existing test protocol;
- fine-tuning and other model surgery only as explicitly specified;
- NF4 re-quantization;
- GPTQ and AWQ conversions as separate format-specific transformations.

Historical noise gates are BER \(\leq 0.02\) at \(\sigma=0.001\) and BER \(\leq 0.10\) at \(\sigma=0.002\). These are pre-existing project gates, not claims of real-world robustness. A failure remains in the results.

## 8. Existing findings that must remain visible

The paper and future result tables must preserve, at minimum, these constraints from the current research plan:

- The original sign embedding's neural detector result of 70.5% remains a FAIL against the historical 55% gate.
- The LWE-labelled grid/parity method is described as LWE-inspired unless an actual LWE/SIS construction and appropriate security analysis are implemented.
- Exp13 keyless recovery remains a failure of the claim that the existing embedding channel is key-gated. AES-GCM confidentiality/integrity is a separate property.
- Exp23's losses after NF4 re-quantization, GPTQ/AWQ conversion, and the specified merge condition remain visible.
- Exp2's residual-magnitude threshold fails on four of seven target NF4 models; do not alter the threshold to make the study pass.
- GPTQ/AWQ clean-channel results do not establish their robustness or detectability.
- Any skipped model, invalid detector control, incomplete model matrix, or protocol deviation must be disclosed.
- Preserve existing result artifacts. A forced rerun must follow the repository's archival workflow.

## 9. Seven-model expansion gate

Do not launch a full new seven-model Contract B sweep until the Qwen2.5-3B pilot has:

1. a documented and reproducible Contract B writer/reader;
2. clean recovery after serialization/reload;
3. verified receiver-input isolation (no reference weights/residual/delta);
4. complete payload accounting;
5. valid matched controls;
6. a reproducible run manifest.

If the pilot fails, first classify the failure: information loss during quantization, unsupported representation, serialization/reader mismatch, allocation drift, extraction error, or cryptographic failure. A negative result should be retained as evidence and used to narrow the paper's claim.

## 10. Reproducibility and artifact manifest

Each run must record:

- Git commit SHA and dirty-tree status;
- model ID and immutable revision;
- quantization format/configuration and package versions;
- device/runtime and deterministic settings;
- payload generation, payload hash, nominal application bits, framing/encryption overhead, and embedded bits;
- encoder/decoder versions, strategy, allocation policy, seeds, carrier count, and layer coverage;
- serialized artifact SHA-256;
- exact receiver input manifest;
- transmitted/extracted bit counts, bit errors, BER, authentication/decryption status, and message equality;
- utility, detector, distortion, and per-transformation robustness results;
- status `PASS`, `FAIL`, `NOT_RUN`, `SKIPPED`, or `ERROR`, with reasons.

Do not overwrite a completed result without preserving the previous artifact. Do not relax thresholds after observing results.

## 11. Current decision

**Frozen now:** the distinction between Contract A and Contract B; the seven-model NF4 target list; the 10,000-bit nominal application payload accounting rule; the existing residual/delta baseline; the GPTQ/AWQ boundary; the evaluation dimensions; and the requirement to preserve failures.

**Not yet frozen:** the specific carrier representation and encoding algorithm for Contract B. This must be specified in a short design note before implementation. The current evidence does not justify pretending that an artifact-only candidate already exists or works.

**Next action:** reproduce/verify the Qwen2.5-3B Contract A baseline using the current production artifacts and record the exact payload accounting. Use a fixed 1,250-byte payload for the normalized 10,000-bit application comparison; separately record the 32-bit length header and cryptographic/framing overhead. Then write the Contract B carrier/writer/reader design and test its serialization/reload round trip. No shared production code is changed by this document, and nothing is pushed to GitHub by creating it.
