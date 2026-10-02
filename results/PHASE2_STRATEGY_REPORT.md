# Phase 2 — Embedding Strategy Head-to-Head

Model: `Qwen/Qwen2.5-3B` · Gate: detector accuracy ≤ 55% · Reference recorded result: 70.5%

## Question

Sign embedding forces a carrier to `±|r|`, which randomises the carrier sign
distribution and a carrier-centered MLP detects it at 70.5%. The alpha/gamma/payload
sweep showed retuning does not help. So the only way under the gate is a scheme that
does **not** flip signs. Eight strategies existed on disk; this compares the four
that were reachable.

## Results

| Strategy | Extractable | BER @ σ=0.001 | Detector | Gate | Verdict |
| --- | --- | --- | --- | --- | --- |
| `sign` (production) | yes | 0.0000 | **72.50%** | FAIL | works, detected |
| `magnitude_aware` | yes | 0.0000 | **67.50%** | FAIL | works, detected |
| `lwe` | **no** | **0.5036** | 50.00% | — | unusable |
| `neural` | no | — | — | — | needs training |

Robustness gate is BER ≤ 0.02 at σ=0.001. Detectability gate is ≤ 55%.

## Verdict: no strategy wins. Phase 3 should not proceed on these candidates.

## Why each failed

**`sign` and `magnitude_aware` are the same scheme.** `magnitude_aware` uses a
per-carrier adaptive margin but still writes `boosted if bit == 1 else -boosted`. It
forces the identical sign flip, with a *larger* perturbation (52 changed values per
patch vs 25). It scores 5 points better, which is not enough to matter — both are far
above the gate, and the gap is measurement noise on a 55-point scale, not a mechanism
difference.

**`lwe` cannot carry data at all.** BER 0.5036 is chance, not degradation. Production
extraction is `SignExtractor`, which reads carrier *signs*; it cannot decode a parity
grid. Its own `extract()` is no better — it calls `residuals_ref[layer].std()` to derive
the grid width, so it needs the original cover residuals, which a real extractor does
not have.

**`neural` needs a model that does not exist.** It raises
`NeuralStrategy requires a trained model`.

## The one real lead

LWE's perturbation is **29× smaller** than sign's — max per-patch delta 0.15 vs 4.41,
and **zero** byte-identical pairs (vs 26 for sign). That is why its detector accuracy sits
at exactly chance.

Two honest caveats:

1. **Chance accuracy is a sensitivity limit, not a security proof.** The detector cannot
   find a signal 29× smaller at 30 epochs. A stronger or longer-trained detector might.
2. **The small perturbation is also why it fails.** Same magnitude that hides the signal
   fails to survive extraction.

So LWE is not "the answer with a small bug". It points at a real mechanism — encode onto
a grid rather than flipping signs — but the implementation cannot carry bits.

## What Phase 3 would actually require

Not a config change. All of this is new work:

1. Derive `interval_width` from the **embedded** tensor's own statistics instead of the
   cover residuals, so an extractor holding only stego weights can size the grid.
2. Write a matching `LweExtractor` and wire it into `DecryptPipeline`, which is
   currently hardcoded to `SignExtractor`.
3. Re-measure. The 29× margin suggests a genuinely different detectability regime, but
   that is a hypothesis to test, not a result.

## Comparability note

Each strategy is measured by **its own** detector, because the dataset is built from
that strategy's embedding. These accuracies answer "a detector trained against this
embedding", not a universal detector score. The recorded 500-pair / 30-epoch figure
stays the headline; this is a comparison at 400 pairs.

## Also found

`sign_bias` is statistically indistinguishable across sign (0.01479), magnitude_aware
(0.01450) and lwe (0.01486). The simple statistical gates therefore **cannot** tell these
schemes apart — only the neural detector can. That is itself a finding worth reporting:
statistical undetectability is a weak proxy for neural security.

Artifact: `results/exp10_strategy_comparison.json`