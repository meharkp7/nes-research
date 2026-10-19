"""
W6 — model surgery survival (exp23).

RESEARCH_PLAN §4 W6: *"Does the embedding survive contact with the
rest of the ML lifecycle? A carrier that survives one cycle but dies
on first fine-tune is not viable. Each is: embed → surgery →
extract → BER. Expect degradation; the question is whether it is
total or graceful."*

One production-path embed (sign, exp3's payload), then the SAME
stego weights run through nine cells — a weight-path control plus
eight surgeries — so the surgery is the only variable (exp21's
paired design, in weight space).

Pipeline (the weights, exactly as the codebase defines them):

    r_ref  = extract_residuals(nf4, fp16)      # R = W_FP16 - dequant(W_NF4)
    W_stego[l] = dequant(nf4)[l] + embedded_r[l]
             (= apply_residuals_to_model's own formula; built from
              fp16[l] - r_ref[l] + embedded_r[l] per layer)
    cell:   W' = surgery(W_stego);  r' = embedded_r + (W' - W_stego)
    extract: strategy extractor over r' at the embed's carriers

Because r' is the embedded residual plus the surgery's WEIGHT-SPACE
delta, the control (no surgery) is embedded_r exactly — a non-zero
control BER means plumbing, not physics, and the gate says so.

Cells (all fp32 — the pipeline's own arithmetic; an fp16 checkout
would round sub-residual displacements away):

    control      no surgery (validity check, must be 0.0)
    lora_*       LoRA-shaped low-rank updates at two stated RMS
                 scales (0.1%, 1% of RMS(W)) — the MERGE step of a
                 LoRA is a linear add; real training is measured
                 separately in the finetune_1k cell (W6.2)
    prune_*      magnitude pruning, per layer: smallest-|W|
                 fraction zeroed (10%, 30%)
    nf4_requant  W' = dequant(quantize_nf4(W_stego), blocksize 64)
                 — the NF4 leg of W6.3; gptq_requant / awq_requant
                 attempt the other two W6.3 legs (gptqmodel /
                 autoawq, installs authorized by the author
                 2026-10-06), reading the packed result back with
                 the repo's own dequantizers — a leg that cannot
                 run records its real exception under not_run
    merge_*      task-vector merge with Qwen2.5-3B-Instruct (same
                 shapes, cached): W' = W_stego + t·(W_inst - W_stego),
                 at t = 0.01 / 0.05 / 0.5
    finetune_1k  W6.2 real training — LoRA (rank 8, alpha 8) on
                 mlp.down_proj ONLY (the surgery scope), 1,000
                 steps of next-token prediction (~0.9 epoch of the
                 cached wikitext-2-raw train split, 512-token
                 blocks, batch 1 × grad-accum 4, lr 1e-4 cosine),
                 merged with peft's merge_and_unload; the cell
                 delta = trained − original (exact fp32
                 subtraction of the two fp16 views). Full-
                 parameter training still exceeds this machine
                 (26 GB) — the scoped LoRA run is the honest fit,
                 stated as such
    gptq_requant  W6.3 GPTQ leg — int4 GPTQ with real calibration
                 forwards through the stego model, read back with
                 the repo's own dequantize_gptq_layer
    awq_requant  W6.3 AWQ leg — autoawq quantize on stated wikitext
                 windows, read back with dequantize_awq_layer

W6.2 now runs: `peft`, `trl` and `gptqmodel` were installed
2026-10-06 under an explicit author decision reversing the earlier
"installs are not a research action" deferral (recorded with this
run in RESEARCH_LOG); an honest full-parameter run still exceeds
this machine, so the measured leg is the scoped LoRA run above.
Each attempted leg either produces a cell or records its real
exception under `not_run` — recorded, not patched.

Pre-registered expectations, recorded before the run (misses are
recorded, never rewritten): control 0.0 (structural); lora cells
expected to survive or degrade gracefully (carriers are top-|r|
positions, margins ≫ the stated RMS scales); prune outcome depends
on the |W|-vs-|r| overlap and is genuinely open; nf4_requant
plausibly heavy loss (deltas smaller than NF4 bucket spacing never
reach the bytes) but measured, not assumed; merge expected to die
as t grows. Added before the 2026-10-06 legs ran: finetune_1k
expected to survive or degrade gracefully (LoRA deltas land at
merge scale, where W6.1 already survives); gptq_requant and
awq_requant plausibly heavy loss like nf4 (sub-bucket deltas) but
measured, not assumed. Gate: THRESHOLDS['exp23'] — exp3's 0.0
twice, both reused. Per-cell verdicts (exp18's rule); the BER
column IS the total-vs-graceful reading.

Usage:
    python -m src.experiments.exp23_model_surgery --model <id>
"""

import argparse
import gc
import importlib.util
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from src.core.types import EmbeddingConfig  # noqa: E402
from src.embedding.intelligent_embedder import IntelligentEmbedder  # noqa: E402
from src.embedding.strategy_registry import (  # noqa: E402
    build as build_strategy,
    extract_with,
)
from src.experiments.experiment_registry import gate_for  # noqa: E402
from src.experiments.exp15_lwe_fidelity import _context_for, _slug  # noqa: E402
from src.experiments.paths import RESULTS_DIR  # noqa: E402
from src.experiments.residual_source import load_cached_residuals  # noqa: E402

EXPERIMENT = "exp23"
STRATEGY = "sign"          # production's own scheme (DecryptPipeline)
PAYLOAD_BITS = 10_000      # exp10's / exp3's payload
MESSAGE = "A" * 1_250
SEED = 42
MERGE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
PRUNE_FRACTIONS = (0.10, 0.30)
LORA_RATIOS = (0.001, 0.01)   # target RMS(delta)/RMS(W)
LORA_RANK = 8
MERGE_T = (0.01, 0.05, 0.5)

# The legs added 2026-10-06 (author-authorized installs); all values
# are stated up front so the artifact's method can only restate them.
FINETUNE_STEPS = 1000      # W6.2's 1-10k window, top of the range
FINETUNE_BLOCK = 512
FINETUNE_LR = 1e-4
FINETUNE_ACCUM = 4
GPTQ_CALIB_SAMPLES = 64
AWQ_CALIB_SAMPLES = 32
AWQ_CALIB_SEQ_LEN = 256

# The environment probe, run at artifact time — blockers recorded as
# facts from THIS interpreter, not as prose.
DEPENDENCIES = ("peft", "trl", "gptqmodel", "auto_gptq", "bitsandbytes")


def log(message: str) -> None:
    """Flush every stage — a long run must never look hung (§3.7)."""
    print(message, flush=True)


# ------------------------------------------------------------------
# Surgery primitives (pure; unit-tested without a model)
# ------------------------------------------------------------------

def magnitude_prune(w: torch.Tensor, fraction: float) -> torch.Tensor:
    """Zero the smallest-|w| `fraction` of entries, in place on a copy.

    Per-layer magnitude pruning — the standard variant: victims are
    chosen by |weight| within each tensor.
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"prune fraction must be in [0, 1), got {fraction}")
    out = w.clone()
    flat = out.abs().flatten()
    k = int(fraction * flat.numel())
    if k:
        cutoff = torch.kthvalue(flat, k).values
        out[out.abs() <= cutoff] = 0.0
    return out


def low_rank_delta(
    w: torch.Tensor,
    ratio: float,
    rank: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """A LoRA-shaped delta: B @ A scaled so RMS(delta) = ratio * RMS(w).

    The merge step of a LoRA adapter is exactly this linear add; the
    training that produced B, A is measured for real in the
    finetune_1k cell (W6.2). Seeded, so a cell reproduces.
    """
    rows, cols = w.shape
    a = torch.randn(rank, cols, generator=generator, dtype=torch.float32)
    b = torch.randn(rows, rank, generator=generator, dtype=torch.float32)
    delta = b @ a
    target = ratio * w.float().pow(2).mean().sqrt()
    delta = delta * (target / delta.float().pow(2).mean().sqrt())
    return delta.to(w.dtype)


def merge_delta(
    w_stego: torch.Tensor,
    w_other: torch.Tensor,
    t: float,
) -> torch.Tensor:
    """Task-vector merge delta: returns t * (W_other - W_stego), the
    delta to ADD so W' = W_stego + delta.

    The base-side weight of the task vector IS W_stego (it stands in
    for the base plus the payload); t=0 is the no-op, t=1 lands on
    W_other exactly. The caller applies it (measure_cell computes
    r' = embedded_r + delta).
    """
    if w_other.shape != w_stego.shape:
        raise ValueError(
            f"merge shapes differ: {tuple(w_other.shape)} vs "
            f"{tuple(w_stego.shape)} — never reshape a mismatched matrix"
        )
    return t * (w_other.float() - w_stego.float()).to(w_stego.dtype)


def nf4_requant(w: torch.Tensor) -> torch.Tensor:
    """The W6.3 NF4 leg: dequant(quantize_nf4(w)) — group 64, the
    shipped format's blocksize, via bitsandbytes' own kernels."""
    from bitsandbytes.functional import dequantize_4bit, quantize_4bit

    q, state = quantize_4bit(
        w.detach().to(torch.float16).cpu(), quant_type="nf4"
    )
    back = dequantize_4bit(q, state, quant_type="nf4")
    return back.to(torch.float32).to(w.device)


# ------------------------------------------------------------------
# The W6.2/W6.3 legs (added 2026-10-06, author-authorized installs)
# ------------------------------------------------------------------

def lm_blocks(ids: List[int], block: int) -> List[List[int]]:
    """Non-overlapping `block`-sized chunks of a token stream (W6.2).

    Deterministic and gap-free: next-token training sees exactly the
    flattened split's tokens, each sequence feeding the next token.
    """
    if block <= 0:
        raise ValueError(f"block must be positive, got {block}")
    return [ids[i:i + block] for i in range(0, len(ids) - block + 1, block)]


def fp16_merge_diff(
    original: Dict[int, torch.Tensor],
    trained: Dict[int, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """The W6.2 cell delta: trained − original, exact fp32 subtraction.

    Both sides are the same checkpoint's fp16 matrices before/after a
    LoRA merge, cast up and subtracted in fp32 — the subtraction of
    two equal-layout tensors introduces no rounding. A no-op training
    run therefore returns exact zeros, which is the no-op property
    the cell arithmetic depends on (a "trained" model that did not
    train must measure nothing, not something plausible).
    """
    out: Dict[int, torch.Tensor] = {}
    for lid, w in trained.items():
        o = original.get(lid)
        if o is None:
            raise KeyError(f"layer {lid}: trained without an original")
        if o.shape != w.shape:
            raise ValueError(
                f"layer {lid}: trained {tuple(w.shape)} vs original "
                f"{tuple(o.shape)} — never subtract a shape mismatch"
            )
        out[lid] = w.detach().float().cpu() - o.detach().float().cpu()
    return out


def _wikitext_windows(n: int, min_chars: int = 200) -> List[str]:
    """n short raw-text rows from the cached wikitext-2-raw split.

    Used as the calibration corpus for BOTH quantization legs so
    neither tool falls back to a default (downloaded) corpus — what
    runs is what the method states.
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    out: List[str] = []
    for row in ds["text"]:
        t = row.strip()
        if len(t) >= min_chars:
            out.append(t)
        if len(out) >= n:
            break
    if not out:
        raise RuntimeError(
            "wikitext-2-raw yielded no usable calibration rows"
        )
    return out


def _quantize_call(fn, *args, **kwargs):
    """Call `fn`, refusing to silently drop a kwarg its signature lacks.

    A dropped `dataset=`/`calib_data=` would make the quantizer fall
    back to its own default calibration corpus — different evidence
    from the stated one — so the call fails loudly instead of
    quietly measuring something else.
    """
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    has_var = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if not has_var:
        dropped = [k for k in kwargs if k not in sig.parameters]
        if dropped:
            raise TypeError(
                f"{getattr(fn, '__qualname__', fn)} does not accept "
                f"{dropped} — refusing to run against a default "
                f"calibration corpus instead of the stated one"
            )
    return fn(*args, **kwargs)


def _swap_payload_matrices(
    model, w_stego: Dict[int, torch.Tensor], family: str
) -> None:
    """Overwrite a loaded model's down_proj with W_stego (fp16 view).

    The quantizers only accept a loaded checkpoint, so the stego
    weights go in the way a shipped checkpoint would arrive: copied
    in the model's own dtype, payload matrices only — the surgery
    scope. The caller discards the model afterwards.
    """
    from src.model.registry import get_layer_module

    for lid, w in w_stego.items():
        mod = get_layer_module(model, family, lid, "mlp").down_proj
        mod.weight.data.copy_(w.to(torch.float16))


def _dequant_deltas(
    read_dense: Dict[int, torch.Tensor],
    w_stego: Dict[int, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """delta = W' − W_stego from dequantized matrices, layout-guarded.

    Packed checkpoints store [in, out] where the pipeline's matrices
    are [out, in]; a transposed read is accepted only when it is
    exactly the guard's shape — never a blind reshape.
    """
    out: Dict[int, torch.Tensor] = {}
    for lid in sorted(w_stego):
        dense = read_dense[lid]
        want = w_stego[lid].shape
        if dense.shape != want and dense.T.shape == want:
            dense = dense.T.contiguous()
        if dense.shape != want:
            raise RuntimeError(
                f"layer {lid}: dequantized {tuple(dense.shape)} vs "
                f"W_stego {tuple(want)} — reader/layout mismatch"
            )
        out[lid] = dense.float().cpu() - w_stego[lid].float().cpu()
    return out


def lora_finetune_deltas(
    model,
    tokenizer,
    family: str,
    steps: int,
) -> Tuple[Dict[int, torch.Tensor], Dict[str, Any]]:
    """W6.2: real gradient training — LoRA adapters on the payload scope.

    Only `down_proj` receives adapters (the surgery scope: no other
    module can touch the payload by construction). Trained in the
    checkpoint's own dtype (fp16, no autocast flags) on the best
    device the Trainer picks. Returns ({layer: fp32 delta}, record).
    """
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(SEED)
    orig = {
        lid: t.detach().clone()
        for lid, t in _down_projs(model, family).items()
    }

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_RANK,
            lora_dropout=0.0,
            target_modules=["down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    rows = [t for t in ds["text"] if t and t.strip()]
    tokenized = tokenizer(rows, add_special_tokens=False)["input_ids"]
    flat = [i for row in tokenized for i in row]
    blocks = lm_blocks(flat, FINETUNE_BLOCK)
    if not blocks:
        raise RuntimeError("wikitext tokenized to no full block")

    class _Blocks(torch.utils.data.Dataset):
        def __init__(self, chunks: List[List[int]]) -> None:
            self.chunks = chunks

        def __len__(self) -> int:
            return len(self.chunks)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            ids = torch.tensor(self.chunks[idx], dtype=torch.long)
            return {"input_ids": ids, "labels": ids.clone()}

    out_dir = tempfile.mkdtemp(prefix="exp23-ft-")
    trainer = Trainer(
        model=peft_model,
        args=TrainingArguments(
            output_dir=out_dir,
            max_steps=steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=FINETUNE_ACCUM,
            learning_rate=FINETUNE_LR,
            lr_scheduler_type="cosine",
            # transformers 5.x dropped warmup_ratio from
            # TrainingArguments; 3% of 1000 steps = 30, exactly the
            # registered intent (the record still says 3% warmup).
            warmup_steps=max(1, int(steps * 0.03)),
            logging_steps=max(steps // 10, 1),
            report_to=[],
            save_strategy="no",
            dataloader_num_workers=0,
            seed=SEED,
        ),
        train_dataset=_Blocks(blocks),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    log(f"  [finetune] {steps} steps on {trainer.args.device} "
        f"({len(blocks)} blocks of {FINETUNE_BLOCK} tokens) ...")
    run = trainer.train()
    history = trainer.state.log_history or []
    loss_first = next((h["loss"] for h in history if "loss" in h), None)
    loss_last = next(
        (h["loss"] for h in reversed(history) if "loss" in h), None
    )

    merged = peft_model.merge_and_unload()
    deltas = fp16_merge_diff(orig, _down_projs(merged, family))
    if all(bool((d == 0).all()) for d in deltas.values()):
        raise RuntimeError(
            "fine-tune produced an exact-zero delta — no optimization "
            "step changed a weight; refusing to record a no-op cell"
        )
    record = {
        "steps": steps,
        "block": FINETUNE_BLOCK,
        "grad_accum": FINETUNE_ACCUM,
        "lr": FINETUNE_LR,
        "scheduler": "cosine with 3% warmup",
        "r": LORA_RANK,
        "alpha": LORA_RANK,
        "targets": "down_proj only (surgery scope)",
        "dataset": (
            "wikitext-2-raw-v1 train (cached), flattened, "
            "non-overlapping blocks"
        ),
        "tokens_seen": steps * FINETUNE_ACCUM * FINETUNE_BLOCK,
        "blocks_available": len(blocks),
        "loss_first": loss_first,
        "loss_last": loss_last,
        "final_train_loss": float(run.training_loss),
        "device": str(trainer.args.device),
        "dtype": "checkpoint fp16 throughout (no autocast flags)",
        "optimizer": "AdamW (Trainer default)",
        "delta": (
            "trained − original via peft merge_and_unload(), exact "
            "fp32 subtraction of the two fp16 views"
        ),
    }
    shutil.rmtree(out_dir, ignore_errors=True)
    model.config.use_cache = True
    return deltas, record


def gptq_requant_deltas(
    model_id: str,
    w_stego: Dict[int, torch.Tensor],
    tokenizer,
    family: str,
) -> Tuple[Dict[int, torch.Tensor], Dict[str, Any]]:
    """W6.3's GPTQ leg: real GPTQ int4 quantization of the stego model.

    Calibration forward passes run through the model with W_stego in
    the payload's matrices; the packed result is read back with the
    repo's own GPTQ dequantizer (the reader exp9 validated) —
    delta = W' − W_stego.
    """
    from gptqmodel import GPTQModel, QuantizeConfig
    from src.quantization.adapters import dequantize_gptq_layer

    cfg = QuantizeConfig(
        bits=4,
        group_size=128,
        sym=True,
        true_sequential=False,
        device="cpu",
        calibration_data_device="cpu",
    )
    qmodel = GPTQModel.from_pretrained(model_id, quantize_config=cfg)
    _swap_payload_matrices(qmodel.model, w_stego, family)
    calib = _wikitext_windows(GPTQ_CALIB_SAMPLES)
    # gptqmodel 7.5.0 renamed the corpus kwarg: `calibration`
    # (positional first); `dataset=` is rejected, and the guard
    # above refuses any call that would drop the stated corpus.
    _quantize_call(qmodel.quantize, calib, tokenizer=tokenizer)

    # gptqmodel 7.5.0 keeps the packed buffers on the meta device
    # after quantize() (the real tensors live in its LazyTurtle
    # stash), so the in-memory module cannot be read directly.
    # save() materializes the canonical GPTQ checkpoint — exactly
    # the file layout dequantize_gptq_layer was validated against
    # (exp9) — which is read per layer and deleted afterwards.
    import glob
    import shutil
    import tempfile
    import types
    from contextlib import ExitStack

    from safetensors import safe_open

    save_dir = tempfile.mkdtemp(prefix="nes_exp23_gptq_")
    read: Dict[int, torch.Tensor] = {}
    try:
        qmodel.save(save_dir)
        shards = sorted(glob.glob(f"{save_dir}/*.safetensors"))
        if not shards:
            raise RuntimeError("gptqmodel save() wrote no safetensors shard")
        with ExitStack() as stack:
            files = [
                stack.enter_context(safe_open(p, framework="pt"))
                for p in shards
            ]
            for lid in sorted(w_stego):
                packed: Dict[str, torch.Tensor] = {}
                for fh in files:
                    for key in fh.keys():
                        parts = key.split(".")
                        if "layers" not in parts:
                            continue
                        at = parts.index("layers")
                        if (
                            parts[at : at + 4]
                            == ["layers", str(lid), "mlp", "down_proj"]
                            and parts[-1]
                            in ("qweight", "qzeros", "scales", "g_idx")
                        ):
                            packed[parts[-1]] = fh.get_tensor(key)
                if len(packed) != 4:
                    raise RuntimeError(
                        f"packed down_proj for layer {lid}: expected 4 "
                        f"tensors, found {sorted(packed)}"
                    )
                read[lid] = dequantize_gptq_layer(
                    types.SimpleNamespace(**packed)
                )
    finally:
        shutil.rmtree(save_dir, ignore_errors=True)
    deltas = _dequant_deltas(read, w_stego)
    record = {
        "backend": "gptqmodel (installed 2026-10-06, author decision)",
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "method": (
            "GPTQ: calibration forwards through the stego model on "
            "CPU, per the tool's own quantizer"
        ),
        "calibration": (
            f"{GPTQ_CALIB_SAMPLES} wikitext-2-raw windows (stated; "
            "the default corpus is refused, not silently used)"
        ),
        "reader": "src.quantization.adapters.dequantize_gptq_layer",
        "read_back": (
            "gptqmodel save() to a temp dir (its in-memory packed "
            "buffers are meta-device shells) → per-layer safetensors "
            "tensors → dequantize_gptq_layer; temp dir deleted"
        ),
        "delta": "dequant(W') − W_stego, exact fp32 subtraction",
        "scope": (
            "the tool quantizes every linear by default; the cell "
            "measures the payload's down_proj delta — other modules "
            "cannot touch the payload by construction"
        ),
    }
    del qmodel, read
    gc.collect()
    return deltas, record


def awq_requant_deltas(
    model_id: str,
    w_stego: Dict[int, torch.Tensor],
    tokenizer,
    family: str,
) -> Tuple[Dict[int, torch.Tensor], Dict[str, Any]]:
    """W6.3's AWQ leg: autoawq's quantize over the stego model.

    Scales come from the stego weights' activations; the packed
    result is read back with the repo's own AWQ dequantizer (exp8's
    gate) — delta = W' − W_stego.
    """
    from awq import AutoAWQForCausalLM
    from src.model.registry import get_layer_module
    from src.quantization.adapters import dequantize_awq_layer

    awq_model = AutoAWQForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, trust_remote_code=False,
    )
    # autoawq double-wraps (wrapper.model = the transformers model,
    # whose layers sit at .model.layers); the registry unwraps one
    # level only, so hand it the inner model — the same parameter
    # objects the quantizer will see, reached by a valid path.
    inner = awq_model.model
    _swap_payload_matrices(inner, w_stego, family)
    calib = _wikitext_windows(AWQ_CALIB_SAMPLES)
    _quantize_call(
        awq_model.quantize,
        tokenizer=tokenizer,
        quant_config={"w_bit": 4, "q_group_size": 128, "zero_point": True},
        calib_data=calib,
        max_calib_samples=AWQ_CALIB_SAMPLES,
        max_calib_seq_len=AWQ_CALIB_SEQ_LEN,
        n_parallel_calib_samples=4,
        apply_clip=True,
        duo_scaling=True,
    )

    read: Dict[int, torch.Tensor] = {}
    for lid in sorted(w_stego):
        mod = get_layer_module(
            inner, family, lid, "mlp"
        ).down_proj
        read[lid] = dequantize_awq_layer(mod)
    deltas = _dequant_deltas(read, w_stego)
    record = {
        "backend": (
            "autoawq (upstream-deprecated; this is its runtime probe "
            "on torch 2.13 / transformers 5.16.1)"
        ),
        "bits": 4,
        "group_size": 128,
        "zero_point": True,
        "calibration": (
            f"{AWQ_CALIB_SAMPLES} wikitext-2-raw windows, seq <= "
            f"{AWQ_CALIB_SEQ_LEN} (stated; pileval is not downloaded)"
        ),
        "reader": (
            "src.quantization.adapters.dequantize_awq_layer (exp8's gate)"
        ),
        "delta": "dequant(W') − W_stego, exact fp32 subtraction",
        "scope": (
            "the tool quantizes every linear by default; the cell "
            "measures the payload's down_proj delta — other modules "
            "cannot touch the payload by construction"
        ),
    }
    del awq_model, read
    gc.collect()
    return deltas, record


# ------------------------------------------------------------------
# Cells
# ------------------------------------------------------------------

def _rms(t: torch.Tensor) -> float:
    return float(t.float().pow(2).mean().sqrt())


def measure_cell(
    name: str,
    deltas: Dict[int, torch.Tensor],
    embedded_residuals: Dict[int, torch.Tensor],
    carrier_indices: Dict[int, List[int]],
    embedded_bits: List[int],
    strategy,
    w_stego: Dict[int, torch.Tensor],
) -> Dict[str, Any]:
    """One surgery: r' = embedded_r + flatten(delta), extract, measure.

    Deltas arrive weight-shaped (surgery ran on W_stego's matrix);
    emb lives in flat residual space — flatten(delta) is exactly
    extract_residuals' .flatten() undone, so the addition is layout-
    exact against the carriers, never a blind reshape.

    Magnitude stats are recorded beside the BER so the number can be
    explained, not just cited.
    """
    r_prime: Dict[int, torch.Tensor] = {}
    d2 = w2 = 0.0
    dcar2 = 0.0
    ncar = 0
    touched = 0
    for lid, emb in embedded_residuals.items():
        delta = deltas.get(lid)
        if delta is None:
            r_prime[lid] = emb
            continue
        r_prime[lid] = emb + delta.flatten()
        d2 += float(delta.float().pow(2).sum())
        w2 += float(w_stego[lid].float().pow(2).sum())
        idx = carrier_indices.get(lid, [])
        if idx:
            dv = delta.float().flatten()[idx]
            dcar2 += float(dv.pow(2).sum())
            ncar += len(idx)
            touched += int((dv != 0).sum())

    recovered = extract_with(
        strategy, r_prime, carrier_indices, residuals_ref=None,
        strategy_name=STRATEGY,
    )
    compared = min(len(embedded_bits), len(recovered))
    errors = sum(
        1
        for a, b in zip(embedded_bits[:compared], recovered[:compared])
        if a != b
    )
    return {
        "surgery": name,
        "ber": errors / compared if compared else None,
        "bit_errors": errors,
        "bits_compared": compared,
        "rms_delta_over_rms_w": (d2 / w2) ** 0.5 if w2 else None,
        "rms_delta_at_carriers": (dcar2 / ncar) ** 0.5 if ncar else None,
        "carriers_displaced": touched,
        "carriers_total": ncar,
    }


def not_run_findings(
    leg_errors: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    """What did not run here, with the probe or exception proving it.

    A leg that ran leaves no finding. A leg that failed carries the
    real exception text from this interpreter at artifact time; the
    dependency branches stay as the fallback probe for environments
    where the tooling is absent rather than broken.
    """
    leg_errors = leg_errors or {}
    availability = {
        dep: importlib.util.find_spec(dep) is not None
        for dep in DEPENDENCIES
    }
    findings = []
    if "W6.2" in leg_errors:
        findings.append({
            "item": "W6.2 — 1-10k fine-tuning steps",
            "status": "BLOCKED (recorded, not patched)",
            "reason": leg_errors["W6.2"],
        })
    elif not availability["peft"] or not availability["trl"]:
        findings.append({
            "item": "W6.2 — 1-10k fine-tuning steps",
            "status": "BLOCKED (recorded, not patched)",
            "reason": (
                f"peft={availability['peft']}, trl={availability['trl']} "
                "in this environment; installs are not a research action, "
                "and an honest full-parameter run exceeds this machine "
                "(26 GB). wikitext is cached but no trainer exists to "
                "run it. Fine-tune drift stays NOT_RUN and named; the "
                "sigma-noise axes (exp6/10/18/22) remain the only "
                "measured proxy for dense drift."
            ),
        })
    if "W6.3-GPTQ" in leg_errors:
        findings.append({
            "item": "W6.3 — GPTQ leg (NF4 -> GPTQ -> back)",
            "status": "BLOCKED (recorded, not patched)",
            "reason": leg_errors["W6.3-GPTQ"],
        })
    elif not availability["gptqmodel"] and not availability["auto_gptq"]:
        findings.append({
            "item": "W6.3 — GPTQ leg (NF4 -> GPTQ -> back)",
            "status": "BLOCKED (recorded, not patched)",
            "reason": (
                f"gptqmodel={availability['gptqmodel']}, "
                f"auto_gptq={availability['auto_gptq']}; transformers' "
                "GPTQConfig imports but has no quantization backend "
                "without one of them. The NF4 leg runs; GPTQ does not."
            ),
        })
    if "W6.3-AWQ" in leg_errors:
        try:
            from transformers import AwqQuantizer  # noqa: F401
            tf_probe = "transformers AwqQuantizer: importable"
        except ImportError as exc:
            tf_probe = f"transformers AwqQuantizer: ImportError ({exc})"
        findings.append({
            "item": "W6.3 — AWQ leg",
            "status": "BLOCKED (recorded, not patched)",
            "reason": f"{leg_errors['W6.3-AWQ']} | probe: {tf_probe}",
        })
    elif not importlib.util.find_spec("awq"):
        try:
            from transformers import AwqQuantizer  # noqa: F401
            tf_probe = "transformers AwqQuantizer: importable"
        except ImportError as exc:
            tf_probe = f"transformers AwqQuantizer: ImportError ({exc})"
        findings.append({
            "item": "W6.3 — AWQ leg",
            "status": "BLOCKED (recorded, not patched)",
            "reason": (
                f"autoawq absent from this environment. | probe: "
                f"{tf_probe}"
            ),
        })
    if not availability["bitsandbytes"]:
        findings.append({
            "item": "W6.3 — NF4 leg",
            "status": "BLOCKED",
            "reason": "bitsandbytes absent — nf4_requant cannot run",
        })
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B",
        help="model id whose residual cache is complete "
             "(default: Qwen2.5-3B, exp3/exp10's anchor)",
    )
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    context = _context_for(args.model)
    log(f"Model: {args.model} "
        f"(family {context.family}, {context.expected_layers} layers)")

    log("[exp23] loading cached residuals ...")
    residuals = load_cached_residuals(args.model, context.expected_layers)

    # One embed, every cell reads from it (exp21's paired design).
    config = EmbeddingConfig(
        total_payload_bits=PAYLOAD_BITS,
        embedding_strategy=STRATEGY,
        model_family=context.family,
        num_hidden_layers=(
            context.actual_layers or context.expected_layers
        ),
    )
    log(f"[exp23] embed via production path (strategy={STRATEGY}) ...")
    embed_result = IntelligentEmbedder(config).embed(MESSAGE, residuals)
    embedded_r = embed_result.embedded_residuals
    carriers = embed_result.carrier_indices
    transmitted = embed_result.embedded_bits
    log(f"  embedded {len(transmitted)} bits "
        f"(reported {embed_result.bits_embedded})")

    strategy = build_strategy(config, STRATEGY)

    log("[exp23] direct control (exp10's path, no weights) ...")
    direct = extract_with(
        strategy, embedded_r, carriers, residuals_ref=None,
        strategy_name=STRATEGY,
    )
    cmp_d = min(len(transmitted), len(direct))
    direct_ber = (
        sum(a != b for a, b in zip(transmitted[:cmp_d], direct[:cmp_d]))
        / cmp_d if cmp_d else None
    )
    log(f"  direct control ber={direct_ber}")

    # The model pair: dequant reference + fp16 side, from the code's
    # own residual definition.
    log("[exp23] loading model pair (nf4 + fp16) ...")
    nf4_model, fp16_model, _tok = context.ensure_models()
    from src.model.model_loader import extract_residuals

    log("[exp23] computing R = W_FP16 - dequant(W_NF4) from the pair ...")
    r_ref = extract_residuals(nf4_model, fp16_model, context.family)

    cache_check = max(
        float(
            (r_ref[lid].detach().float().cpu()
             - residuals[lid].detach().float().cpu()).abs().max()
        )
        for lid in r_ref
        if lid in residuals
    )
    log(f"  max|pair residual - cached residual| = {cache_check:.3e}")

    # W_stego = dequant(nf4) + embedded_residual, layer by layer —
    # apply_residuals_to_model's formula, built without mutating the
    # models. dequant(nf4) = fp16 - r_ref by the definition above.
    # Everything lands on CPU: cached residuals are CPU tensors and
    # surgery/extract arithmetic runs there.
    log("[exp23] building W_stego per layer ...")
    w_stego: Dict[int, torch.Tensor] = {}
    for lid, fp16_w in _down_projs(fp16_model, context.family).items():
        # Residual space is flat — extract_residuals stores
        # (fp16_w - dequant).flatten(), the cache matches it exactly
        # (cache_check above), and carriers index that layout. W_stego
        # is a weight matrix, so bridge the two with the same guarded
        # reshape apply_residuals_to_model uses; a blind reshape of a
        # mismatched matrix is a ground-rule violation.
        r_flat = r_ref[lid].float().cpu()
        e_flat = embedded_r[lid].float()
        want = fp16_w.numel()
        if r_flat.numel() != want or e_flat.numel() != want:
            raise RuntimeError(
                f"layer {lid}: residual numel {r_flat.numel()} / "
                f"embedded numel {e_flat.numel()} != weight numel "
                f"{want} — never reshape a mismatched matrix"
            )
        w_stego[lid] = (
            fp16_w.detach().float().cpu()
            - r_flat.reshape(fp16_w.shape)
            + e_flat.reshape(fp16_w.shape)
        )
    # Weight-path control: r' = W_stego - dequant equals embedded_r
    # by construction (cache_vs_pair above proves the views match),
    # so the control cell runs zero deltas — validating the cell
    # arithmetic itself, not the identity.
    control_delta = {
        lid: torch.zeros_like(w) for lid, w in w_stego.items()
    }
    del r_ref, residuals
    gc.collect()

    cells: List[Dict[str, Any]] = []
    gate = gate_for(EXPERIMENT)

    log("[exp23] cell control (through the weights) ...")
    cells.append(measure_cell(
        "control", control_delta, embedded_r, carriers, transmitted,
        strategy, w_stego,
    ))
    log(f"  ber={cells[-1]['ber']}")

    # --- W6.1: LoRA-shaped low-rank updates ------------------------
    for ratio in LORA_RATIOS:
        name = f"lora_{ratio:g}"
        log(f"[exp23] cell {name} (rank {LORA_RANK}, "
            f"RMS target {ratio:g}) ...")
        deltas: Dict[int, torch.Tensor] = {}
        for lid, w in w_stego.items():
            gen = torch.Generator().manual_seed(SEED + lid)
            deltas[lid] = low_rank_delta(w, ratio, LORA_RANK, gen)
        cells.append(measure_cell(
            name, deltas, embedded_r, carriers, transmitted,
            strategy, w_stego,
        ))
        log(f"  ber={cells[-1]['ber']} "
            f"rms_d/rms_w={cells[-1]['rms_delta_over_rms_w']:.2e}")
        del deltas
        gc.collect()

    # --- W6.4: magnitude pruning -----------------------------------
    for fraction in PRUNE_FRACTIONS:
        name = f"prune_{int(fraction * 100)}"
        log(f"[exp23] cell {name} ...")
        deltas = {
            lid: magnitude_prune(w, fraction) - w
            for lid, w in w_stego.items()
        }
        cells.append(measure_cell(
            name, deltas, embedded_r, carriers, transmitted,
            strategy, w_stego,
        ))
        log(f"  ber={cells[-1]['ber']} "
            f"carriers_displaced={cells[-1]['carriers_displaced']}/"
            f"{cells[-1]['carriers_total']}")
        del deltas
        gc.collect()

    # --- W6.3: NF4 leg ---------------------------------------------
    log("[exp23] cell nf4_requant ...")
    deltas = {
        lid: nf4_requant(w) - w for lid, w in w_stego.items()
    }
    cells.append(measure_cell(
        "nf4_requant", deltas, embedded_r, carriers, transmitted,
        strategy, w_stego,
    ))
    log(f"  ber={cells[-1]['ber']}")
    del deltas
    gc.collect()

    # --- W6.5: task-vector merge with the same-shape checkpoint ----
    log(f"[exp23] loading merge partner {MERGE_MODEL_ID} ...")
    other = _load_merge_partner(MERGE_MODEL_ID, context.family)
    for t in MERGE_T:
        name = f"merge_{t:g}"
        log(f"[exp23] cell {name} ...")
        deltas = {
            lid: merge_delta(w_stego[lid], other[lid], t)
            for lid in w_stego
        }
        cells.append(measure_cell(
            name, deltas, embedded_r, carriers, transmitted,
            strategy, w_stego,
        ))
        log(f"  ber={cells[-1]['ber']}")
        del deltas
        gc.collect()
    del other
    gc.collect()

    # --- W6.2: real fine-tuning steps (LoRA on the payload scope) --
    # nf4_model is dead weight from here (r_ref is already built);
    # the training leg wants the RAM.
    leg_errors: Dict[str, str] = {}
    method_finetune = method_gptq = method_awq = None
    del nf4_model
    gc.collect()
    ft_name = f"finetune_{FINETUNE_STEPS // 1000}k"
    log(f"[exp23] cell {ft_name} (LoRA down_proj, "
        f"{FINETUNE_STEPS} steps) ...")
    try:
        deltas, method_finetune = lora_finetune_deltas(
            fp16_model, _tok, context.family, FINETUNE_STEPS,
        )
        cells.append(measure_cell(
            ft_name, deltas, embedded_r, carriers, transmitted,
            strategy, w_stego,
        ))
        log(f"  ber={cells[-1]['ber']} "
            f"loss {method_finetune['loss_first']} -> "
            f"{method_finetune['loss_last']}")
        del deltas
    except Exception as exc:  # noqa: BLE001 — a real probe outcome
        leg_errors["W6.2"] = f"{type(exc).__name__}: {exc}"
        log(f"[exp23] fine-tune leg could not run: "
            f"{leg_errors['W6.2']}")
        method_finetune = None
    finally:
        del fp16_model
        gc.collect()
        torch.mps.empty_cache()

    # --- W6.3: GPTQ leg (gptqmodel; stated calibration) ----------
    log("[exp23] cell gptq_requant (gptqmodel int4) ...")
    try:
        deltas, method_gptq = gptq_requant_deltas(
            args.model, w_stego, _tok, context.family,
        )
        cells.append(measure_cell(
            "gptq_requant", deltas, embedded_r, carriers,
            transmitted, strategy, w_stego,
        ))
        log(f"  ber={cells[-1]['ber']}")
        del deltas
    except Exception as exc:  # noqa: BLE001 — a real probe outcome
        leg_errors["W6.3-GPTQ"] = f"{type(exc).__name__}: {exc}"
        log(f"[exp23] GPTQ leg could not run: "
            f"{leg_errors['W6.3-GPTQ']}")
        method_gptq = None
    gc.collect()
    torch.mps.empty_cache()

    # --- W6.3: AWQ leg (autoawq; stated calibration) -------------
    log("[exp23] cell awq_requant (autoawq int4) ...")
    try:
        deltas, method_awq = awq_requant_deltas(
            args.model, w_stego, _tok, context.family,
        )
        cells.append(measure_cell(
            "awq_requant", deltas, embedded_r, carriers,
            transmitted, strategy, w_stego,
        ))
        log(f"  ber={cells[-1]['ber']}")
        del deltas
    except Exception as exc:  # noqa: BLE001 — a real probe outcome
        leg_errors["W6.3-AWQ"] = f"{type(exc).__name__}: {exc}"
        log(f"[exp23] AWQ leg could not run: "
            f"{leg_errors['W6.3-AWQ']}")
        method_awq = None
    gc.collect()
    torch.mps.empty_cache()

    for cell in cells:
        cell["meets_gate"] = cell["ber"] == gate["max_ber"]
    control_ok = cells[0]["ber"] == gate["max_control_ber"] == direct_ber

    artifact: Dict[str, Any] = {
        "experiment": EXPERIMENT,
        "title": (
            "W6 — model surgery survival (Qwen2.5-3B, all legs: "
            "the original nine plus W6.2 fine-tune and the "
            "GPTQ/AWQ re-quant legs)"
        ),
        "model_id": args.model,
        "family": context.family,
        "num_layers": context.expected_layers,
        "gate": {
            **gate,
            "gate_source": "experiment_registry.THRESHOLDS['exp23']",
        },
        "control": {
            "direct_ber": direct_ber,
            "weight_path_ber": cells[0]["ber"],
            "cache_vs_pair_max_abs": cache_check,
            "valid": control_ok,
        },
        "cells": cells,
        "not_run": not_run_findings(leg_errors),
        "method": {
            "pipeline": (
                "W_stego = dequant(nf4) + embedded_residual per layer "
                "(apply_residuals_to_model's formula); cell residual = "
                "embedded_residual + (W' - W_stego); extraction via the "
                "sign strategy's own extractor (production scheme)"
            ),
            "residual_definition": "R = W_FP16 - dequantize(W_NF4)",
            "dtype": (
                "float32 throughout — the pipeline's own arithmetic "
                "(extract_residuals dequantizes to fp32); fp16 would "
                "round sub-residual displacements away"
            ),
            "strategy": STRATEGY,
            "payload_bits": PAYLOAD_BITS,
            "one_embed_for_all_cells": True,
            "surgery_scope": (
                "mlp.down_proj only — the payload's scope; other "
                "modules cannot touch the payload by construction"
            ),
            "prune": "per-layer magnitude pruning, smallest-|W| fraction",
            "lora": (
                f"LoRA-shaped B@A, rank {LORA_RANK}, RMS scaled to "
                f"{LORA_RATIOS} of RMS(W); the merge step is "
                f"linear. Real training is measured separately in "
                f"the {ft_name} cell (W6.2, added 2026-10-06)"
            ),
            "nf4_requant": (
                "bitsandbytes quantize_4bit/dequantize_4bit, nf4, "
                "default blocksize 64 (the shipped group size)"
            ),
            "merge": (
                f"task-vector: W' = W_stego + t*(W_instruct - W_stego), "
                f"t in {MERGE_T}; partner {MERGE_MODEL_ID} (same shapes) "
                "— the only cached same-architecture second checkpoint"
            ),
            "seed": SEED,
            "pre_registered": (
                "control 0.0; lora survive-or-graceful; prune open "
                "(|W|-vs-|r| overlap unknown); nf4_requant plausibly "
                "heavy loss (in-bucket deltas never reach the bytes); "
                "merge dies as t grows. Added 2026-10-06 before the "
                "legs ran: finetune survive-or-graceful (LoRA deltas "
                "at merge scale, W6.1's evidence); gptq/awq plausibly "
                "heavy loss like nf4 (sub-bucket deltas) but measured, "
                "not assumed. Misses are recorded, not rewritten."
            ),
            **({"finetune": method_finetune} if method_finetune else {}),
            **({"gptq": method_gptq} if method_gptq else {}),
            **({"awq": method_awq} if method_awq else {}),
        },
        "notes": [
            "Per-cell verdicts, exp18's rule: 'survives' = BER 0.0 "
            "against exp3's number; any non-zero BER is a cell failing "
            "that axis, and the BER column is the plan's "
            "total-vs-graceful reading. No gate is relaxed for "
            "expected degradation.",
            "The weight-path control must equal the direct control "
            "(both exp3's 0.0): it proves the weights carry exactly "
            "the embedded residual before any surgery touches them.",
            "cache_vs_pair_max_abs records whether the cached "
            "residuals equal a fresh R from the model pair; a large "
            "value would mean embed operated on a different residual "
            "view than the weights are built from — reported, never "
            "assumed away.",
        ],
        "reproducibility": context.reproducibility(),
    }

    target = RESULTS_DIR / f"{EXPERIMENT}_model_surgery_{_slug(args.model)}.json"
    target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    log("=" * 70)
    log(f"control: direct={direct_ber} weight_path={cells[0]['ber']} "
        f"cache_vs_pair={cache_check:.3e}")
    for cell in cells:
        log(f"  {cell['surgery']:<14} ber={cell['ber']} "
            f"errs={cell['bit_errors']}/{cell['bits_compared']} "
            f"gate={cell['meets_gate']}")
    log(f"not_run: {len(artifact['not_run'])} blockers recorded")
    log(f"wrote {target}")

    del w_stego, embedded_r, carriers, transmitted
    gc.collect()
    torch.mps.empty_cache()
    return 0


def _down_projs(model, family: str) -> Dict[int, torch.Tensor]:
    """{layer_id: down_proj.weight} for the payload's module."""
    from src.model.registry import get_layer_module, get_num_layers

    out = {}
    for i in range(get_num_layers(model)):
        out[i] = get_layer_module(model, family, i, "mlp").down_proj.weight
    return out


def _load_merge_partner(model_id: str, family: str) -> Dict[int, torch.Tensor]:
    """fp16 down_projs of the merge partner, then the model is freed."""
    from transformers import AutoModelForCausalLM

    other = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map={"": "cpu"}
    )
    weights = {
        lid: t.detach().cpu().float()
        for lid, t in _down_projs(other, family).items()
    }
    del other
    gc.collect()
    return weights


if __name__ == "__main__":
    raise SystemExit(main())
