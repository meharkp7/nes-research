"""
Full experiment suite — runs all experiments needed for the paper.

Updated to use correct residual extraction:
    R = W_fp16 - dequantize(quantize(W_fp16))
"""

import json, os
from typing import Dict, Optional
import torch


class FullExperimentSuite:
    """
    Runs all paper experiments and saves results to JSON.

    Usage:
        suite = FullExperimentSuite(model, tokenizer, residuals,
                                    nf4_dequant, module_refs, module_names)
        suite.run_all(output_dir="results/")
    """

    def __init__(
        self,
        model,
        tokenizer,
        residuals:    Dict[int, torch.Tensor],
        nf4_dequant:  Dict[int, torch.Tensor],
        module_refs:  Dict[int, object],
        module_names: Dict[int, str],
        output_dir:   str  = "results/",
        device:       str  = "cpu",
        n_ppl:        int  = 200,
        n_task:       int  = 100,
    ):
        self.model        = model
        self.tokenizer    = tokenizer
        self.residuals    = residuals
        self.nf4_dequant  = nf4_dequant
        self.module_refs  = module_refs
        self.module_names = module_names
        self.output_dir   = output_dir
        self.device       = device
        self.n_ppl        = n_ppl
        self.n_task       = n_task
        os.makedirs(output_dir, exist_ok=True)

    def run_table1(self) -> dict:
        """Strategy comparison table."""
        from src.evaluation.strategy_comparator import StrategyComparator
        print("\n[Suite] Running Table 1: Strategy Comparison...")
        comparator = StrategyComparator(
            sigmas=[0.0, 0.001, 0.002, 0.005], num_trials=3, max_samples=2_000_000
        )
        results = comparator.compare(
            self.residuals, total_bits=50000,
            strategies=["sign", "magnitude_aware", "lwe", "neural"],
        )
        comparator.print_table(results)
        self._save(results, "table1_strategy_comparison.json")
        return results

    def run_table2(self) -> dict:
        """Baseline comparison table."""
        from src.evaluation.baseline_comparator import BaselineComparator
        print("\n[Suite] Running Table 2: Baseline Comparison...")
        comparator = BaselineComparator(
            sigmas=[0.0, 0.001, 0.002, 0.005], num_trials=3, max_samples=2_000_000
        )
        results = comparator.compare(self.residuals, total_bits=50000)
        comparator.print_table(results)
        self._save(results, "table2_baseline_comparison.json")
        return results

    def run_table3(self) -> dict:
        """Real PPL and task accuracy using correct residual method."""
        import math
        from src.embedding.intelligent_embedder import IntelligentEmbedder
        from src.model.weight_patcher           import WeightPatcher
        from src.evaluation.task_accuracy_experiment import TaskAccuracyExperiment
        from src.core.types                     import EmbeddingConfig

        print("\n[Suite] Running Table 3: PPL and Task Accuracy...")
        patcher  = WeightPatcher()
        config   = EmbeddingConfig(total_payload_bits=50000, embedding_strategy="sign")
        embedder = IntelligentEmbedder(config)
        result   = embedder.embed("A" * 500, self.residuals)

        model_device = next(self.model.parameters()).device

        def compute_ppl(texts, batch_size=2, max_length=512):
            self.model.eval()
            total_loss = total_tokens = 0
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch  = texts[i:i+batch_size]
                    inputs = self.tokenizer(
                        batch, return_tensors="pt", truncation=True,
                        max_length=max_length, padding=True
                    )
                    ids    = inputs["input_ids"].to(model_device)
                    labels = ids.clone()
                    if "attention_mask" in inputs:
                        labels[inputs["attention_mask"].to(model_device) == 0] = -100
                    try:
                        out = self.model(input_ids=ids, labels=labels)
                        n   = (labels != -100).sum().item()
                        total_loss   += out.loss.item() * n
                        total_tokens += n
                    except Exception as e:
                        print(f"  Batch {i} failed: {e}")
            return math.exp(total_loss / max(total_tokens, 1))

        # Load WikiText-2
        try:
            from datasets import load_dataset
            ds    = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            texts = [x["text"] for x in ds if len(x["text"].strip()) > 100][:self.n_ppl]
            print(f"[Suite] Loaded {len(texts)} WikiText-2 samples")
        except Exception:
            texts = ["The quick brown fox jumps over the lazy dog. " * 10] * 20

        # Baseline PPL
        ppl_base = compute_ppl(texts)
        print(f"[Suite] Baseline PPL: {ppl_base:.4f}")

        # Patch and measure
        patcher.patch(self.module_refs, self.nf4_dequant, result.embedded_residuals)
        ppl_emb     = compute_ppl(texts)
        degradation = (ppl_emb - ppl_base) / max(ppl_base, 1e-8)
        print(f"[Suite] Embedded PPL: {ppl_emb:.4f}  ({degradation*100:.4f}%)")

        # Restore for task accuracy baseline
        patcher.restore(self.module_refs, self.nf4_dequant, self.residuals)

        task_exp  = TaskAccuracyExperiment(self.model, self.tokenizer, device=str(model_device))
        questions = task_exp.load_mmlu(self.n_task)

        acc_base = task_exp.compute_accuracy(questions)
        print(f"[Suite] Baseline MMLU: {acc_base*100:.2f}%")

        patcher.patch(self.module_refs, self.nf4_dequant, result.embedded_residuals)
        acc_emb  = task_exp.compute_accuracy(questions)
        acc_loss = acc_base - acc_emb
        print(f"[Suite] Embedded MMLU: {acc_emb*100:.2f}%")

        patcher.restore(self.module_refs, self.nf4_dequant, self.residuals)

        results = {
            "ppl": {
                "baseline":    ppl_base,
                "embedded":    ppl_emb,
                "degradation": degradation,
                "passed":      degradation <= 0.02,
            },
            "task_accuracy": {
                "baseline":    acc_base,
                "embedded":    acc_emb,
                "loss":        acc_loss,
                "passed":      acc_loss <= 0.01,
            },
        }
        self._save(results, "table3_ppl_task_accuracy.json")

        print(f"\n  WikiText-2 PPL  : {ppl_base:.4f} → {ppl_emb:.4f}  "
              f"[{'PASS' if degradation<=0.02 else 'FAIL'}]")
        print(f"  MMLU Accuracy   : {acc_base*100:.2f}% → {acc_emb*100:.2f}%  "
              f"[{'PASS' if acc_loss<=0.01 else 'FAIL'}]")
        return results

    def run_capacity_curve(self) -> dict:
        """Capacity vs BER tradeoff."""
        from src.evaluation.capacity_robustness_tradeoff import CapacityRobustnessAnalyser
        print("\n[Suite] Running Capacity-Robustness Curve...")
        analyser = CapacityRobustnessAnalyser(num_trials=2)
        surface  = analyser.sweep(
            self.residuals,
            payload_sizes=[1000, 5000, 10000, 25000, 50000],
            sigmas=[0.0, 0.001, 0.002, 0.005],
        )
        analyser.print_surface(surface)
        surface_json = {f"{p}_{s}": v for (p, s), v in surface.items()}
        self._save(surface_json, "figure2_capacity_curve.json")
        return surface

    def run_all(self) -> dict:
        print(f"\n{'='*60}\n  FULL EXPERIMENT SUITE\n{'='*60}")
        results = {}
        results["table1"]  = self.run_table1()
        results["table2"]  = self.run_table2()
        results["table3"]  = self.run_table3()
        results["figure2"] = self.run_capacity_curve()
        print(f"\n{'='*60}\n  ALL COMPLETE → {self.output_dir}\n{'='*60}")
        return results

    def _save(self, data, filename):
        path = os.path.join(self.output_dir, filename)
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            print(f"[Suite] Saved {filename}")
        except Exception as e:
            print(f"[Suite] Could not save {filename}: {e}")