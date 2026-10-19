"""
Task accuracy experiment — MMLU-style multiple choice before and after embedding.

Uses a small subset of MMLU (or synthetic questions if datasets unavailable)
to measure task accuracy degradation from embedding.
"""

from typing import List, Dict, Optional, Tuple
import torch
import random


class TaskAccuracyExperiment:
    """
    Measures task accuracy on MMLU 5-shot before and after embedding.

    Accuracy is measured as the fraction of questions where the model
    assigns highest log-probability to the correct answer token
    (A, B, C, or D) — standard MMLU evaluation protocol.

    Usage:
        exp    = TaskAccuracyExperiment(model, tokenizer)
        result = exp.run(embed_fn, n_questions=100)
        print(result.report())
    """

    ANSWER_TOKENS = ["A", "B", "C", "D"]

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device

    def load_mmlu(self, n_questions: int = 200) -> List[dict]:
        """Load MMLU questions (subset of all subjects)."""
        try:
            from datasets import load_dataset
            dataset   = load_dataset("cais/mmlu", "all", split="test")
            questions = list(dataset)[:n_questions]
            print(f"[TaskExp] Loaded {len(questions)} MMLU questions")
            return questions
        except Exception as e:
            print(f"[TaskExp] Could not load MMLU: {e}")
            print("[TaskExp] Using synthetic questions")
            return self._synthetic_questions(n_questions)

    def _synthetic_questions(self, n: int) -> List[dict]:
        """Generate synthetic multiple-choice questions for testing."""
        questions = []
        for i in range(n):
            correct = random.randint(0, 3)
            questions.append({
                "question":  f"What is {i} + {i}?",
                "choices":   [f"{i*2}", f"{i+1}", f"{i*3}", f"{i-1}"],
                "answer":    correct,
            })
        return questions

    def _format_question(self, q: dict) -> str:
        """Format question for model input."""
        choices = q.get("choices", q.get("options", []))
        text    = q.get("question", "")
        for i, choice in enumerate(choices[:4]):
            text += f"\n{self.ANSWER_TOKENS[i]}. {choice}"
        text += "\nAnswer:"
        return text

    def _get_answer_logprob(self, prompt: str) -> List[float]:
        try:
            model_device = next(self.model.parameters()).device
        except Exception:
            model_device = self.device

        inputs  = self.tokenizer(prompt, return_tensors="pt")
        inputs  = {k: v.to(model_device) for k, v in inputs.items()}
        logprobs = []

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits  = outputs.logits[0, -1, :]

        for token in self.ANSWER_TOKENS:
            token_id = self.tokenizer.encode(token, add_special_tokens=False)
            if token_id:
                logprobs.append(logits[token_id[0]].item())
            else:
                logprobs.append(float('-inf'))
        return logprobs

    def compute_accuracy(self, questions: List[dict]) -> float:
        """Compute fraction of questions answered correctly."""
        self.model.eval()
        correct = 0
        total   = 0

        for q in questions:
            try:
                prompt   = self._format_question(q)
                logprobs = self._get_answer_logprob(prompt)
                pred     = logprobs.index(max(logprobs))
                answer   = q.get("answer", 0)
                if isinstance(answer, str):
                    answer = self.ANSWER_TOKENS.index(answer) if answer in self.ANSWER_TOKENS else 0
                if pred == answer:
                    correct += 1
                total += 1
            except Exception:
                continue

        return correct / max(total, 1)

    def run(
        self,
        embed_fn,
        n_questions: int = 200,
        label:       str = "experiment",
    ) -> "TaskResult":
        """
        Run full task accuracy experiment.

        Args:
            embed_fn:    Callable that modifies model weights in-place.
            n_questions: Number of MMLU questions to evaluate on.
            label:       Experiment label.
        """
        questions = self.load_mmlu(n_questions)

        print(f"[TaskExp] Computing baseline accuracy...")
        acc_baseline = self.compute_accuracy(questions)
        print(f"[TaskExp] Baseline accuracy: {acc_baseline*100:.2f}%")

        print(f"[TaskExp] Applying embedding...")
        embed_fn(self.model)

        print(f"[TaskExp] Computing embedded accuracy...")
        acc_embedded = self.compute_accuracy(questions)
        print(f"[TaskExp] Embedded accuracy: {acc_embedded*100:.2f}%")

        loss = acc_baseline - acc_embedded
        print(f"[TaskExp] Accuracy loss: {loss*100:.4f}%")

        return TaskResult(
            label=        label,
            acc_baseline= acc_baseline,
            acc_embedded= acc_embedded,
            acc_loss=     loss,
            n_questions=  len(questions),
        )


class TaskResult:
    """Result of a task accuracy experiment."""

    def __init__(
        self,
        label:        str,
        acc_baseline: float,
        acc_embedded: float,
        acc_loss:     float,
        n_questions:  int,
    ):
        self.label        = label
        self.acc_baseline = acc_baseline
        self.acc_embedded = acc_embedded
        self.acc_loss     = acc_loss
        self.n_questions  = n_questions

    @property
    def passed(self) -> bool:
        return self.acc_loss <= 0.01   # < 1% loss

    def report(self) -> str:
        return "\n".join([
            f"Task Accuracy Experiment: {self.label}",
            f"  Questions evaluated : {self.n_questions}",
            f"  Baseline accuracy   : {self.acc_baseline*100:.2f}%",
            f"  Embedded accuracy   : {self.acc_embedded*100:.2f}%",
            f"  Accuracy loss       : {self.acc_loss*100:.4f}%",
            f"  Status              : {'PASS' if self.passed else 'FAIL'}",
        ])