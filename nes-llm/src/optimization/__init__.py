"""Experimental optimization primitives for distribution-constrained embedding.

These modules are research prototypes. They are not wired into the production
NES embedding registry until the model-level evaluation protocol is validated.
"""

from src.optimization.candidate_generator import Candidate, CandidateGenerator
from src.optimization.cost_function import CostWeights, EmbeddingCost
from src.optimization.optimizer import CandidateOptimizer, OptimizationResult

__all__ = [
    "Candidate",
    "CandidateGenerator",
    "CostWeights",
    "EmbeddingCost",
    "CandidateOptimizer",
    "OptimizationResult",
]
