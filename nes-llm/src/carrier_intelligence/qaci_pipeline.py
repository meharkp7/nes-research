"""
QACI Pipeline — ties LayerProfiler → CarrierScheduler → CarrierSelector together.

Usage:
    pipeline  = QACIPipeline(total_layers=32)
    result    = pipeline.select(residuals, total_payload_bits=50000)
    # result.selected_indices  →  {layer_id: [flat_indices]}
    # result.layer_allocation  →  {layer_id: num_bits}
"""

from typing import Dict, List

import torch

from src.carrier_intelligence.layer_profiler    import LayerProfiler
from src.carrier_intelligence.carrier_scheduler import CarrierScheduler
from src.carrier_intelligence.selector          import CarrierSelector


class QACIPipeline:
    """
    Full QACI carrier selection pipeline.

    Steps per call:
        1. LayerProfiler.profile()      → quality + position bias per layer
        2. CarrierScheduler.allocate()  → bit budget per layer
        3. CarrierSelector.select()     → top-K indices per layer
    """

    def __init__(self, total_layers: int, gamma: float = 2.5):
        self.total_layers = total_layers
        self.profiler = LayerProfiler()
        self.scheduler = CarrierScheduler(gamma=gamma)

    def select(
        self,
        residuals: Dict[int, torch.Tensor],
        total_payload_bits: int,
        fp16_weights: Dict[int, torch.Tensor] = None,
        quantized_weights: Dict[int, torch.Tensor] = None,
        module_names: Dict[int, str] = None,
        use_position_bias: bool = True,
    ) -> "CarrierSelectionResult":
        """
        Args:
            residuals:          {layer_id: residual_tensor}
            total_payload_bits: Total bits to embed.
            fp16_weights:       Optional — enables full QACI feature scoring.
            quantized_weights:  Optional — enables full QACI feature scoring.
            module_names:       Optional — for logging/debugging.
        """
        module_names      = module_names      or {lid: "unknown"  for lid in residuals}
        fp16_weights      = fp16_weights      or {}
        quantized_weights = quantized_weights or {}

        # Step 1 — Profile every layer
        layer_profiles = [
            self.profiler.profile(
                residual=    residuals[lid],
                layer_id=    lid,
                module_name= module_names.get(lid, "unknown"),
                total_layers=self.total_layers,
            )
            for lid in sorted(residuals.keys())
        ]
        # Research mode can remove the hand-designed positional prior and rank
        # layers only by measured tensor statistics.
        if not use_position_bias:
            for profile in layer_profiles:
                profile["position_bias"] = 1.0
                profile["adjusted_quality"] = profile["quality_score"]

        # Step 2 — Allocate bits (Hamilton Largest-Remainder)
        allocations = self.scheduler.allocate(layer_profiles, total_payload_bits)
        alloc_dict  = self.scheduler.allocation_to_dict(allocations)
        profile_map = {p["layer_id"]: p for p in layer_profiles}

        # Step 3 — Select carrier indices per layer
        selected_indices = {}
        quality_scores   = {}

        for alloc in allocations:
            lid      = alloc.layer_id
            num_bits = alloc.allocated_bits

            if num_bits == 0:
                selected_indices[lid] = []
                quality_scores[lid]   = torch.zeros(residuals[lid].numel())
                continue

            selector = CarrierSelector(
                layer_quality_factor=profile_map[lid]["adjusted_quality"]
            )
            fp16      = fp16_weights.get(lid)
            quantized = quantized_weights.get(lid)

            if fp16 is not None and quantized is not None:
                indices, scores = selector.select(residuals[lid], fp16, quantized, num_bits)
            else:
                indices = selector.select_by_magnitude(residuals[lid], num_bits)
                scores  = residuals[lid].flatten().abs()

            selected_indices[lid] = indices
            quality_scores[lid]   = scores

        return CarrierSelectionResult(
            selected_indices=selected_indices,
            layer_allocation=alloc_dict,
            layer_profiles=profile_map,
            quality_scores=quality_scores,
            total_selected=sum(len(v) for v in selected_indices.values()),
        )


class CarrierSelectionResult:
    def __init__(self, selected_indices, layer_allocation, layer_profiles,
                 quality_scores, total_selected):
        self.selected_indices = selected_indices
        self.layer_allocation = layer_allocation
        self.layer_profiles   = layer_profiles
        self.quality_scores   = quality_scores
        self.total_selected   = total_selected

    def summary(self) -> dict:
        return {
            "total_layers":   len(self.layer_allocation),
            "total_bits":     self.total_selected,
            "nonzero_layers": sum(1 for b in self.layer_allocation.values() if b > 0),
            "per_layer_bits": self.layer_allocation,
        }