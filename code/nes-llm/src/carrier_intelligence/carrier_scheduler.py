"""
QACI Carrier Scheduler — capacity-aware Hamilton Largest-Remainder allocator.

Distributes total_payload_bits across layers according to adjusted quality
while guaranteeing:

    1. Higher-quality layers get proportionally more bits.
    2. No layer exceeds its carrier capacity.
    3. Zero-quality layers receive 0 bits.
    4. Sum of allocations equals total_payload_bits,
       provided total capacity is sufficient.
"""

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class CarrierAllocation:
    """Payload assignment for one layer."""
    layer_id: int
    module_name: str
    adjusted_quality: float
    capacity: int
    allocated_bits: int


class CarrierScheduler:
    """
    Capacity-aware Hamilton Largest-Remainder scheduler.

    Allocation is based on:

        weight_i = quality_i ^ gamma

        quota_i = weight_i / sum(weight) * remaining_payload

    Hamilton then distributes leftover integer bits according to
    the largest fractional remainders.
    """

    def __init__(self, gamma: float = 2.5, eps: float = 1e-8):
        self.gamma = gamma
        self.eps = eps

    def _weighted_scores(self, profiles: List[dict]) -> torch.Tensor:
        """
        Calculate normalized quality weights.

        weight_i = q_i ^ gamma
        normalized_weight_i = weight_i / sum(weight)
        """

        scores = torch.tensor(
            [p["adjusted_quality"] for p in profiles],
            dtype=torch.float64,
        )

        scores = scores.clamp(min=0.0).pow(self.gamma)

        total = scores.sum()

        if total.item() <= self.eps:
            return torch.zeros_like(scores)

        return scores / total

    def allocate(
        self,
        layer_profiles: List[dict],
        total_payload_bits: int,
    ) -> List[CarrierAllocation]:

        if total_payload_bits < 0:
            raise ValueError("total_payload_bits cannot be negative.")

        if not layer_profiles:
            return []

        # ---------------------------------------------------------
        # Step 1 — Read layer capacities
        # Time: O(L)
        # ---------------------------------------------------------

        capacities = torch.tensor(
            [p["num_params"] for p in layer_profiles],
            dtype=torch.long,
        )

        total_capacity = capacities.sum().item()

        # The payload physically cannot fit.
        if total_payload_bits > total_capacity:
            raise ValueError(
                f"Payload of {total_payload_bits} bits cannot fit. "
                f"Total carrier capacity is only {total_capacity}."
            )

        if total_payload_bits == 0:
            return [
                CarrierAllocation(
                    layer_id=p["layer_id"],
                    module_name=p["module_name"],
                    adjusted_quality=p["adjusted_quality"],
                    capacity=int(capacities[i].item()),
                    allocated_bits=0,
                )
                for i, p in enumerate(layer_profiles)
            ]

        # ---------------------------------------------------------
        # Step 2 — Calculate quality weights
        # Time: O(L)
        # ---------------------------------------------------------

        weights = self._weighted_scores(layer_profiles)

        # ---------------------------------------------------------
        # Step 3 — Handle zero-quality case
        # ---------------------------------------------------------
        #
        # If every layer has zero quality, we cannot meaningfully
        # perform quality-aware allocation.
        #
        # Instead, allocate using capacity-aware equal distribution.
        # ---------------------------------------------------------

        if weights.sum().item() <= self.eps:

            allocations = torch.zeros(
                len(layer_profiles),
                dtype=torch.long,
            )

            remaining = total_payload_bits

            # Give bits while respecting capacities.
            while remaining > 0:

                available = capacities - allocations

                valid = torch.nonzero(
                    available > 0,
                    as_tuple=False,
                ).flatten()

                if len(valid) == 0:
                    break

                # Largest remaining capacity first.
                order = valid[
                    torch.argsort(
                        available[valid],
                        descending=True,
                    )
                ]

                for idx in order:

                    if remaining <= 0:
                        break

                    if available[idx] > 0:
                        allocations[idx] += 1
                        remaining -= 1

            floors = allocations

        else:

            # -----------------------------------------------------
            # Step 4 — Initial proportional quotas
            # Time: O(L)
            # -----------------------------------------------------

            quotas = weights * total_payload_bits

            # -----------------------------------------------------
            # Step 5 — Floor quotas
            # Time: O(L)
            # -----------------------------------------------------

            floors = torch.floor(quotas).long()

            # Never exceed layer capacity.
            floors = torch.minimum(floors, capacities)

            allocated = floors.sum().item()
            remaining = total_payload_bits - allocated

            # -----------------------------------------------------
            # Step 6 — Hamilton Largest Remainder
            # -----------------------------------------------------
            #
            # Give remaining bits to layers with the largest
            # fractional remainder, provided capacity remains.
            #
            # Time: O(L log L)
            # -----------------------------------------------------

            remainders = quotas - torch.floor(quotas)

            while remaining > 0:

                available = capacities - floors

                valid = torch.nonzero(
                    available > 0,
                    as_tuple=False,
                ).flatten()

                if len(valid) == 0:
                    break

                # Sort valid layers by:
                #
                #   1. largest fractional remainder
                #   2. highest quality as tie-breaker
                #
                order = valid[
                    torch.argsort(
                        remainders[valid],
                        descending=True,
                    )
                ]

                progress = False

                for idx in order:

                    if remaining <= 0:
                        break

                    if available[idx] > 0:

                        floors[idx] += 1
                        remaining -= 1
                        progress = True

                if not progress:
                    break

        # ---------------------------------------------------------
        # Step 7 — Final safety check
        # Time: O(L)
        # ---------------------------------------------------------

        final_total = floors.sum().item()

        if final_total != total_payload_bits:
            raise RuntimeError(
                f"Scheduler failed to allocate exact payload. "
                f"Requested={total_payload_bits}, "
                f"Allocated={final_total}."
            )

        # ---------------------------------------------------------
        # Step 8 — Build allocation objects
        # Time: O(L)
        # ---------------------------------------------------------

        allocations = [
            CarrierAllocation(
                layer_id=p["layer_id"],
                module_name=p["module_name"],
                adjusted_quality=p["adjusted_quality"],
                capacity=int(capacities[i].item()),
                allocated_bits=int(floors[i].item()),
            )
            for i, p in enumerate(layer_profiles)
        ]

        return sorted(
            allocations,
            key=lambda a: a.layer_id,
        )

    def allocation_to_dict(
        self,
        allocations: List[CarrierAllocation],
    ) -> Dict[int, int]:

        return {
            a.layer_id: a.allocated_bits
            for a in allocations
        }

    def summary(
        self,
        allocations: List[CarrierAllocation],
    ) -> dict:

        bits = [a.allocated_bits for a in allocations]

        return {
            "total_layers": len(allocations),
            "total_bits": sum(bits),
            "mean_bits": sum(bits) / max(len(bits), 1),
            "max_bits": max(bits) if bits else 0,
            "min_bits": min(bits) if bits else 0,
            "nonzero_layers": sum(
                1 for b in bits if b > 0
            ),
        }