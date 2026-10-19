import json
from collections import defaultdict
from pathlib import Path

import numpy as np


class LayerProfiler:

    TENSOR_FAMILIES = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    @staticmethod
    def get_family(tensor_name: str):

        for family in LayerProfiler.TENSOR_FAMILIES:
            if tensor_name.endswith(family):
                return family

        return None

    @staticmethod
    def aggregate(profile_file):

        with open(profile_file, "r") as f:
            profiles = json.load(f)

        grouped = defaultdict(list)

        for tensor_name, stats in profiles.items():

            family = LayerProfiler.get_family(tensor_name)

            if family is None:
                continue

            grouped[family].append(stats)

        results = {}

        metrics = [
            "mean",
            "std",
            "variance",
            "entropy",
            "skewness",
            "kurtosis",
        ]

        for family, entries in grouped.items():

            family_stats = {}

            for metric in metrics:

                values = [e[metric] for e in entries]

                family_stats[f"avg_{metric}"] = float(np.mean(values))
                family_stats[f"min_{metric}"] = float(np.min(values))
                family_stats[f"max_{metric}"] = float(np.max(values))

            family_stats["num_layers"] = len(entries)

            results[family] = family_stats

        return results