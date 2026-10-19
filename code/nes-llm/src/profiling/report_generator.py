import json
from pathlib import Path

import matplotlib.pyplot as plt


class ReportGenerator:

    @staticmethod
    def load_summary(path):

        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def create_bar_chart(
        summary,
        metric,
        output_path,
    ):

        families = []
        values = []

        for family, stats in summary.items():
            families.append(family)
            values.append(stats[f"avg_{metric}"])

        plt.figure(figsize=(10, 6))

        plt.bar(families, values)

        plt.title(f"Average {metric} by Tensor Family")
        plt.xlabel("Tensor Family")
        plt.ylabel(metric)

        plt.tight_layout()

        plt.savefig(output_path)

        plt.close()