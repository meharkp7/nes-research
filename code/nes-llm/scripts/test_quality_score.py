import torch

from src.carrier_intelligence.quality_score import QualityScore


# ---------------------------------------------------------
# Create a synthetic feature matrix
# Shape: [5 carriers, 9 features]
# Time: O(N) where N = number of carriers
# ---------------------------------------------------------

features = torch.tensor([
    # magnitude variance std  quant_error stability robustness cost zscore distortion
    [0.90,    0.80,   0.85,   0.10,        0.90,     0.95,     0.10, 0.05, 0.10],
    [0.70,    0.70,   0.75,   0.20,        0.80,     0.85,     0.20, 0.10, 0.20],
    [0.50,    0.50,   0.55,   0.40,        0.60,     0.60,     0.40, 0.30, 0.40],
    [0.30,    0.40,   0.35,   0.70,        0.30,     0.30,     0.70, 0.60, 0.70],
    [0.10,    0.20,   0.15,   0.90,        0.10,     0.10,     0.90, 0.90, 0.90],
], dtype=torch.float32)


# ---------------------------------------------------------
# Compute QACI quality scores
# Time: O(N)
# ---------------------------------------------------------

scorer = QualityScore()

quality = scorer.compute(features)


# ---------------------------------------------------------
# Print results
# Time: O(N)
# ---------------------------------------------------------

print("\nQuality Scores")
print("=" * 40)

for i, score in enumerate(quality):
    print(f"Carrier {i}: {score.item():.4f}")


# ---------------------------------------------------------
# Check ranking
# Time: O(N log N)
# ---------------------------------------------------------

ranking = torch.argsort(
    quality,
    descending=True
)

print("\nRanking (best → worst):")
print(ranking.tolist())


# ---------------------------------------------------------
# Statistics
# Time: O(N)
# ---------------------------------------------------------

print("\nStatistics:")
print(scorer.statistics(quality))