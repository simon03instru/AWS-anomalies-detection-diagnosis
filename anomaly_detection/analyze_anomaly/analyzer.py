import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Analyze anomaly score distribution')
parser.add_argument('csv_file', type=str, help='Path to the CSV file containing anomaly scores')
args = parser.parse_args()

# Load your CSV file
df = pd.read_csv(args.csv_file)

# Extract anomaly scores
scores = df['anomaly_score'].values

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Histogram
axes[0, 0].hist(scores, bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Anomaly Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Anomaly Scores')
axes[0, 0].grid(True, alpha=0.3)

# 2. Histogram with KDE (to identify clusters better)
axes[0, 1].hist(scores, bins=30, edgecolor='black', alpha=0.6, density=True)
kde = gaussian_kde(scores)
x_range = np.linspace(scores.min(), scores.max(), 200)
axes[0, 1].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
axes[0, 1].set_xlabel('Anomaly Score')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Distribution with KDE')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Cumulative distribution
sorted_scores = np.sort(scores)
cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
axes[1, 0].plot(sorted_scores, cumulative, linewidth=2)
axes[1, 0].set_xlabel('Anomaly Score')
axes[1, 0].set_ylabel('Cumulative Probability')
axes[1, 0].set_title('Cumulative Distribution')
axes[1, 0].grid(True, alpha=0.3)

# 4. Box plot and statistics
axes[1, 1].boxplot(scores, vert=True)
axes[1, 1].set_ylabel('Anomaly Score')
axes[1, 1].set_title('Box Plot')
axes[1, 1].grid(True, alpha=0.3)

# Print statistics
print(f"Min: {scores.min():.4f}")
print(f"Max: {scores.max():.4f}")
print(f"Mean: {scores.mean():.4f}")
print(f"Median: {np.median(scores):.4f}")
print(f"Std: {scores.std():.4f}")
print(f"25th percentile: {np.percentile(scores, 25):.4f}")
print(f"75th percentile: {np.percentile(scores, 75):.4f}")

plt.tight_layout()
plt.savefig('anomaly_distribution.png', dpi=100, bbox_inches='tight')
print("\nPlot saved as 'anomaly_distribution.png'")
plt.show()

# Interactive threshold testing (optional)
print("\n--- Test different thresholds ---")
for threshold in [np.percentile(scores, p) for p in [50, 75, 90, 95, 99]]:
    n_anomalies = (scores > threshold).sum()
    pct = (n_anomalies / len(scores)) * 100
    print(f"Threshold {threshold:.4f}: {n_anomalies} anomalies ({pct:.2f}%)")