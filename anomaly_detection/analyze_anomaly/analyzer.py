"""
GMM-Based Bimodal Cluster Detection for Anomaly Threshold Selection

Uses Gaussian Mixture Model to:
1. Fit two Gaussian distributions to anomaly scores
2. Find the intersection point between distributions
3. Use intersection as threshold for anomaly detection

Based on the paper's methodology of detecting bimodal distributions
and setting threshold at the natural separation point.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import argparse

def fit_gmm_and_find_threshold(scores, n_components=2, random_state=42):
    """
    Fit Gaussian Mixture Model and find threshold between components.
    
    Parameters:
    - scores: array of anomaly scores
    - n_components: number of Gaussian components (default: 2 for bimodal)
    - random_state: random seed for reproducibility
    
    Returns:
    - gmm: fitted GaussianMixture model
    - threshold: intersection point between the two Gaussians
    - cluster_info: dict with information about each cluster
    """
    
    # Reshape for sklearn
    scores_reshaped = scores.reshape(-1, 1)
    
    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(scores_reshaped)
    
    # Get parameters
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_
    
    # Sort by mean (ascending)
    sorted_idx = np.argsort(means)
    means = means[sorted_idx]
    stds = stds[sorted_idx]
    weights = weights[sorted_idx]
    
    # Find intersection point between two Gaussians
    # This is where the two probability densities are equal
    # For two Gaussians, we solve: w1*N(μ1,σ1) = w2*N(μ2,σ2)
    
    if n_components == 2:
        mu1, mu2 = means[0], means[1]
        sigma1, sigma2 = stds[0], stds[1]
        w1, w2 = weights[0], weights[1]
        
        # Find intersection by evaluating densities across the range
        x_range = np.linspace(scores.min(), scores.max(), 10000)
        pdf1 = w1 * norm.pdf(x_range, mu1, sigma1)
        pdf2 = w2 * norm.pdf(x_range, mu2, sigma2)
        
        # Find where PDFs cross (difference changes sign)
        diff = pdf1 - pdf2
        # Find the crossing point between the two peaks
        crossing_indices = np.where(np.diff(np.sign(diff)))[0]
        
        if len(crossing_indices) > 0:
            # Use the crossing point closest to the midpoint between means
            midpoint = (mu1 + mu2) / 2
            crossing_points = x_range[crossing_indices]
            threshold = crossing_points[np.argmin(np.abs(crossing_points - midpoint))]
        else:
            # Fallback: use midpoint between means
            threshold = midpoint
    else:
        threshold = np.mean(means)
    
    # Cluster information
    cluster_info = {
        'means': means,
        'stds': stds,
        'weights': weights,
        'labels': ['Normal', 'Anomaly'] if n_components == 2 else [f'Cluster {i+1}' for i in range(n_components)]
    }
    
    return gmm, threshold, cluster_info

def main():
    parser = argparse.ArgumentParser(description='GMM-based bimodal anomaly threshold detection')
    parser.add_argument('csv_file', type=str, help='Path to CSV file with anomaly scores')
    parser.add_argument('plot_name', type=str, nargs='?', default=None,
                        help='Name for the output plot (without extension)')
    parser.add_argument('--show-stats', action='store_true',
                        help='Show detailed statistics')
    args = parser.parse_args()
    
    # Set output filename based on plot_name
    if args.plot_name:
        args.output = f'gmm_threshold_{args.plot_name}.png'
    else:
        args.output = 'gmm_threshold_analysis.png'
    
    # Load data
    df = pd.read_csv(args.csv_file)
    scores = df['anomaly_score'].values
    
    print("="*70)
    print("GMM-BASED ANOMALY THRESHOLD DETECTION")
    print("="*70)
    print(f"\nDataset: {args.csv_file}")
    print(f"Total samples: {len(scores)}")
    print(f"Score range: [{scores.min():.6f}, {scores.max():.6f}]")
    print(f"Mean: {scores.mean():.6f}, Std: {scores.std():.6f}")
    
    # Fit GMM
    print("\nFitting Gaussian Mixture Model with 2 components...")
    gmm, threshold, cluster_info = fit_gmm_and_find_threshold(scores)
    
    # Calculate anomaly rate
    n_anomalies = (scores > threshold).sum()
    anomaly_rate = n_anomalies / len(scores)
    
    # Print results
    print("\n" + "="*70)
    print("GMM RESULTS")
    print("="*70)
    
    means = cluster_info['means']
    stds = cluster_info['stds']
    weights = cluster_info['weights']
    labels = cluster_info['labels']
    
    for i in range(len(means)):
        n_in_cluster = int(weights[i] * len(scores))
        pct_in_cluster = weights[i] * 100
        print(f"\n{labels[i]} Cluster (Gaussian {i+1}):")
        print(f"  Mean (μ):       {means[i]:.6f}")
        print(f"  Std Dev (σ):    {stds[i]:.6f}")
        print(f"  Weight:         {weights[i]:.4f}")
        print(f"  Est. samples:   {n_in_cluster} ({pct_in_cluster:.2f}%)")
        print(f"  Range (μ±2σ):   [{means[i]-2*stds[i]:.6f}, {means[i]+2*stds[i]:.6f}]")
    
    print(f"\n" + "="*70)
    print("THRESHOLD")
    print("="*70)
    print(f"\nGaussian intersection point: {threshold:.6f}")
    print(f"Detected anomalies:          {n_anomalies} ({anomaly_rate*100:.3f}%)")
    
    
    # Visualization
    print(f"\n" + "="*70)
    print("Creating visualization...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Prepare data for plotting
    x_range = np.linspace(scores.min(), scores.max(), 1000)
    normal_scores = scores[scores <= threshold]
    anomaly_scores = scores[scores > threshold]
    
    # Plot 1: GMM fit with both Gaussians
    ax1 = fig.add_subplot(gs[0, :])
    
    # Histogram
    ax1.hist(scores, bins=60, density=True, alpha=0.5, color='lightblue', 
             edgecolor='black', label='Data')
    
    # Individual Gaussians
    for i in range(len(means)):
        gaussian = weights[i] * norm.pdf(x_range, means[i], stds[i])
        color = 'green' if i == 0 else 'red'
        ax1.plot(x_range, gaussian, '--', linewidth=2, color=color, 
                label=f'{labels[i]} Gaussian (μ={means[i]:.3f})')
    
    # Combined GMM
    gmm_pdf = np.zeros_like(x_range)
    for i in range(len(means)):
        gmm_pdf += weights[i] * norm.pdf(x_range, means[i], stds[i])
    ax1.plot(x_range, gmm_pdf, 'b-', linewidth=3, label='GMM (combined)', alpha=0.8)
    
    # Threshold line
    ax1.axvline(threshold, color='black', linestyle='--', linewidth=2.5, 
               label=f'Threshold: {threshold:.4f}')
    
    ax1.set_xlabel('Anomaly Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('Gaussian Mixture Model Fit', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Separated clusters (histogram)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(normal_scores, bins=30, alpha=0.7, color='green', 
             edgecolor='black', label=f'Normal ({len(normal_scores)})')
    ax2.hist(anomaly_scores, bins=30, alpha=0.7, color='red', 
             edgecolor='black', label=f'Anomaly ({len(anomaly_scores)})')
    ax2.axvline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax2.set_xlabel('Anomaly Score', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Separated Clusters', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative distribution
    ax3 = fig.add_subplot(gs[1, 1])
    sorted_scores = np.sort(scores)
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax3.plot(sorted_scores, cumulative, linewidth=2, color='purple')
    ax3.axvline(threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax3.axhline(1 - anomaly_rate, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('Anomaly Score', fontsize=11)
    ax3.set_ylabel('Cumulative Probability', fontsize=11)
    ax3.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Classification results
    ax4 = fig.add_subplot(gs[1, 2])
    labels_data = ['Normal\n(Below threshold)', 'Anomaly\n(Above threshold)']
    counts = [len(normal_scores), len(anomaly_scores)]
    colors_bar = ['green', 'red']
    bars = ax4.bar(labels_data, counts, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = (count / len(scores)) * 100
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({pct:.2f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('Classification Results', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('GMM-Based Anomaly Threshold Detection', fontsize=16, fontweight='bold', y=0.995)
    
    # Save plot
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved as '{args.output}'")
    
    # Optional: detailed statistics
    if args.show_stats:
        print(f"\n" + "="*70)
        print("DETAILED STATISTICS")
        print("="*70)
        
        print("\nNormal cluster statistics:")
        print(f"  Count: {len(normal_scores)}")
        print(f"  Mean: {normal_scores.mean():.6f}")
        print(f"  Std: {normal_scores.std():.6f}")
        print(f"  Min: {normal_scores.min():.6f}")
        print(f"  Max: {normal_scores.max():.6f}")
        
        if len(anomaly_scores) > 0:
            print("\nAnomaly cluster statistics:")
            print(f"  Count: {len(anomaly_scores)}")
            print(f"  Mean: {anomaly_scores.mean():.6f}")
            print(f"  Std: {anomaly_scores.std():.6f}")
            print(f"  Min: {anomaly_scores.min():.6f}")
            print(f"  Max: {anomaly_scores.max():.6f}")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
    print(f"\n✓ Recommended threshold: {threshold:.6f}")
    print(f"✓ Expected anomaly rate: {anomaly_rate*100:.3f}%")
    print("\nUse this threshold in your anomaly detection system.")
    print("="*70)
    
    plt.show()

if __name__ == '__main__':
    main()