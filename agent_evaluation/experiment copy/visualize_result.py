import matplotlib.pyplot as plt
import numpy as np

# Data for each experiment
experiments = ['Exp 1', 'Exp 2', 'Exp 3', 'Exp 4', 'Exp 5', 'Exp 6', 'Exp 7']

# Faithfulness scores
faith_120b = [0.7531, 0.5439, 0.7161, 0.8265, 0.9235, 0.7464, 0.8235]
faith_20b = [0.8391, 0.6174, 0.8049, 0.7436, 0.7689, 0.6952, 0.7200]
faith_nano = [0.7255, 0.9091, 0.6753, 0.7759, 0.7889, 0.5909, 0.6905]

# Relevancy scores
rel_120b = [0.7946, 0.8571, 0.8155, 0.8577, 0.7711, 0.7673, 0.8075]
rel_20b = [0.7900, 0.7839, 0.8163, 0.8586, 0.8138, 0.8634, 0.8500]
rel_nano = [0.8361, 0.8129, 0.8155, 0.8060, 0.7923, 0.7364, 0.7750]

# Overall scores
overall_120b = [0.7739, 0.7005, 0.7658, 0.8421, 0.8473, 0.7569, 0.8155]
overall_20b = [0.8145, 0.7006, 0.8106, 0.8011, 0.7910, 0.7793, 0.7850]
overall_nano = [0.7808, 0.8610, 0.7454, 0.7909, 0.7906, 0.6637, 0.7327]

# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 14))

# Subplot 1: Faithfulness
ax1 = axes[0]
ax1.plot(experiments, faith_120b, marker='o', linewidth=2.5, markersize=10, 
         label='GPT OSS 120B', color='#3498db', linestyle='-')
ax1.plot(experiments, faith_20b, marker='s', linewidth=2.5, markersize=10, 
         label='GPT OSS 20B', color='#2ecc71', linestyle='-')
ax1.plot(experiments, faith_nano, marker='^', linewidth=2.5, markersize=10, 
         label='GPT 4 Nano', color='#e74c3c', linestyle='-')

ax1.set_ylabel('Faithfulness Score', fontsize=12, fontweight='bold')
ax1.set_title('Faithfulness Across Experiments', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(0.5, 1.0)

# Add average lines
ax1.axhline(y=np.mean(faith_120b), color='#3498db', linestyle=':', alpha=0.5, linewidth=1.5)
ax1.axhline(y=np.mean(faith_20b), color='#2ecc71', linestyle=':', alpha=0.5, linewidth=1.5)
ax1.axhline(y=np.mean(faith_nano), color='#e74c3c', linestyle=':', alpha=0.5, linewidth=1.5)

# Subplot 2: Relevancy
ax2 = axes[1]
ax2.plot(experiments, rel_120b, marker='o', linewidth=2.5, markersize=10, 
         label='GPT OSS 120B', color='#3498db', linestyle='-')
ax2.plot(experiments, rel_20b, marker='s', linewidth=2.5, markersize=10, 
         label='GPT OSS 20B', color='#2ecc71', linestyle='-')
ax2.plot(experiments, rel_nano, marker='^', linewidth=2.5, markersize=10, 
         label='GPT 4 Nano', color='#e74c3c', linestyle='-')

ax2.set_ylabel('Relevancy Score', fontsize=12, fontweight='bold')
ax2.set_title('Relevancy Across Experiments', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim(0.7, 0.9)

# Add average lines
ax2.axhline(y=np.mean(rel_120b), color='#3498db', linestyle=':', alpha=0.5, linewidth=1.5)
ax2.axhline(y=np.mean(rel_20b), color='#2ecc71', linestyle=':', alpha=0.5, linewidth=1.5)
ax2.axhline(y=np.mean(rel_nano), color='#e74c3c', linestyle=':', alpha=0.5, linewidth=1.5)

# Subplot 3: Overall
ax3 = axes[2]
ax3.plot(experiments, overall_120b, marker='o', linewidth=2.5, markersize=10, 
         label='GPT OSS 120B', color='#3498db', linestyle='-')
ax3.plot(experiments, overall_20b, marker='s', linewidth=2.5, markersize=10, 
         label='GPT OSS 20B', color='#2ecc71', linestyle='-')
ax3.plot(experiments, overall_nano, marker='^', linewidth=2.5, markersize=10, 
         label='GPT 4 Nano', color='#e74c3c', linestyle='-')

ax3.set_xlabel('Experiment', fontsize=12, fontweight='bold')
ax3.set_ylabel('Overall Score', fontsize=12, fontweight='bold')
ax3.set_title('Overall Performance Across Experiments', fontsize=14, fontweight='bold')
ax3.legend(loc='best', fontsize=10, framealpha=0.9)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_ylim(0.65, 0.9)

# Add average lines
ax3.axhline(y=np.mean(overall_120b), color='#3498db', linestyle=':', alpha=0.5, linewidth=1.5)
ax3.axhline(y=np.mean(overall_20b), color='#2ecc71', linestyle=':', alpha=0.5, linewidth=1.5)
ax3.axhline(y=np.mean(overall_nano), color='#e74c3c', linestyle=':', alpha=0.5, linewidth=1.5)

plt.tight_layout()
plt.savefig('experiment_performance_detailed.png', dpi=300, bbox_inches='tight')
plt.savefig('experiment_performance_detailed.pdf', bbox_inches='tight')
plt.show()

# Print statistics
print("=" * 70)
print("EXPERIMENT STATISTICS")
print("=" * 70)
for i, exp in enumerate(experiments):
    print(f"\n{exp}:")
    print(f"  GPT OSS 120B: Faith={faith_120b[i]:.4f}, Rel={rel_120b[i]:.4f}, Overall={overall_120b[i]:.4f}")
    print(f"  GPT OSS 20B:  Faith={faith_20b[i]:.4f}, Rel={rel_20b[i]:.4f}, Overall={overall_20b[i]:.4f}")
    print(f"  GPT 4 Nano:   Faith={faith_nano[i]:.4f}, Rel={rel_nano[i]:.4f}, Overall={overall_nano[i]:.4f}")