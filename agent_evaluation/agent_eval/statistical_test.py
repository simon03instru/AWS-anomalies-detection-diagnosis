import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Data
experiments_120b = [
    {'exp': 1, 'faith_avg': 0.6615, 'relev_avg': 0.6069, 'contexts': 8},
    {'exp': 2, 'faith_avg': 0.9091, 'relev_avg': 0.4216, 'contexts': 7},
    {'exp': 3, 'faith_avg': 0.9032, 'relev_avg': 0.5279, 'contexts': 8},
    {'exp': 4, 'faith_avg': 0.8276, 'relev_avg': (0.6112 + 0.6113)/2, 'contexts': 5},
    {'exp': 5, 'faith_avg': 0.7391, 'relev_avg': 0.5686, 'contexts': 8},
    {'exp': 6, 'faith_avg': 0.8387, 'relev_avg': 0.5289, 'contexts': 7},
    {'exp': 7, 'faith_avg': 0.8333, 'relev_avg': 0.5686, 'contexts': 7},
    {'exp': 8, 'faith_avg': 0.8444, 'relev_avg': 0.5386, 'contexts': 7},
    {'exp': 9, 'faith_avg': 0.7200, 'relev_avg': 0.5978, 'contexts': 8},
    {'exp': 10, 'faith_avg': 0.8095, 'relev_avg': 0.5031, 'contexts': 6},
    {'exp': 11, 'faith_avg': 0.8214, 'relev_avg': 0.5097, 'contexts': 6},
    {'exp': 12, 'faith_avg': 0.9167, 'relev_avg': 0.5900, 'contexts': 11},
    {'exp': 13, 'faith_avg': 0.7353, 'relev_avg': 0.7915, 'contexts': 9},
    {'exp': 14, 'faith_avg': 0.9375, 'relev_avg': 0.5073, 'contexts': 13},
    {'exp': 15, 'faith_avg': (0.6383 + 0.6122)/2, 'relev_avg': 0.6103, 'contexts': 9},
    {'exp': 16, 'faith_avg': (0.9800 + 0.9787)/2, 'relev_avg': (0.5568 + 0.5497)/2, 'contexts': 7},
    {'exp': 17, 'faith_avg': 0.8125, 'relev_avg': 0.5578, 'contexts': 8},
    {'exp': 18, 'faith_avg': (0.8400 + 0.8519)/2, 'relev_avg': (0.5481 + 0.5482)/2, 'contexts': 6},
    {'exp': 19, 'faith_avg': 0.9000, 'relev_avg': 0.5327, 'contexts': 6},
    {'exp': 20, 'faith_avg': 0.8163, 'relev_avg': 0.6023, 'contexts': 13},
]

experiments_20b = [
    {'exp': 1, 'faith_avg': 0.7600, 'relev_avg': (0.5237 + 0.5257)/2, 'contexts': 12},
    {'exp': 2, 'faith_avg': 0.7000, 'relev_avg': (0.4888 + 0.4889)/2, 'contexts': 7},
    {'exp': 3, 'faith_avg': (0.4167 + 0.6667)/2, 'relev_avg': (0.5192 + 0.4703)/2, 'contexts': 14},
    {'exp': 4, 'faith_avg': 0.6562, 'relev_avg': 0.5735, 'contexts': 10},
    {'exp': 5, 'faith_avg': (0.6000 + 0.7586)/2, 'relev_avg': (0.5740 + 0.5739)/2, 'contexts': 12},
    {'exp': 6, 'faith_avg': 0.5000, 'relev_avg': 0.5933, 'contexts': 18},
    {'exp': 7, 'faith_avg': 0.5652, 'relev_avg': 0.6247, 'contexts': 7},
    {'exp': 8, 'faith_avg': 0.9130, 'relev_avg': (0.5520 + 0.5640)/2, 'contexts': 16},
    {'exp': 9, 'faith_avg': (0.9200 + 0.9231)/2, 'relev_avg': 0.6235, 'contexts': 16},
    {'exp': 10, 'faith_avg': 0.6111, 'relev_avg': (0.5513 + 0.5512)/2, 'contexts': 15},
    {'exp': 11, 'faith_avg': 0.8333, 'relev_avg': 0.5077, 'contexts': 12},
    {'exp': 12, 'faith_avg': 0.3000, 'relev_avg': (0.6769 + 0.6721)/2, 'contexts': 17},
    {'exp': 13, 'faith_avg': (0.7600 + 0.7500)/2, 'relev_avg': 0.7771, 'contexts': 17},
    {'exp': 14, 'faith_avg': (0.6216 + 0.5946)/2, 'relev_avg': (0.7510 + 0.7509)/2, 'contexts': 13},
    {'exp': 15, 'faith_avg': 0.5185, 'relev_avg': 0.5924, 'contexts': 14},
    {'exp': 16, 'faith_avg': (0.6000 + 0.5333)/2, 'relev_avg': (0.5191 + 0.5192)/2, 'contexts': 15},
    {'exp': 17, 'faith_avg': (0.1053 + 0.1081)/2, 'relev_avg': (0.7328 + 0.7346)/2, 'contexts': 9},
    {'exp': 18, 'faith_avg': (0.8571 + 0.8095)/2, 'relev_avg': (0.6326 + 0.7031)/2, 'contexts': 14},
    {'exp': 19, 'faith_avg': (0.4074 + 0.5556)/2, 'relev_avg': 0.6619, 'contexts': 11},
    {'exp': 20, 'faith_avg': (0.7500 + 0.8065)/2, 'relev_avg': (0.6618 + 0.6669)/2, 'contexts': 20},
]

experiments_4_1_mini = [
    {'exp': 1, 'faith_avg': (0.7273 + 0.6818)/2, 'relev_avg': (0.6987 + 0.7016)/2, 'contexts': 8},
    {'exp': 2, 'faith_avg': (0.8148 + 0.8400)/2, 'relev_avg': 0.7427, 'contexts': 12},
    {'exp': 3, 'faith_avg': 0.8235, 'relev_avg': 0.6373, 'contexts': 20},
    {'exp': 4, 'faith_avg': 0.8214, 'relev_avg': 0.7023, 'contexts': 17},
    {'exp': 5, 'faith_avg': 0.4783, 'relev_avg': (0.7038 + 0.6900)/2, 'contexts': 20},
    {'exp': 6, 'faith_avg': 0.9667, 'relev_avg': (0.5336 + 0.5569)/2, 'contexts': 20},
    {'exp': 7, 'faith_avg': 0.7600, 'relev_avg': 0.7512, 'contexts': 12},
    {'exp': 8, 'faith_avg': 0.8235, 'relev_avg': 0.7190, 'contexts': 20},
    {'exp': 9, 'faith_avg': 0.7778, 'relev_avg': (0.7121 + 0.7154)/2, 'contexts': 20},
    {'exp': 10, 'faith_avg': 1.0000, 'relev_avg': (0.6750 + 0.6769)/2, 'contexts': 20},
    {'exp': 11, 'faith_avg': 0.6897, 'relev_avg': (0.7217 + 0.7209)/2, 'contexts': 18},
    {'exp': 12, 'faith_avg': (0.7188 + 0.6875)/2, 'relev_avg': (0.7461 + 0.7463)/2, 'contexts': 8},
    {'exp': 13, 'faith_avg': 0.9189, 'relev_avg': 0.7336, 'contexts': 20},
    {'exp': 14, 'faith_avg': 1.0000, 'relev_avg': 0.7340, 'contexts': 20},
    {'exp': 15, 'faith_avg': (0.3158 + 0.3684)/2, 'relev_avg': (0.8018 + 0.8061)/2, 'contexts': 20},
    {'exp': 16, 'faith_avg': (0.7273 + 0.5938)/2, 'relev_avg': 0.6826, 'contexts': 20},
    {'exp': 17, 'faith_avg': (0.8158 + 0.7632)/2, 'relev_avg': (0.7037 + 0.7036)/2, 'contexts': 20},
    {'exp': 18, 'faith_avg': (0.4000 + 0.4667)/2, 'relev_avg': (0.7036 + 0.7034)/2, 'contexts': 16},
    {'exp': 19, 'faith_avg': (0.8438 + 0.8065)/2, 'relev_avg': 0.7436, 'contexts': 20},
    {'exp': 20, 'faith_avg': 0.8214, 'relev_avg': 0.4974, 'contexts': 20},
]

# Extract data
faith_120b = [exp['faith_avg'] for exp in experiments_120b]
relev_120b = [exp['relev_avg'] for exp in experiments_120b]
contexts_120b = [exp['contexts'] for exp in experiments_120b]

faith_20b = [exp['faith_avg'] for exp in experiments_20b]
relev_20b = [exp['relev_avg'] for exp in experiments_20b]
contexts_20b = [exp['contexts'] for exp in experiments_20b]

faith_4_1_mini = [exp['faith_avg'] for exp in experiments_4_1_mini]
relev_4_1_mini = [exp['relev_avg'] for exp in experiments_4_1_mini]
contexts_4_1_mini = [exp['contexts'] for exp in experiments_4_1_mini]

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Model Performance Comparison: GPT-OSS 120B vs GPT-OSS 20B vs GPT-4.1 Mini', 
             fontsize=16, fontweight='bold', y=0.995)

# Color palette
colors = ['#3498db', '#e74c3c', '#2ecc71']
model_names = ['GPT-OSS\n120B', 'GPT-OSS\n20B', 'GPT-4.1\nMini']

# 1. Box plot for Faithfulness
ax1 = axes[0, 0]
data_faith = [faith_120b, faith_20b, faith_4_1_mini]
bp1 = ax1.boxplot(data_faith, labels=model_names, patch_artist=True, 
                   notch=True, showmeans=True)
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax1.set_ylabel('Faithfulness Score', fontsize=11, fontweight='bold')
ax1.set_title('Faithfulness Distribution', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 1.05])

# Add significance stars
ax1.text(1.5, 0.95, '***', ha='center', fontsize=14, fontweight='bold')
ax1.plot([1, 2], [0.93, 0.93], 'k-', linewidth=1)

# 2. Box plot for Relevance
ax2 = axes[0, 1]
data_relev = [relev_120b, relev_20b, relev_4_1_mini]
bp2 = ax2.boxplot(data_relev, labels=model_names, patch_artist=True,
                   notch=True, showmeans=True)
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax2.set_ylabel('Relevance Score', fontsize=11, fontweight='bold')
ax2.set_title('Relevance Distribution', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0.3, 0.9])

# Add significance stars
ax2.text(1.5, 0.82, '***', ha='center', fontsize=14, fontweight='bold')
ax2.plot([1, 3], [0.80, 0.80], 'k-', linewidth=1)
ax2.text(2.5, 0.77, '***', ha='center', fontsize=14, fontweight='bold')
ax2.plot([2, 3], [0.75, 0.75], 'k-', linewidth=1)

# 3. Bar plot with error bars
ax3 = axes[0, 2]
x_pos = np.arange(len(model_names))
faith_means = [np.mean(faith_120b), np.mean(faith_20b), np.mean(faith_4_1_mini)]
faith_stds = [np.std(faith_120b, ddof=1), np.std(faith_20b, ddof=1), np.std(faith_4_1_mini, ddof=1)]
relev_means = [np.mean(relev_120b), np.mean(relev_20b), np.mean(relev_4_1_mini)]
relev_stds = [np.std(relev_120b, ddof=1), np.std(relev_20b, ddof=1), np.std(relev_4_1_mini, ddof=1)]

width = 0.35
bars1 = ax3.bar(x_pos - width/2, faith_means, width, yerr=faith_stds, 
                label='Faithfulness', alpha=0.8, color='#3498db', capsize=5)
bars2 = ax3.bar(x_pos + width/2, relev_means, width, yerr=relev_stds,
                label='Relevance', alpha=0.8, color='#e74c3c', capsize=5)

ax3.set_ylabel('Score', fontsize=11, fontweight='bold')
ax3.set_title('Mean Performance ± SD', fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(model_names)
ax3.legend(loc='lower right')
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim([0, 1.0])

# Add values on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# 4. Scatter plot: Faithfulness vs Relevance
ax4 = axes[1, 0]
ax4.scatter(faith_120b, relev_120b, alpha=0.6, s=100, color=colors[0], label='GPT-OSS 120B')
ax4.scatter(faith_20b, relev_20b, alpha=0.6, s=100, color=colors[1], label='GPT-OSS 20B')
ax4.scatter(faith_4_1_mini, relev_4_1_mini, alpha=0.6, s=100, color=colors[2], label='GPT-4.1 Mini')
ax4.set_xlabel('Faithfulness Score', fontsize=11, fontweight='bold')
ax4.set_ylabel('Relevance Score', fontsize=11, fontweight='bold')
ax4.set_title('Faithfulness vs Relevance Trade-off', fontsize=12, fontweight='bold')
ax4.legend(loc='best')
ax4.grid(alpha=0.3)

# Add ideal region
ax4.axhline(y=0.7, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax4.axvline(x=0.8, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax4.text(0.85, 0.75, 'Ideal Region', fontsize=9, color='gray', alpha=0.7)

# 5. Violin plot comparison
ax5 = axes[1, 1]
positions = [1, 2, 3, 4.5, 5.5, 6.5]
parts = ax5.violinplot([faith_120b, faith_20b, faith_4_1_mini, 
                         relev_120b, relev_20b, relev_4_1_mini],
                        positions=positions, showmeans=True, showmedians=True)

for i, pc in enumerate(parts['bodies']):
    if i < 3:
        pc.set_facecolor(colors[i])
    else:
        pc.set_facecolor(colors[i-3])
    pc.set_alpha(0.6)

ax5.set_xticks(positions)
ax5.set_xticklabels(['120B', '20B', '4.1M', '120B', '20B', '4.1M'])
ax5.set_ylabel('Score', fontsize=11, fontweight='bold')
ax5.set_title('Distribution Shapes', fontsize=12, fontweight='bold')
ax5.axvline(x=3.75, color='black', linestyle='--', linewidth=1.5)
ax5.text(2, 0.95, 'Faithfulness', ha='center', fontsize=10, fontweight='bold')
ax5.text(5.5, 0.95, 'Relevance', ha='center', fontsize=10, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

# 6. Context count distribution
ax6 = axes[1, 2]
ax6.hist([contexts_120b, contexts_20b, contexts_4_1_mini], 
         bins=15, alpha=0.6, label=model_names, color=colors)
ax6.set_xlabel('Number of Contexts', fontsize=11, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax6.set_title('Context Count Distribution', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_visualization.png', dpi=300, bbox_inches='tight')
print("Visualization saved to model_comparison_visualization.png")

# Create a second figure for additional analysis
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Additional Performance Analysis', fontsize=14, fontweight='bold')

# Correlation between contexts and performance
ax7 = axes2[0]
ax7.scatter(contexts_120b, faith_120b, alpha=0.6, s=80, color=colors[0], label='120B - Faith')
ax7.scatter(contexts_20b, faith_20b, alpha=0.6, s=80, color=colors[1], label='20B - Faith')
ax7.scatter(contexts_4_1_mini, faith_4_1_mini, alpha=0.6, s=80, color=colors[2], label='4.1M - Faith')
ax7.set_xlabel('Number of Contexts', fontsize=11, fontweight='bold')
ax7.set_ylabel('Faithfulness Score', fontsize=11, fontweight='bold')
ax7.set_title('Context Count vs Faithfulness', fontsize=12, fontweight='bold')
ax7.legend(loc='best', fontsize=8)
ax7.grid(alpha=0.3)

# Add trend lines
z1 = np.polyfit(contexts_120b, faith_120b, 1)
p1 = np.poly1d(z1)
z2 = np.polyfit(contexts_20b, faith_20b, 1)
p2 = np.poly1d(z2)
z3 = np.polyfit(contexts_4_1_mini, faith_4_1_mini, 1)
p3 = np.poly1d(z3)

x_line = np.linspace(5, 20, 100)
ax7.plot(x_line, p1(x_line), color=colors[0], linestyle='--', alpha=0.5, linewidth=2)
ax7.plot(x_line, p2(x_line), color=colors[1], linestyle='--', alpha=0.5, linewidth=2)
ax7.plot(x_line, p3(x_line), color=colors[2], linestyle='--', alpha=0.5, linewidth=2)

# Context count vs relevance
ax8 = axes2[1]
ax8.scatter(contexts_120b, relev_120b, alpha=0.6, s=80, color=colors[0], label='120B - Relev')
ax8.scatter(contexts_20b, relev_20b, alpha=0.6, s=80, color=colors[1], label='20B - Relev')
ax8.scatter(contexts_4_1_mini, relev_4_1_mini, alpha=0.6, s=80, color=colors[2], label='4.1M - Relev')
ax8.set_xlabel('Number of Contexts', fontsize=11, fontweight='bold')
ax8.set_ylabel('Relevance Score', fontsize=11, fontweight='bold')
ax8.set_title('Context Count vs Relevance', fontsize=12, fontweight='bold')
ax8.legend(loc='best', fontsize=8)
ax8.grid(alpha=0.3)

# Add trend lines
z1r = np.polyfit(contexts_120b, relev_120b, 1)
p1r = np.poly1d(z1r)
z2r = np.polyfit(contexts_20b, relev_20b, 1)
p2r = np.poly1d(z2r)
z3r = np.polyfit(contexts_4_1_mini, relev_4_1_mini, 1)
p3r = np.poly1d(z3r)

ax8.plot(x_line, p1r(x_line), color=colors[0], linestyle='--', alpha=0.5, linewidth=2)
ax8.plot(x_line, p2r(x_line), color=colors[1], linestyle='--', alpha=0.5, linewidth=2)
ax8.plot(x_line, p3r(x_line), color=colors[2], linestyle='--', alpha=0.5, linewidth=2)

plt.tight_layout()
plt.savefig('context_analysis_visualization.png', dpi=300, bbox_inches='tight')
print("Context analysis visualization saved to context_analysis_visualization.png")

print("\nAll visualizations created successfully!")