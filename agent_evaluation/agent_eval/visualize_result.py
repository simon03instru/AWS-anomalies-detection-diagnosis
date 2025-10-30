import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Process the data - calculating averages for each experiment
# GPT-OSS 120B experiments
experiments_120b = [
    {'exp': 1, 'faith_avg': 0.9778, 'relev_avg': 0.8537, 'contexts': 8},
    {'exp': 2, 'faith_avg': 0.9434, 'relev_avg': 0.7628, 'contexts': 7},
    {'exp': 3, 'faith_avg': 0.5862, 'relev_avg': 0.7682, 'contexts': 8},
    {'exp': 4, 'faith_avg': 0.7419, 'relev_avg': 0.8169, 'contexts': 5},
    {'exp': 5, 'faith_avg': (0.4444 + 0.4231)/2, 'relev_avg': 0.7856, 'contexts': 8},
    {'exp': 6, 'faith_avg': 0.8000, 'relev_avg': 0.8920, 'contexts': 7},
    {'exp': 7, 'faith_avg': 0.9483, 'relev_avg': 0.8008, 'contexts': 7},
    {'exp': 8, 'faith_avg': 0.6786, 'relev_avg': 0.7268, 'contexts': 7},
    {'exp': 9, 'faith_avg': 0.7500, 'relev_avg': 0.8147, 'contexts': 8},
    {'exp': 10, 'faith_avg': 0.7576, 'relev_avg': 0.7244, 'contexts': 6},
    {'exp': 11, 'faith_avg': 0.9091, 'relev_avg': 0.7851, 'contexts': 6},
    {'exp': 12, 'faith_avg': 0.8750, 'relev_avg': 0.8364, 'contexts': 11},
    {'exp': 13, 'faith_avg': 0.7353, 'relev_avg': 0.7915, 'contexts': 9},
    {'exp': 14, 'faith_avg': 0.6047, 'relev_avg': 0.8828, 'contexts': 13},
    {'exp': 15, 'faith_avg': 0.5294, 'relev_avg': 0.8402, 'contexts': 9},
    {'exp': 16, 'faith_avg': 0.3784, 'relev_avg': 0.7773, 'contexts': 7},
    {'exp': 17, 'faith_avg': 0.6944, 'relev_avg': 0.7847, 'contexts': 8},
    {'exp': 18, 'faith_avg': 0.5652, 'relev_avg': 0.7854, 'contexts': 6},
    {'exp': 19, 'faith_avg': 0.7188, 'relev_avg': 0.7816, 'contexts': 6},
    {'exp': 20, 'faith_avg': (0.6190 + 0.5301)/2, 'relev_avg': 0.7950, 'contexts': 13},
]

# GPT-OSS 20B experiments
experiments_20b = [
    {'exp': 1, 'faith_avg': 0.8750, 'relev_avg': 0.7535, 'contexts': 12},
    {'exp': 2, 'faith_avg': 0.4412, 'relev_avg': 0.7664, 'contexts': 7},
    {'exp': 3, 'faith_avg': 0.9111, 'relev_avg': 0.8484, 'contexts': 14},
    {'exp': 4, 'faith_avg': (0.2857 + 0.2143)/2, 'relev_avg': 0.7808, 'contexts': 10},
    {'exp': 5, 'faith_avg': 0.3158, 'relev_avg': 0.7903, 'contexts': 12},
    {'exp': 6, 'faith_avg': 0.3810, 'relev_avg': 0.7114, 'contexts': 18},
    {'exp': 7, 'faith_avg': 0.4815, 'relev_avg': 0.8403, 'contexts': 6},
    {'exp': 8, 'faith_avg': 0.2500, 'relev_avg': 0.8531, 'contexts': 16},
    {'exp': 9, 'faith_avg': 0.8696, 'relev_avg': 0.7748, 'contexts': 16},
    {'exp': 10, 'faith_avg': (0.6111 + 0.6667)/2, 'relev_avg': 0.7359, 'contexts': 15},
    {'exp': 11, 'faith_avg': 0.8889, 'relev_avg': 0.6959, 'contexts': 12},
    {'exp': 12, 'faith_avg': (0.2727 + 0.1818)/2, 'relev_avg': 0.8529, 'contexts': 17},
    {'exp': 13, 'faith_avg': 0.5417, 'relev_avg': 0.8533, 'contexts': 17},
    {'exp': 14, 'faith_avg': (0.5000 + 0.5294)/2, 'relev_avg': 0.8745, 'contexts': 13},
    {'exp': 15, 'faith_avg': (0.4643 + 0.5000)/2, 'relev_avg': 0.8104, 'contexts': 14},
    {'exp': 16, 'faith_avg': (0.3684 + 0.4211)/2, 'relev_avg': 0.7307, 'contexts': 15},
    {'exp': 17, 'faith_avg': 0.5417, 'relev_avg': 0.8141, 'contexts': 9},
    {'exp': 18, 'faith_avg': 0.6800, 'relev_avg': 0.8057, 'contexts': 14},
    {'exp': 19, 'faith_avg': (0.5769 + 0.6154)/2, 'relev_avg': 0.8439, 'contexts': 11},
    {'exp': 20, 'faith_avg': 0.7812, 'relev_avg': 0.7580, 'contexts': 20},
]

# GPT-4.1 Mini experiments (excluding experiment 10 which had partial results)
experiments_4_1_mini = [
    {'exp': 1, 'faith_avg': 0.6429, 'relev_avg': 0.8004, 'contexts': 8},
    {'exp': 2, 'faith_avg': (0.5854 + 0.5882)/2, 'relev_avg': 0.8525, 'contexts': 12},
    {'exp': 3, 'faith_avg': 0.6471, 'relev_avg': 0.8421, 'contexts': 20},
    {'exp': 4, 'faith_avg': (0.7037 + 0.6897)/2, 'relev_avg': 0.8181, 'contexts': 17},
    {'exp': 5, 'faith_avg': (0.4643 + 0.4667)/2, 'relev_avg': 0.8340, 'contexts': 20},
    {'exp': 6, 'faith_avg': 0.7368, 'relev_avg': 0.8152, 'contexts': 20},
    {'exp': 7, 'faith_avg': 0.4118, 'relev_avg': 0.8663, 'contexts': 11},
    {'exp': 8, 'faith_avg': 0.3750, 'relev_avg': 0.8212, 'contexts': 20},
    {'exp': 9, 'faith_avg': 0.7188, 'relev_avg': 0.8463, 'contexts': 20},
    # Experiment 10 excluded - partial results
    {'exp': 11, 'faith_avg': 0.7143, 'relev_avg': 0.8260, 'contexts': 18},
    {'exp': 12, 'faith_avg': (0.5556 + 0.4706)/2, 'relev_avg': 0.8023, 'contexts': 8},
    {'exp': 13, 'faith_avg': 0.3750, 'relev_avg': 0.8714, 'contexts': 20},
    {'exp': 14, 'faith_avg': 0.8966, 'relev_avg': 0.8253, 'contexts': 20},
    {'exp': 15, 'faith_avg': 0.2381, 'relev_avg': 0.8589, 'contexts': 20},
    {'exp': 16, 'faith_avg': 0.4643, 'relev_avg': 0.8004, 'contexts': 20},
    {'exp': 17, 'faith_avg': 0.5741, 'relev_avg': 0.8335, 'contexts': 20},
    {'exp': 18, 'faith_avg': 0.3684, 'relev_avg': 0.8340, 'contexts': 16},
    {'exp': 19, 'faith_avg': 0.5926, 'relev_avg': 0.8731, 'contexts': 20},
    {'exp': 20, 'faith_avg': 0.6957, 'relev_avg': 0.8349, 'contexts': 20},
]

# Calculate overall statistics
def calc_stats(experiments):
    faithfulness = [exp['faith_avg'] for exp in experiments]
    relevancy = [exp['relev_avg'] for exp in experiments]
    contexts = [exp['contexts'] for exp in experiments]
    overall = [(f + r) / 2 for f, r in zip(faithfulness, relevancy)]
    
    return {
        'faithfulness': faithfulness,
        'relevancy': relevancy,
        'overall': overall,
        'contexts': contexts,
        'avg_faith': np.mean(faithfulness),
        'avg_relev': np.mean(relevancy),
        'avg_overall': np.mean(overall),
        'avg_contexts': np.mean(contexts)
    }

stats_120b = calc_stats(experiments_120b)
stats_20b = calc_stats(experiments_20b)
stats_4_1_mini = calc_stats(experiments_4_1_mini)

# Model labels
models = ['gpt_oss_120b', 'gpt_oss_20b', 'gpt_4_1_mini']
model_labels = ['GPT-OSS 120B', 'GPT-OSS 20B', 'GPT-4.1 Mini']

# Summary statistics
faithfulness = [stats_120b['avg_faith'], stats_20b['avg_faith'], stats_4_1_mini['avg_faith']]
relevancy = [stats_120b['avg_relev'], stats_20b['avg_relev'], stats_4_1_mini['avg_relev']]
overall = [stats_120b['avg_overall'], stats_20b['avg_overall'], stats_4_1_mini['avg_overall']]
avg_contexts = [stats_120b['avg_contexts'], stats_20b['avg_contexts'], stats_4_1_mini['avg_contexts']]

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))
colors_scatter = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# 1. Bar chart comparing average metrics
ax1 = plt.subplot(2, 3, 1)
x = np.arange(len(models))
width = 0.25

bars1 = ax1.bar(x - width, faithfulness, width, label='Faithfulness', alpha=0.8, color='#FF6B6B')
bars2 = ax1.bar(x, relevancy, width, label='Relevancy', alpha=0.8, color='#4ECDC4')
bars3 = ax1.bar(x + width, overall, width, label='Overall', alpha=0.8, color='#45B7D1')

ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Average Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(model_labels, rotation=15, ha='right')
ax1.legend()
ax1.set_ylim([0, 1])
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 2. Radar chart
ax2 = plt.subplot(2, 3, 2, projection='polar')
categories = ['Faithfulness', 'Relevancy', 'Overall', 'Efficiency\n(inv. contexts)']
N = len(categories)

# Normalize contexts (inverse - lower is better, scale to 0-1)
max_contexts = max(avg_contexts)
efficiency = [(max_contexts - c) / max_contexts for c in avg_contexts]

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

for i, (model, label) in enumerate(zip(models, model_labels)):
    values = [faithfulness[i], relevancy[i], overall[i], efficiency[i]]
    values += values[:1]
    ax2.plot(angles, values, 'o-', linewidth=2, label=label, color=colors_scatter[i])
    ax2.fill(angles, values, alpha=0.15, color=colors_scatter[i])

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories, size=10)
ax2.set_ylim(0, 1)
ax2.set_title('Multi-Dimensional Performance', fontsize=14, fontweight='bold', pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax2.grid(True)

# 3. Box plot for faithfulness distribution
ax3 = plt.subplot(2, 3, 3)
faithfulness_data = [
    stats_120b['faithfulness'],
    stats_20b['faithfulness'],
    stats_4_1_mini['faithfulness']
]

bp = ax3.boxplot(faithfulness_data, tick_labels=model_labels, patch_artist=True,
                 showmeans=True, meanline=True)

for patch, color in zip(bp['boxes'], colors_scatter):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax3.set_ylabel('Faithfulness Score', fontsize=12, fontweight='bold')
ax3.set_title('Faithfulness Distribution Across Experiments', fontsize=14, fontweight='bold')
ax3.tick_params(axis='x', rotation=15)
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim([0, 1])

# 4. Scatter plot: Faithfulness vs Relevancy
ax4 = plt.subplot(2, 3, 4)
markers = ['o', 's', '^']

for i, (stat, label, color, marker) in enumerate(zip(
    [stats_120b, stats_20b, stats_4_1_mini],
    model_labels, colors_scatter, markers
)):
    ax4.scatter(stat['faithfulness'], stat['relevancy'], label=label, alpha=0.6, s=100,
               color=color, marker=marker, edgecolors='black', linewidth=0.5)

ax4.set_xlabel('Faithfulness', fontsize=12, fontweight='bold')
ax4.set_ylabel('Relevancy', fontsize=12, fontweight='bold')
ax4.set_title('Faithfulness vs Relevancy (All Experiments)', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim([0, 1])
ax4.set_ylim([0.65, 0.92])

# Add quadrant lines
ax4.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
ax4.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)

# 5. Average contexts comparison
ax5 = plt.subplot(2, 3, 5)
bars = ax5.bar(model_labels, avg_contexts, color=colors_scatter, alpha=0.7, edgecolor='black')
ax5.set_ylabel('Average Number of Contexts', fontsize=12, fontweight='bold')
ax5.set_title('Context Efficiency (Lower is Better)', fontsize=14, fontweight='bold')
ax5.tick_params(axis='x', rotation=15)
ax5.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 6. Performance ranking table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Create ranking data
metrics_ranking = {
    'Faithfulness': sorted(zip(model_labels, faithfulness), key=lambda x: x[1], reverse=True),
    'Relevancy': sorted(zip(model_labels, relevancy), key=lambda x: x[1], reverse=True),
    'Overall': sorted(zip(model_labels, overall), key=lambda x: x[1], reverse=True),
    'Efficiency': sorted(zip(model_labels, avg_contexts), key=lambda x: x[1])
}

table_data = []
headers = ['Metric', '1st', '2nd', '3rd']

for metric, rankings in metrics_ranking.items():
    row = [metric]
    for model, score in rankings:
        if metric == 'Efficiency':
            row.append(f'{model}\n({score:.1f} ctx)')
        else:
            row.append(f'{model}\n({score:.4f})')
    table_data.append(row)

table = ax6.table(cellText=table_data, colLabels=headers,
                  cellLoc='center', loc='center',
                  bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Color code the cells
for i in range(len(table_data)):
    for j in range(4):
        cell = table[(i+1, j)]
        if j == 0:
            cell.set_facecolor('#E8E8E8')
            cell.set_text_props(weight='bold')
        elif j == 1:
            cell.set_facecolor('#FFD700')
        elif j == 2:
            cell.set_facecolor('#C0C0C0')
        elif j == 3:
            cell.set_facecolor('#CD7F32')

# Header styling
for j in range(4):
    cell = table[(0, j)]
    cell.set_facecolor('#4A4A4A')
    cell.set_text_props(weight='bold', color='white')

ax6.set_title('Performance Rankings', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('rag_evaluation_new_visualization.png', dpi=300, bbox_inches='tight')
print("Main visualization saved!")

# Create detailed analysis figure
fig2, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Line plot showing experiment progression
ax1 = axes[0, 0]
experiment_nums = list(range(1, 21))

# Plot for 120B
ax1.plot(experiment_nums, stats_120b['overall'], marker='o', label=model_labels[0], 
        linewidth=2, color=colors_scatter[0], markersize=6)
# Plot for 20B
ax1.plot(experiment_nums, stats_20b['overall'], marker='s', label=model_labels[1], 
        linewidth=2, color=colors_scatter[1], markersize=6)
# Plot for 4.1 mini (adjust for missing experiment 10)
exp_nums_mini = [i for i in range(1, 21) if i != 10]
ax1.plot(exp_nums_mini, stats_4_1_mini['overall'], marker='^', label=model_labels[2], 
        linewidth=2, color=colors_scatter[2], markersize=6)

ax1.set_xlabel('Experiment Number', fontsize=12, fontweight='bold')
ax1.set_ylabel('Overall Score', fontsize=12, fontweight='bold')
ax1.set_title('Performance Across Experiments', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.4, 1.0])

# 2. Violin plot for relevancy distribution
ax2 = axes[0, 1]
relevancy_data = [
    stats_120b['relevancy'],
    stats_20b['relevancy'],
    stats_4_1_mini['relevancy']
]

parts = ax2.violinplot(relevancy_data, positions=[1, 2, 3], showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors_scatter[i])
    pc.set_alpha(0.6)

ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels(model_labels, rotation=15, ha='right')
ax2.set_ylabel('Relevancy Score', fontsize=12, fontweight='bold')
ax2.set_title('Relevancy Distribution (Violin Plot)', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 3. Histogram comparison
ax3 = axes[1, 0]
bins = np.linspace(0, 1, 20)
for i, (stat, label, color) in enumerate(zip(
    [stats_120b, stats_20b, stats_4_1_mini],
    model_labels, colors_scatter
)):
    ax3.hist(stat['faithfulness'], bins=bins, alpha=0.5, label=label, 
            color=color, edgecolor='black')

ax3.set_xlabel('Faithfulness Score', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax3.set_title('Faithfulness Score Distribution', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. Heatmap of correlation
ax4 = axes[1, 1]
summary_data = np.array([
    faithfulness,
    relevancy,
    overall,
    [100/c for c in avg_contexts]  # Inverse for "efficiency"
])

im = ax4.imshow(summary_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

ax4.set_xticks(np.arange(len(model_labels)))
ax4.set_yticks(np.arange(4))
ax4.set_xticklabels(model_labels)
ax4.set_yticklabels(['Faithfulness', 'Relevancy', 'Overall', 'Efficiency*'])
ax4.set_title('Performance Heatmap (* normalized)', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(4):
    for j in range(3):
        text = ax4.text(j, i, f'{summary_data[i, j]:.3f}',
                       ha="center", va="center", color="black", fontweight='bold')

plt.colorbar(im, ax=ax4, label='Score')

plt.tight_layout()
plt.savefig('rag_evaluation_new_detailed.png', dpi=300, bbox_inches='tight')
print("Detailed visualization saved!")

# Print summary statistics
print("\n" + "="*80)
print("UPDATED EVALUATION SUMMARY")
print("="*80)
print(f"\n{'Model':<20} {'Faith.':<12} {'Relev.':<12} {'Overall':<12} {'Contexts':<10}")
print("-"*80)
for i, model in enumerate(model_labels):
    print(f"{model:<20} {faithfulness[i]:<12.4f} {relevancy[i]:<12.4f} {overall[i]:<12.4f} {avg_contexts[i]:<10.1f}")
print("="*80)

print("\nExperiment Counts:")
print(f"  GPT-OSS 120B: {len(experiments_120b)} experiments")
print(f"  GPT-OSS 20B:  {len(experiments_20b)} experiments")
print(f"  GPT-4.1 Mini: {len(experiments_4_1_mini)} experiments (exp 10 excluded)")

print("\nKey Insights:")
print(f"• GPT-OSS 120B: Best overall ({overall[0]:.4f}), best faithfulness ({faithfulness[0]:.4f}), most efficient ({avg_contexts[0]:.1f} contexts)")
print(f"• GPT-OSS 20B:  Lowest overall ({overall[1]:.4f}), lowest faithfulness ({faithfulness[1]:.4f})")
print(f"• GPT-4.1 Mini: Best relevancy ({relevancy[2]:.4f}), but least efficient ({avg_contexts[2]:.1f} contexts)")
print("="*80)