import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# CORRECTED DATA - Experiment 13 for gpt_oss_120b has 0.7915 relevancy

# GPT-OSS 120B experiments
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
    {'exp': 13, 'faith_avg': 0.7353, 'relev_avg': 0.7915, 'contexts': 9},  # CORRECTED!
    {'exp': 14, 'faith_avg': 0.9375, 'relev_avg': 0.5073, 'contexts': 13},
    {'exp': 15, 'faith_avg': (0.6383 + 0.6122)/2, 'relev_avg': 0.6103, 'contexts': 9},
    {'exp': 16, 'faith_avg': (0.9800 + 0.9787)/2, 'relev_avg': (0.5568 + 0.5497)/2, 'contexts': 7},
    {'exp': 17, 'faith_avg': 0.8125, 'relev_avg': 0.5578, 'contexts': 8},
    {'exp': 18, 'faith_avg': (0.8400 + 0.8519)/2, 'relev_avg': (0.5481 + 0.5482)/2, 'contexts': 6},
    {'exp': 19, 'faith_avg': 0.9000, 'relev_avg': 0.5327, 'contexts': 6},
    {'exp': 20, 'faith_avg': 0.8163, 'relev_avg': 0.6023, 'contexts': 13},
]

# GPT-OSS 20B experiments
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

# GPT-4.1 Mini experiments
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

# Calculate statistics
def calc_stats(experiments):
    faithfulness = [exp['faith_avg'] for exp in experiments]
    relevancy = [exp['relev_avg'] for exp in experiments]
    contexts = [exp['contexts'] for exp in experiments]
    
    return {
        'faithfulness': faithfulness,
        'relevancy': relevancy,
        'contexts': contexts,
        'avg_faith': np.mean(faithfulness),
        'avg_relev': np.mean(relevancy),
        'avg_contexts': np.mean(contexts),
        'std_faith': np.std(faithfulness, ddof=1),
        'std_relev': np.std(relevancy, ddof=1)
    }

stats_120b = calc_stats(experiments_120b)
stats_20b = calc_stats(experiments_20b)
stats_4_1_mini = calc_stats(experiments_4_1_mini)

# Model labels
models = ['gpt_oss_120b', 'gpt_oss_20b', 'gpt_4_1_mini']
model_labels = ['GPT-OSS 120B', 'GPT-OSS 20B', 'GPT-4.1 Mini']

# Summary statistics
faithfulness = [stats_120b['avg_faith'], stats_20b['avg_faith'], stats_4_1_mini['avg_faith']]
faithfulness_std = [stats_120b['std_faith'], stats_20b['std_faith'], stats_4_1_mini['std_faith']]
relevancy = [stats_120b['avg_relev'], stats_20b['avg_relev'], stats_4_1_mini['avg_relev']]
relevancy_std = [stats_120b['std_relev'], stats_20b['std_relev'], stats_4_1_mini['std_relev']]
avg_contexts = [stats_120b['avg_contexts'], stats_20b['avg_contexts'], stats_4_1_mini['avg_contexts']]

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))
colors_scatter = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# 1. Bar chart comparing average metrics (ONLY Faithfulness and Relevancy)
ax1 = plt.subplot(2, 3, 1)
x = np.arange(len(models))
width = 0.35

bars1 = ax1.bar(x - width/2, faithfulness, width, label='Faithfulness', alpha=0.8, color='#FF6B6B')
bars2 = ax1.bar(x + width/2, relevancy, width, label='Relevancy', alpha=0.8, color='#4ECDC4')

ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Average Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(model_labels, rotation=15, ha='right')
ax1.legend()
ax1.set_ylim([0, 1])
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars with std
for bars, vals, stds in [(bars1, faithfulness, faithfulness_std), 
                          (bars2, relevancy, relevancy_std)]:
    for bar, val, std in zip(bars, vals, stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=8)

# 2. Radar chart (ONLY Faithfulness, Relevancy, Efficiency)
ax2 = plt.subplot(2, 3, 2, projection='polar')
categories = ['Faithfulness', 'Relevancy', 'Efficiency\n(inv. contexts)']
N = len(categories)

# Normalize contexts (inverse - lower is better, scale to 0-1)
max_contexts = max(avg_contexts)
efficiency = [(max_contexts - c) / max_contexts for c in avg_contexts]

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

for i, (model, label) in enumerate(zip(models, model_labels)):
    values = [faithfulness[i], relevancy[i], efficiency[i]]
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

bp = ax3.boxplot(faithfulness_data, labels=model_labels, patch_artist=True,
                 showmeans=True, meanline=True)

for patch, color in zip(bp['boxes'], colors_scatter):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax3.set_ylabel('Faithfulness Score', fontsize=12, fontweight='bold')
ax3.set_title('Faithfulness Distribution Across Experiments', fontsize=14, fontweight='bold')
ax3.tick_params(axis='x', rotation=15)
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim([0, 1.05])

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
ax4.set_xlim([0, 1.05])
ax4.set_ylim([0.35, 0.85])

# Add quadrant lines
ax4.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5)
ax4.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5)

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

# 6. Performance ranking table (Updated without Overall)
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Create ranking data
metrics_ranking = {
    'Faithfulness': sorted(zip(model_labels, faithfulness), key=lambda x: x[1], reverse=True),
    'Relevancy': sorted(zip(model_labels, relevancy), key=lambda x: x[1], reverse=True),
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
table.scale(1, 3.0)

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
plt.savefig('ag_evaluation_no_overall.png', dpi=300, bbox_inches='tight')
print("Visualization without overall score saved!")

# Print summary statistics
print("\n" + "="*80)
print("EVALUATION SUMMARY (Faithfulness & Relevancy Only)")
print("="*80)
print(f"\n{'Model':<20} {'Faithfulness':<15} {'Relevancy':<15} {'Contexts':<10}")
print("-"*80)
for i, model in enumerate(model_labels):
    print(f"{model:<20} {faithfulness[i]:<15.4f} {relevancy[i]:<15.4f} {avg_contexts[i]:<10.1f}")
print("="*80)

print("\nStandard Deviations:")
print(f"{'Model':<20} {'Faith. SD':<15} {'Relev. SD':<15}")
print("-"*80)
for i, model in enumerate(model_labels):
    print(f"{model:<20} {faithfulness_std[i]:<15.4f} {relevancy_std[i]:<15.4f}")
print("="*80)

print("\nKey Insights:")
print(f"• GPT-OSS 120B: Best faithfulness ({faithfulness[0]:.4f}), most efficient ({avg_contexts[0]:.1f} contexts)")
print(f"• GPT-OSS 20B:  Moderate performance, medium contexts ({avg_contexts[1]:.1f})")
print(f"• GPT-4.1 Mini: Best relevancy ({relevancy[2]:.4f}), but uses most contexts ({avg_contexts[2]:.1f})")
print("="*80)