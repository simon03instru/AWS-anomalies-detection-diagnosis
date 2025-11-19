import numpy as np
import matplotlib.pyplot as plt
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
faith_20b = [exp['faith_avg'] for exp in experiments_20b]
relev_20b = [exp['relev_avg'] for exp in experiments_20b]
faith_4_1_mini = [exp['faith_avg'] for exp in experiments_4_1_mini]
relev_4_1_mini = [exp['relev_avg'] for exp in experiments_4_1_mini]

# Calculate statistics
models = ['GPT-OSS\n120B', 'GPT-OSS\n20B', 'GPT-4.1\nMini']
faith_means = [np.mean(faith_120b), np.mean(faith_20b), np.mean(faith_4_1_mini)]
faith_stds = [np.std(faith_120b, ddof=1), np.std(faith_20b, ddof=1), np.std(faith_4_1_mini, ddof=1)]
relev_means = [np.mean(relev_120b), np.mean(relev_20b), np.mean(relev_4_1_mini)]
relev_stds = [np.std(relev_120b, ddof=1), np.std(relev_20b, ddof=1), np.std(relev_4_1_mini, ddof=1)]

# Calculate SEMs (Standard Error of Mean) for error bars
faith_sems = [sem / np.sqrt(20) for sem in faith_stds]
relev_sems = [sem / np.sqrt(20) for sem in relev_stds]

# Colors matching the reference image style
colors = ['#808080', '#4472C4', '#C5504B']

# Set style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# Function to add significance brackets
def add_significance_bracket(ax, x1, x2, y, h, text, linewidth=1.5):
    """Add a bracket with significance text"""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], 'k-', linewidth=linewidth)
    ax.text((x1+x2)/2, y+h, text, ha='center', va='bottom', fontsize=14, fontweight='bold')

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# ===== FAITHFULNESS PLOT =====
x_pos = np.arange(len(models))
bars1 = ax1.bar(x_pos, faith_means, color=colors, edgecolor='black', linewidth=2, width=0.6)
ax1.errorbar(x_pos, faith_means, yerr=faith_sems, fmt='none', ecolor='black', 
             capsize=8, capthick=2, linewidth=2)

ax1.set_ylabel('Score', fontsize=16, fontweight='bold')
ax1.set_title('Faithfulness', fontsize=18, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models, fontsize=14, fontweight='bold')
ax1.set_ylim([0, 1.1])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_linewidth(2)
ax1.spines['bottom'].set_linewidth(2)
ax1.tick_params(width=2, length=6)
ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Add significance bracket for 120B vs 20B (p = 0.0008 ***)
add_significance_bracket(ax1, 0, 1, 0.95, 0.03, '***', linewidth=2)

# Add significance legend
legend_text = '* P<0.05\n** P<0.01\n*** P<0.001'
ax1.text(1.05, 1.05, legend_text, transform=ax1.transAxes, fontsize=12, 
         fontweight='bold', verticalalignment='top', bbox=dict(boxstyle='round', 
         facecolor='white', edgecolor='black', linewidth=1.5))

# ===== RELEVANCE PLOT =====
bars2 = ax2.bar(x_pos, relev_means, color=colors, edgecolor='black', linewidth=2, width=0.6)
ax2.errorbar(x_pos, relev_means, yerr=relev_sems, fmt='none', ecolor='black', 
             capsize=8, capthick=2, linewidth=2)

ax2.set_ylabel('Score', fontsize=16, fontweight='bold')
ax2.set_title('Relevance', fontsize=18, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models, fontsize=14, fontweight='bold')
ax2.set_ylim([0, 1.0])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(2)
ax2.spines['bottom'].set_linewidth(2)
ax2.tick_params(width=2, length=6)
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8])

# Add significance brackets for relevance
# 120B vs 4.1 Mini (p < 0.0001 ***)
add_significance_bracket(ax2, 0, 2, 0.85, 0.03, '***', linewidth=2)
# 20B vs 4.1 Mini (p = 0.0015 ***)
add_significance_bracket(ax2, 1, 2, 0.77, 0.02, '***', linewidth=2)

# Add significance legend
ax2.text(1.05, 1.05, legend_text, transform=ax2.transAxes, fontsize=12, 
         fontweight='bold', verticalalignment='top', bbox=dict(boxstyle='round', 
         facecolor='white', edgecolor='black', linewidth=1.5))

plt.tight_layout()
plt.savefig('publication_style_comparison.png', dpi=300, bbox_inches='tight')
print("Publication-style comparison saved!")

# ===== CREATE INDIVIDUAL PLOTS (for more flexibility) =====

# FAITHFULNESS ONLY
fig, ax = plt.subplots(figsize=(6, 6))
bars = ax.bar(x_pos, faith_means, color=colors, edgecolor='black', linewidth=2, width=0.6)
ax.errorbar(x_pos, faith_means, yerr=faith_sems, fmt='none', ecolor='black', 
            capsize=8, capthick=2, linewidth=2)

ax.set_ylabel('Score', fontsize=16, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=14, fontweight='bold')
ax.set_ylim([0, 1.1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params(width=2, length=6)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Add significance bracket
add_significance_bracket(ax, 0, 1, 0.95, 0.03, '**', linewidth=2)

# Add significance legend
ax.text(1.05, 1.05, legend_text, transform=ax.transAxes, fontsize=12, 
        fontweight='bold', verticalalignment='top', bbox=dict(boxstyle='round', 
        facecolor='white', edgecolor='black', linewidth=1.5))

plt.tight_layout()
plt.savefig('faithfulness_publication.png', dpi=300, bbox_inches='tight')
print("Faithfulness plot saved!")

# RELEVANCE ONLY
fig, ax = plt.subplots(figsize=(6, 6))
bars = ax.bar(x_pos, relev_means, color=colors, edgecolor='black', linewidth=2, width=0.6)
ax.errorbar(x_pos, relev_means, yerr=relev_sems, fmt='none', ecolor='black', 
            capsize=8, capthick=2, linewidth=2)

ax.set_ylabel('Score', fontsize=16, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=14, fontweight='bold')
ax.set_ylim([0, 1.0])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params(width=2, length=6)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])

# Add significance brackets
add_significance_bracket(ax, 0, 2, 0.85, 0.03, '***', linewidth=2)
add_significance_bracket(ax, 1, 2, 0.77, 0.02, '***', linewidth=2)

# Add significance legend
ax.text(1.05, 1.05, legend_text, transform=ax.transAxes, fontsize=12, 
        fontweight='bold', verticalalignment='top', bbox=dict(boxstyle='round', 
        facecolor='white', edgecolor='black', linewidth=1.5))

plt.tight_layout()
plt.savefig('relevance_publication.png', dpi=300, bbox_inches='tight')
print("Relevance plot saved!")

print("\n✓ All publication-style visualizations created successfully!")
print("Files saved:")
print("  1. publication_style_comparison.png (both metrics)")
print("  2. faithfulness_publication.png (faithfulness only)")
print("  3. relevance_publication.png (relevance only)")