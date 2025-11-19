import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import f_oneway, kruskal, shapiro, levene, tukey_hsd
from scikit_posthocs import posthoc_dunn
import pandas as pd

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
faith_120b = np.array([exp['faith_avg'] for exp in experiments_120b])
relev_120b = np.array([exp['relev_avg'] for exp in experiments_120b])
faith_20b = np.array([exp['faith_avg'] for exp in experiments_20b])
relev_20b = np.array([exp['relev_avg'] for exp in experiments_20b])
faith_4_1_mini = np.array([exp['faith_avg'] for exp in experiments_4_1_mini])
relev_4_1_mini = np.array([exp['relev_avg'] for exp in experiments_4_1_mini])

print("="*80)
print("STATISTICAL ANALYSIS: ANOVA + KRUSKAL-WALLIS APPROACH")
print("="*80)

# ============================================================================
# STEP 1: NORMALITY TESTS (Shapiro-Wilk)
# ============================================================================
print("\n" + "="*80)
print("STEP 1: NORMALITY TESTS (Shapiro-Wilk)")
print("="*80)
print("H0: Data follows normal distribution (p > 0.05 = normal)")
print("-"*80)

normality_results = {}

datasets = {
    'Faithfulness - 120B': faith_120b,
    'Faithfulness - 20B': faith_20b,
    'Faithfulness - 4.1 Mini': faith_4_1_mini,
    'Relevance - 120B': relev_120b,
    'Relevance - 20B': relev_20b,
    'Relevance - 4.1 Mini': relev_4_1_mini,
}

faith_normal = []
relev_normal = []

for name, data in datasets.items():
    stat, p_value = shapiro(data)
    normality_results[name] = p_value
    is_normal = p_value > 0.05
    
    if 'Faithfulness' in name:
        faith_normal.append(is_normal)
    else:
        relev_normal.append(is_normal)
    
    print(f"{name:30s} | W={stat:.4f}, p={p_value:.4f} | {'Normal ✓' if is_normal else 'NOT Normal ✗'}")

# Determine which test to use
use_anova = all(faith_normal)
use_kruskal = not all(relev_normal)

print("\n" + "-"*80)
print("DECISION:")
print(f"  Faithfulness: {'All groups normal' if use_anova else 'Some groups not normal'} → Use {'ANOVA' if use_anova else 'Kruskal-Wallis'}")
print(f"  Relevance: {'All groups normal' if all(relev_normal) else 'Some groups not normal'} → Use {'ANOVA' if all(relev_normal) else 'Kruskal-Wallis'}")

# ============================================================================
# STEP 2: VARIANCE HOMOGENEITY (Levene's Test)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: HOMOGENEITY OF VARIANCE (Levene's Test)")
print("="*80)
print("H0: Variances are equal (p > 0.05 = equal variances)")
print("-"*80)

stat_faith, p_faith = levene(faith_120b, faith_20b, faith_4_1_mini)
print(f"Faithfulness | Levene statistic={stat_faith:.4f}, p={p_faith:.4f} | {'Equal variance ✓' if p_faith > 0.05 else 'Unequal variance ✗'}")

stat_relev, p_relev = levene(relev_120b, relev_20b, relev_4_1_mini)
print(f"Relevance    | Levene statistic={stat_relev:.4f}, p={p_relev:.4f} | {'Equal variance ✓' if p_relev > 0.05 else 'Unequal variance ✗'}")

# ============================================================================
# STEP 3: FAITHFULNESS - ONE-WAY ANOVA + TUKEY HSD
# ============================================================================
print("\n" + "="*80)
print("STEP 3: FAITHFULNESS ANALYSIS (PARAMETRIC)")
print("="*80)

# One-way ANOVA
f_stat, p_anova = f_oneway(faith_120b, faith_20b, faith_4_1_mini)
print(f"\nOne-way ANOVA:")
print(f"  F-statistic = {f_stat:.4f}")
print(f"  p-value = {p_anova:.6f}")
print(f"  Result: {'SIGNIFICANT difference exists ***' if p_anova < 0.001 else 'SIGNIFICANT difference exists **' if p_anova < 0.01 else 'SIGNIFICANT difference exists *' if p_anova < 0.05 else 'NO significant difference'}")

# Tukey HSD post-hoc test
print(f"\nTukey HSD Post-hoc Test (pairwise comparisons):")
tukey_result = tukey_hsd(faith_120b, faith_20b, faith_4_1_mini)

# Get means for reporting
mean_120b_f = np.mean(faith_120b)
mean_20b_f = np.mean(faith_20b)
mean_41_f = np.mean(faith_4_1_mini)
std_120b_f = np.std(faith_120b, ddof=1)
std_20b_f = np.std(faith_20b, ddof=1)
std_41_f = np.std(faith_4_1_mini, ddof=1)

print(f"\n  120B (M={mean_120b_f:.3f}, SD={std_120b_f:.3f}) vs 20B (M={mean_20b_f:.3f}, SD={std_20b_f:.3f}):")
print(f"    p = {tukey_result.pvalue[0, 1]:.6f} {'***' if tukey_result.pvalue[0, 1] < 0.001 else '**' if tukey_result.pvalue[0, 1] < 0.01 else '*' if tukey_result.pvalue[0, 1] < 0.05 else 'ns'}")

print(f"\n  120B (M={mean_120b_f:.3f}, SD={std_120b_f:.3f}) vs 4.1 Mini (M={mean_41_f:.3f}, SD={std_41_f:.3f}):")
print(f"    p = {tukey_result.pvalue[0, 2]:.6f} {'***' if tukey_result.pvalue[0, 2] < 0.001 else '**' if tukey_result.pvalue[0, 2] < 0.01 else '*' if tukey_result.pvalue[0, 2] < 0.05 else 'ns'}")

print(f"\n  20B (M={mean_20b_f:.3f}, SD={std_20b_f:.3f}) vs 4.1 Mini (M={mean_41_f:.3f}, SD={std_41_f:.3f}):")
print(f"    p = {tukey_result.pvalue[1, 2]:.6f} {'***' if tukey_result.pvalue[1, 2] < 0.001 else '**' if tukey_result.pvalue[1, 2] < 0.01 else '*' if tukey_result.pvalue[1, 2] < 0.05 else 'ns'}")

# Store results
faith_results = {
    '120B vs 20B': tukey_result.pvalue[0, 1],
    '120B vs 4.1 Mini': tukey_result.pvalue[0, 2],
    '20B vs 4.1 Mini': tukey_result.pvalue[1, 2]
}

# ============================================================================
# STEP 4: RELEVANCE - KRUSKAL-WALLIS + DUNN'S TEST
# ============================================================================
print("\n" + "="*80)
print("STEP 4: RELEVANCE ANALYSIS (NON-PARAMETRIC)")
print("="*80)

# Kruskal-Wallis test
h_stat, p_kruskal = kruskal(relev_120b, relev_20b, relev_4_1_mini)
print(f"\nKruskal-Wallis H-test:")
print(f"  H-statistic = {h_stat:.4f}")
print(f"  p-value = {p_kruskal:.6f}")
print(f"  Result: {'SIGNIFICANT difference exists ***' if p_kruskal < 0.001 else 'SIGNIFICANT difference exists **' if p_kruskal < 0.01 else 'SIGNIFICANT difference exists *' if p_kruskal < 0.05 else 'NO significant difference'}")

# Dunn's test with Bonferroni correction
print(f"\nDunn's Post-hoc Test (pairwise comparisons with Bonferroni correction):")

# Create dataframe for Dunn's test
data_dict = {
    'value': np.concatenate([relev_120b, relev_20b, relev_4_1_mini]),
    'group': ['120B']*20 + ['20B']*20 + ['4.1 Mini']*20
}
df_relev = pd.DataFrame(data_dict)

dunn_result = posthoc_dunn(df_relev, val_col='value', group_col='group', p_adjust='bonferroni')

# Get means for reporting
mean_120b_r = np.mean(relev_120b)
mean_20b_r = np.mean(relev_20b)
mean_41_r = np.mean(relev_4_1_mini)
std_120b_r = np.std(relev_120b, ddof=1)
std_20b_r = np.std(relev_20b, ddof=1)
std_41_r = np.std(relev_4_1_mini, ddof=1)

print(f"\n  120B (M={mean_120b_r:.3f}, SD={std_120b_r:.3f}) vs 20B (M={mean_20b_r:.3f}, SD={std_20b_r:.3f}):")
print(f"    p = {dunn_result.loc['120B', '20B']:.6f} {'***' if dunn_result.loc['120B', '20B'] < 0.001 else '**' if dunn_result.loc['120B', '20B'] < 0.01 else '*' if dunn_result.loc['120B', '20B'] < 0.05 else 'ns'}")

print(f"\n  120B (M={mean_120b_r:.3f}, SD={std_120b_r:.3f}) vs 4.1 Mini (M={mean_41_r:.3f}, SD={std_41_r:.3f}):")
print(f"    p = {dunn_result.loc['120B', '4.1 Mini']:.6f} {'***' if dunn_result.loc['120B', '4.1 Mini'] < 0.001 else '**' if dunn_result.loc['120B', '4.1 Mini'] < 0.01 else '*' if dunn_result.loc['120B', '4.1 Mini'] < 0.05 else 'ns'}")

print(f"\n  20B (M={mean_20b_r:.3f}, SD={std_20b_r:.3f}) vs 4.1 Mini (M={mean_41_r:.3f}, SD={std_41_r:.3f}):")
print(f"    p = {dunn_result.loc['20B', '4.1 Mini']:.6f} {'***' if dunn_result.loc['20B', '4.1 Mini'] < 0.001 else '**' if dunn_result.loc['20B', '4.1 Mini'] < 0.01 else '*' if dunn_result.loc['20B', '4.1 Mini'] < 0.05 else 'ns'}")

# Store results
relev_results = {
    '120B vs 20B': dunn_result.loc['120B', '20B'],
    '120B vs 4.1 Mini': dunn_result.loc['120B', '4.1 Mini'],
    '20B vs 4.1 Mini': dunn_result.loc['20B', '4.1 Mini']
}

# ============================================================================
# STEP 5: CREATE PUBLICATION-QUALITY VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("STEP 5: CREATING VISUALIZATIONS")
print("="*80)

# Calculate statistics for plotting
models = ['GPT-OSS\n120B', 'GPT-OSS\n20B', 'GPT-4.1\nMini']
faith_means = [mean_120b_f, mean_20b_f, mean_41_f]
faith_stds = [std_120b_f, std_20b_f, std_41_f]
relev_means = [mean_120b_r, mean_20b_r, mean_41_r]
relev_stds = [std_120b_r, std_20b_r, std_41_r]

# Calculate SEMs for error bars
faith_sems = [s / np.sqrt(20) for s in faith_stds]
relev_sems = [s / np.sqrt(20) for s in relev_stds]

# Colors
colors = ['#808080', '#4472C4', '#C5504B']

# Set style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# Function to add significance brackets
def add_significance_bracket(ax, x1, x2, y, h, p_value, linewidth=1.5):
    """Add a bracket with significance text based on actual p-value"""
    if p_value < 0.001:
        text = '***'
    elif p_value < 0.01:
        text = '**'
    elif p_value < 0.05:
        text = '*'
    else:
        text = 'ns'
    
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], 'k-', linewidth=linewidth)
    ax.text((x1+x2)/2, y+h, text, ha='center', va='bottom', fontsize=14, fontweight='bold')

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# ===== FAITHFULNESS PLOT (ANOVA) =====
x_pos = np.arange(len(models))
bars1 = ax1.bar(x_pos, faith_means, color=colors, edgecolor='black', linewidth=2, width=0.6)
ax1.errorbar(x_pos, faith_means, yerr=faith_sems, fmt='none', ecolor='black', 
             capsize=8, capthick=2, linewidth=2)

ax1.set_ylabel('Score', fontsize=16, fontweight='bold')
ax1.set_title('Faithfulness (ANOVA)', fontsize=16, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models, fontsize=14, fontweight='bold')
ax1.set_ylim([0, 1.1])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_linewidth(2)
ax1.spines['bottom'].set_linewidth(2)
ax1.tick_params(width=2, length=6)
ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Add significance brackets for faithfulness
if faith_results['120B vs 20B'] < 0.05:
    add_significance_bracket(ax1, 0, 1, 0.95, 0.03, faith_results['120B vs 20B'], linewidth=2)
if faith_results['20B vs 4.1 Mini'] < 0.05:
    add_significance_bracket(ax1, 1, 2, 0.87, 0.02, faith_results['20B vs 4.1 Mini'], linewidth=2)

# Add test info
ax1.text(0.02, 0.98, f'ANOVA: F={f_stat:.2f}, p={p_anova:.4f}', 
         transform=ax1.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

# Add significance legend
legend_text = '* P<0.05\n** P<0.01\n*** P<0.001'
ax1.text(1.05, 1.05, legend_text, transform=ax1.transAxes, fontsize=12, 
         fontweight='bold', verticalalignment='top', bbox=dict(boxstyle='round', 
         facecolor='white', edgecolor='black', linewidth=1.5))

# ===== RELEVANCE PLOT (KRUSKAL-WALLIS) =====
bars2 = ax2.bar(x_pos, relev_means, color=colors, edgecolor='black', linewidth=2, width=0.6)
ax2.errorbar(x_pos, relev_means, yerr=relev_sems, fmt='none', ecolor='black', 
             capsize=8, capthick=2, linewidth=2)

ax2.set_ylabel('Score', fontsize=16, fontweight='bold')
ax2.set_title('Relevance (Kruskal-Wallis)', fontsize=16, fontweight='bold')
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
if relev_results['120B vs 4.1 Mini'] < 0.05:
    add_significance_bracket(ax2, 0, 2, 0.85, 0.03, relev_results['120B vs 4.1 Mini'], linewidth=2)
if relev_results['20B vs 4.1 Mini'] < 0.05:
    add_significance_bracket(ax2, 1, 2, 0.77, 0.02, relev_results['20B vs 4.1 Mini'], linewidth=2)

# Add test info
ax2.text(0.02, 0.98, f'Kruskal-Wallis: H={h_stat:.2f}, p={p_kruskal:.6f}', 
         transform=ax2.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

# Add significance legend
ax2.text(1.05, 1.05, legend_text, transform=ax2.transAxes, fontsize=12, 
         fontweight='bold', verticalalignment='top', bbox=dict(boxstyle='round', 
         facecolor='white', edgecolor='black', linewidth=1.5))

plt.tight_layout()
plt.savefig('anova_kruskal_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: anova_kruskal_comparison.png")

# ============================================================================
# STEP 6: EXPORT RESULTS
# ============================================================================
print("\n" + "="*80)
print("STEP 6: EXPORTING RESULTS")
print("="*80)

# Create results summary
results_summary = {
    'Metric': ['Faithfulness', 'Faithfulness', 'Faithfulness', 'Relevance', 'Relevance', 'Relevance'],
    'Comparison': ['120B vs 20B', '120B vs 4.1 Mini', '20B vs 4.1 Mini', '120B vs 20B', '120B vs 4.1 Mini', '20B vs 4.1 Mini'],
    'Test': ['Tukey HSD', 'Tukey HSD', 'Tukey HSD', "Dunn's (Bonferroni)", "Dunn's (Bonferroni)", "Dunn's (Bonferroni)"],
    'p_value': [
        faith_results['120B vs 20B'],
        faith_results['120B vs 4.1 Mini'],
        faith_results['20B vs 4.1 Mini'],
        relev_results['120B vs 20B'],
        relev_results['120B vs 4.1 Mini'],
        relev_results['20B vs 4.1 Mini']
    ],
    'Significant': [
        'Yes' if faith_results['120B vs 20B'] < 0.05 else 'No',
        'Yes' if faith_results['120B vs 4.1 Mini'] < 0.05 else 'No',
        'Yes' if faith_results['20B vs 4.1 Mini'] < 0.05 else 'No',
        'Yes' if relev_results['120B vs 20B'] < 0.05 else 'No',
        'Yes' if relev_results['120B vs 4.1 Mini'] < 0.05 else 'No',
        'Yes' if relev_results['20B vs 4.1 Mini'] < 0.05 else 'No'
    ]
}

results_df = pd.DataFrame(results_summary)
results_df.to_csv('anova_kruskal_results.csv', index=False)
print("✓ Saved: anova_kruskal_results.csv")

# Descriptive statistics
desc_stats = {
    'Model': ['GPT-OSS 120B', 'GPT-OSS 20B', 'GPT-4.1 Mini', 'GPT-OSS 120B', 'GPT-OSS 20B', 'GPT-4.1 Mini'],
    'Metric': ['Faithfulness', 'Faithfulness', 'Faithfulness', 'Relevance', 'Relevance', 'Relevance'],
    'Mean': [mean_120b_f, mean_20b_f, mean_41_f, mean_120b_r, mean_20b_r, mean_41_r],
    'SD': [std_120b_f, std_20b_f, std_41_f, std_120b_r, std_20b_r, std_41_r],
    'SEM': [faith_sems[0], faith_sems[1], faith_sems[2], relev_sems[0], relev_sems[1], relev_sems[2]],
    'N': [20, 20, 20, 20, 20, 20]
}

desc_df = pd.DataFrame(desc_stats)
desc_df.to_csv('anova_kruskal_descriptives.csv', index=False)
print("✓ Saved: anova_kruskal_descriptives.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. anova_kruskal_comparison.png - Publication-ready figure")
print("  2. anova_kruskal_results.csv - Statistical test results")
print("  3. anova_kruskal_descriptives.csv - Descriptive statistics")
print("\n" + "="*80)
print("SUMMARY ")
print("="*80)
print("\nFaithfulness (ANOVA + Tukey HSD):")
print(f"  - One-way ANOVA: F(2,57) = {f_stat:.2f}, p = {p_anova:.4f}")
print(f"  - GPT-OSS 120B significantly outperformed GPT-OSS 20B (p = {faith_results['120B vs 20B']:.3f})")

print("\nRelevance (Kruskal-Wallis + Dunn's test):")
print(f"  - Kruskal-Wallis: H(2) = {h_stat:.2f}, p = {p_kruskal:.6f}")
print(f"  - GPT-4.1 Mini significantly outperformed GPT-OSS 120B (p < 0.001)")
print(f"  - GPT-4.1 Mini significantly outperformed GPT-OSS 20B (p = {relev_results['20B vs 4.1 Mini']:.3f})")