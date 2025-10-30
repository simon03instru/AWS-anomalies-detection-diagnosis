import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare, shapiro
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("PAIRED STATISTICAL ANALYSIS (CORRECTED)")
print("="*80)
print("\nYou're RIGHT! Since each experiment uses the same query across models,")
print("we should use PAIRED tests, not independent tests.")
print("\nThis is more powerful and accounts for query difficulty variation.")
print("="*80)

# Data - matched by experiment number
experiments_120b = [
    {'exp': 1, 'faith': 0.9778, 'relev': 0.8537, 'overall': 0.9158, 'contexts': 8},
    {'exp': 2, 'faith': 0.9434, 'relev': 0.7628, 'overall': 0.8531, 'contexts': 7},
    {'exp': 3, 'faith': 0.5862, 'relev': 0.7682, 'overall': 0.6772, 'contexts': 8},
    {'exp': 4, 'faith': 0.7419, 'relev': 0.8169, 'overall': 0.7794, 'contexts': 5},
    {'exp': 5, 'faith': 0.4338, 'relev': 0.7856, 'overall': 0.6097, 'contexts': 8},
    {'exp': 6, 'faith': 0.8000, 'relev': 0.8920, 'overall': 0.8460, 'contexts': 7},
    {'exp': 7, 'faith': 0.9483, 'relev': 0.8008, 'overall': 0.8746, 'contexts': 7},
    {'exp': 8, 'faith': 0.6786, 'relev': 0.7268, 'overall': 0.7027, 'contexts': 7},
    {'exp': 9, 'faith': 0.7500, 'relev': 0.8147, 'overall': 0.7824, 'contexts': 8},
    {'exp': 10, 'faith': 0.7576, 'relev': 0.7244, 'overall': 0.7410, 'contexts': 6},
    {'exp': 11, 'faith': 0.9091, 'relev': 0.7851, 'overall': 0.8471, 'contexts': 6},
    {'exp': 12, 'faith': 0.8750, 'relev': 0.8364, 'overall': 0.8557, 'contexts': 11},
    {'exp': 13, 'faith': 0.7353, 'relev': 0.7915, 'overall': 0.7634, 'contexts': 9},
    {'exp': 14, 'faith': 0.6047, 'relev': 0.8828, 'overall': 0.7438, 'contexts': 13},
    {'exp': 15, 'faith': 0.5294, 'relev': 0.8402, 'overall': 0.6848, 'contexts': 9},
    {'exp': 16, 'faith': 0.3784, 'relev': 0.7773, 'overall': 0.5779, 'contexts': 7},
    {'exp': 17, 'faith': 0.6944, 'relev': 0.7847, 'overall': 0.7396, 'contexts': 8},
    {'exp': 18, 'faith': 0.5652, 'relev': 0.7854, 'overall': 0.6753, 'contexts': 6},
    {'exp': 19, 'faith': 0.7188, 'relev': 0.7816, 'overall': 0.7502, 'contexts': 6},
    {'exp': 20, 'faith': 0.5746, 'relev': 0.7950, 'overall': 0.6848, 'contexts': 13},
]

experiments_20b = [
    {'exp': 1, 'faith': 0.8750, 'relev': 0.7535, 'overall': 0.8143, 'contexts': 12},
    {'exp': 2, 'faith': 0.4412, 'relev': 0.7664, 'overall': 0.6038, 'contexts': 7},
    {'exp': 3, 'faith': 0.9111, 'relev': 0.8484, 'overall': 0.8798, 'contexts': 14},
    {'exp': 4, 'faith': 0.2500, 'relev': 0.7808, 'overall': 0.5154, 'contexts': 10},
    {'exp': 5, 'faith': 0.3158, 'relev': 0.7903, 'overall': 0.5531, 'contexts': 12},
    {'exp': 6, 'faith': 0.3810, 'relev': 0.7114, 'overall': 0.5462, 'contexts': 18},
    {'exp': 7, 'faith': 0.4815, 'relev': 0.8403, 'overall': 0.6609, 'contexts': 6},
    {'exp': 8, 'faith': 0.2500, 'relev': 0.8531, 'overall': 0.5516, 'contexts': 16},
    {'exp': 9, 'faith': 0.8696, 'relev': 0.7748, 'overall': 0.8222, 'contexts': 16},
    {'exp': 10, 'faith': 0.6389, 'relev': 0.7359, 'overall': 0.6874, 'contexts': 15},
    {'exp': 11, 'faith': 0.8889, 'relev': 0.6959, 'overall': 0.7924, 'contexts': 12},
    {'exp': 12, 'faith': 0.2273, 'relev': 0.8529, 'overall': 0.5401, 'contexts': 17},
    {'exp': 13, 'faith': 0.5417, 'relev': 0.8533, 'overall': 0.6975, 'contexts': 17},
    {'exp': 14, 'faith': 0.5147, 'relev': 0.8745, 'overall': 0.6946, 'contexts': 13},
    {'exp': 15, 'faith': 0.4822, 'relev': 0.8104, 'overall': 0.6463, 'contexts': 14},
    {'exp': 16, 'faith': 0.3948, 'relev': 0.7307, 'overall': 0.5628, 'contexts': 15},
    {'exp': 17, 'faith': 0.5417, 'relev': 0.8141, 'overall': 0.6779, 'contexts': 9},
    {'exp': 18, 'faith': 0.6800, 'relev': 0.8057, 'overall': 0.7429, 'contexts': 14},
    {'exp': 19, 'faith': 0.5962, 'relev': 0.8439, 'overall': 0.7200, 'contexts': 11},
    {'exp': 20, 'faith': 0.7812, 'relev': 0.7580, 'overall': 0.7696, 'contexts': 20},
]

experiments_4_1_mini = [
    {'exp': 1, 'faith': 0.6429, 'relev': 0.8004, 'overall': 0.7217, 'contexts': 8},
    {'exp': 2, 'faith': 0.5868, 'relev': 0.8525, 'overall': 0.7197, 'contexts': 12},
    {'exp': 3, 'faith': 0.6471, 'relev': 0.8421, 'overall': 0.7446, 'contexts': 20},
    {'exp': 4, 'faith': 0.6967, 'relev': 0.8181, 'overall': 0.7574, 'contexts': 17},
    {'exp': 5, 'faith': 0.4655, 'relev': 0.8340, 'overall': 0.6498, 'contexts': 20},
    {'exp': 6, 'faith': 0.7368, 'relev': 0.8152, 'overall': 0.7760, 'contexts': 20},
    {'exp': 7, 'faith': 0.4118, 'relev': 0.8663, 'overall': 0.6391, 'contexts': 11},
    {'exp': 8, 'faith': 0.3750, 'relev': 0.8212, 'overall': 0.5981, 'contexts': 20},
    {'exp': 9, 'faith': 0.7188, 'relev': 0.8463, 'overall': 0.7826, 'contexts': 20},
    {'exp': 10, 'faith': None, 'relev': None, 'overall': None, 'contexts': None},  # Missing
    {'exp': 11, 'faith': 0.7143, 'relev': 0.8260, 'overall': 0.7702, 'contexts': 18},
    {'exp': 12, 'faith': 0.5131, 'relev': 0.8023, 'overall': 0.6577, 'contexts': 8},
    {'exp': 13, 'faith': 0.3750, 'relev': 0.8714, 'overall': 0.6232, 'contexts': 20},
    {'exp': 14, 'faith': 0.8966, 'relev': 0.8253, 'overall': 0.8610, 'contexts': 20},
    {'exp': 15, 'faith': 0.2381, 'relev': 0.8589, 'overall': 0.5485, 'contexts': 20},
    {'exp': 16, 'faith': 0.4643, 'relev': 0.8004, 'overall': 0.6324, 'contexts': 20},
    {'exp': 17, 'faith': 0.5741, 'relev': 0.8335, 'overall': 0.7038, 'contexts': 20},
    {'exp': 18, 'faith': 0.3684, 'relev': 0.8340, 'overall': 0.6012, 'contexts': 16},
    {'exp': 19, 'faith': 0.5926, 'relev': 0.8731, 'overall': 0.7329, 'contexts': 20},
    {'exp': 20, 'faith': 0.6957, 'relev': 0.8349, 'overall': 0.7653, 'contexts': 20},
]

# Create matched pairs (excluding experiment 10 for 4.1 mini)
def create_matched_arrays(metric):
    """Create matched arrays for paired analysis, excluding missing data"""
    data_120b = []
    data_20b = []
    data_4_1 = []
    
    for i in range(20):
        if experiments_4_1_mini[i][metric] is not None:
            data_120b.append(experiments_120b[i][metric])
            data_20b.append(experiments_20b[i][metric])
            data_4_1.append(experiments_4_1_mini[i][metric])
    
    return np.array(data_120b), np.array(data_20b), np.array(data_4_1)

faith_120b, faith_20b, faith_4_1 = create_matched_arrays('faith')
relev_120b, relev_20b, relev_4_1 = create_matched_arrays('relev')
overall_120b, overall_20b, overall_4_1 = create_matched_arrays('overall')
contexts_120b, contexts_20b, contexts_4_1 = create_matched_arrays('contexts')

n_paired = len(faith_120b)
print(f"\nNumber of matched pairs: {n_paired}")

print("\n" + "="*80)
print("1. VISUALIZING PAIRED DIFFERENCES")
print("="*80)

# Create visualization of paired differences
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Faithfulness paired comparison
ax1 = axes[0, 0]
exp_nums = list(range(1, n_paired + 1))
ax1.plot(exp_nums, faith_120b, 'o-', label='GPT-OSS 120B', color='#FF6B6B', linewidth=2, markersize=8)
ax1.plot(exp_nums, faith_20b, 's-', label='GPT-OSS 20B', color='#4ECDC4', linewidth=2, markersize=8)
ax1.plot(exp_nums, faith_4_1, '^-', label='GPT-4.1 Mini', color='#45B7D1', linewidth=2, markersize=8)
ax1.set_xlabel('Experiment Number', fontsize=12, fontweight='bold')
ax1.set_ylabel('Faithfulness Score', fontsize=12, fontweight='bold')
ax1.set_title('Faithfulness: Paired by Experiment', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])

# 2. Differences plot (120B - 4.1 Mini)
ax2 = axes[0, 1]
diff_faith = faith_120b - faith_4_1
diff_overall = overall_120b - overall_4_1
colors = ['green' if d > 0 else 'red' for d in diff_faith]
ax2.bar(exp_nums, diff_faith, color=colors, alpha=0.6, edgecolor='black', label='Faithfulness')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=2)
ax2.axhline(y=np.mean(diff_faith), color='green', linestyle='--', linewidth=2, label=f'Mean diff: {np.mean(diff_faith):.3f}')
ax2.set_xlabel('Experiment Number', fontsize=12, fontweight='bold')
ax2.set_ylabel('Difference (120B - 4.1 Mini)', fontsize=12, fontweight='bold')
ax2.set_title('Faithfulness Difference by Experiment', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Overall paired comparison
ax3 = axes[1, 0]
ax3.plot(exp_nums, overall_120b, 'o-', label='GPT-OSS 120B', color='#FF6B6B', linewidth=2, markersize=8)
ax3.plot(exp_nums, overall_20b, 's-', label='GPT-OSS 20B', color='#4ECDC4', linewidth=2, markersize=8)
ax3.plot(exp_nums, overall_4_1, '^-', label='GPT-4.1 Mini', color='#45B7D1', linewidth=2, markersize=8)
ax3.set_xlabel('Experiment Number', fontsize=12, fontweight='bold')
ax3.set_ylabel('Overall Score', fontsize=12, fontweight='bold')
ax3.set_title('Overall: Paired by Experiment', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0.4, 1.0])

# 4. Win/Loss matrix
ax4 = axes[1, 1]
metrics = ['Faithfulness', 'Relevancy', 'Overall', 'Contexts']
wins_120b_vs_20b = [
    sum(faith_120b > faith_20b),
    sum(relev_120b > relev_20b),
    sum(overall_120b > overall_20b),
    sum(contexts_120b < contexts_20b)  # Lower is better for contexts
]
wins_120b_vs_4_1 = [
    sum(faith_120b > faith_4_1),
    sum(relev_120b > relev_4_1),
    sum(overall_120b > overall_4_1),
    sum(contexts_120b < contexts_4_1)
]

x = np.arange(len(metrics))
width = 0.35
bars1 = ax4.bar(x - width/2, wins_120b_vs_20b, width, label='120B vs 20B', alpha=0.8, color='#FF6B6B')
bars2 = ax4.bar(x + width/2, wins_120b_vs_4_1, width, label='120B vs 4.1 Mini', alpha=0.8, color='#45B7D1')
ax4.axhline(y=n_paired/2, color='black', linestyle='--', linewidth=2, label='Tie line')
ax4.set_ylabel('Number of Wins for GPT-OSS 120B', fontsize=12, fontweight='bold')
ax4.set_title('Head-to-Head Wins per Experiment', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(metrics, rotation=15, ha='right')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)
ax4.set_ylim([0, n_paired])

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}/{n_paired}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('paired_analysis_visualization.png', dpi=300, bbox_inches='tight')
print("\n✅ Paired analysis visualization saved!")

print("\n" + "="*80)
print("2. FRIEDMAN TEST (Non-parametric ANOVA for repeated measures)")
print("="*80)
print("\nTests if there's ANY significant difference among the 3 models")
print("Accounts for the fact that each experiment is the same query across models")

def friedman_test(metric_name, data1, data2, data3):
    print(f"\n{metric_name}:")
    print("-" * 80)
    
    stat, p_value = friedmanchisquare(data1, data2, data3)
    
    result = "SIGNIFICANT ✓" if p_value < 0.05 else "NOT SIGNIFICANT ❌"
    print(f"  Chi-square statistic: {stat:.4f}")
    print(f"  p-value: {p_value:.6f}  {result}")
    
    if p_value < 0.001:
        print(f"  → HIGHLY SIGNIFICANT (p < 0.001) ***")
    elif p_value < 0.01:
        print(f"  → VERY SIGNIFICANT (p < 0.01) **")
    elif p_value < 0.05:
        print(f"  → SIGNIFICANT (p < 0.05) *")
    
    return p_value

p_faith_friedman = friedman_test("FAITHFULNESS", faith_120b, faith_20b, faith_4_1)
p_relev_friedman = friedman_test("RELEVANCY", relev_120b, relev_20b, relev_4_1)
p_overall_friedman = friedman_test("OVERALL", overall_120b, overall_20b, overall_4_1)
p_contexts_friedman = friedman_test("CONTEXTS", contexts_120b, contexts_20b, contexts_4_1)

print("\n" + "="*80)
print("3. WILCOXON SIGNED-RANK TEST (Pairwise comparisons)")
print("="*80)
print("\nPaired test comparing each model pair directly")
print("Using Bonferroni correction: α = 0.05/3 = 0.0167")

def wilcoxon_pairwise(metric_name, data1, data2, data3, labels):
    print(f"\n{metric_name}:")
    print("-" * 80)
    
    comparisons = [
        (data1, data2, f"{labels[0]} vs {labels[1]}"),
        (data1, data3, f"{labels[0]} vs {labels[2]}"),
        (data2, data3, f"{labels[1]} vs {labels[2]}")
    ]
    
    results = []
    for d1, d2, comparison_name in comparisons:
        # Wilcoxon signed-rank test (paired)
        stat, p_value = wilcoxon(d1, d2, alternative='two-sided')
        
        # Calculate effect size (r = Z / sqrt(N))
        # For Wilcoxon, approximate Z from the statistic
        n = len(d1)
        z_approx = (stat - n*(n+1)/4) / np.sqrt(n*(n+1)*(2*n+1)/24)
        effect_size = abs(z_approx) / np.sqrt(n)
        
        # Mean difference
        mean_diff = np.mean(d1 - d2)
        
        # Win rate
        wins = sum(d1 > d2)
        win_rate = wins / n * 100
        
        # Determine significance with Bonferroni correction
        if p_value < 0.0167:
            sig = "SIGNIFICANT ✓"
        elif p_value < 0.05:
            sig = "MARGINAL (~)"
        else:
            sig = "NOT SIGNIFICANT ❌"
        
        # Effect size interpretation
        if effect_size < 0.1:
            effect_interp = "negligible"
        elif effect_size < 0.3:
            effect_interp = "small"
        elif effect_size < 0.5:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        
        print(f"\n  {comparison_name}:")
        print(f"    Mean difference: {mean_diff:+.4f}")
        print(f"    Win rate:        {wins}/{n} ({win_rate:.1f}%)")
        print(f"    p-value:         {p_value:.6f}  {sig}")
        print(f"    Effect size:     {effect_size:.3f} ({effect_interp})")
        
        results.append({
            'comparison': comparison_name,
            'mean_diff': mean_diff,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < 0.0167,
            'win_rate': win_rate
        })
    
    return results

labels = ["GPT-OSS 120B", "GPT-OSS 20B", "GPT-4.1 Mini"]

faith_results = wilcoxon_pairwise("FAITHFULNESS", faith_120b, faith_20b, faith_4_1, labels)
relev_results = wilcoxon_pairwise("RELEVANCY", relev_120b, relev_20b, relev_4_1, labels)
overall_results = wilcoxon_pairwise("OVERALL", overall_120b, overall_20b, overall_4_1, labels)
contexts_results = wilcoxon_pairwise("CONTEXTS", contexts_120b, contexts_20b, contexts_4_1, labels)

print("\n" + "="*80)
print("4. COMPARING UNPAIRED vs PAIRED ANALYSIS")
print("="*80)

print("\nIMPORTANT: Paired analysis is MORE POWERFUL because it:")
print("  1. Controls for query difficulty variation")
print("  2. Each model is tested on the SAME queries")
print("  3. Reduces noise and increases statistical power")

print("\n" + "="*80)
print("5. FINAL CONCLUSIONS (CORRECTED WITH PAIRED TESTS)")
print("="*80)

print("\n📊 STATISTICAL SIGNIFICANCE SUMMARY:")
print("-" * 80)

metrics = [
    ("Faithfulness", p_faith_friedman, faith_results),
    ("Relevancy", p_relev_friedman, relev_results),
    ("Overall", p_overall_friedman, overall_results),
    ("Contexts", p_contexts_friedman, contexts_results)
]

for metric_name, friedman_p, pairwise_results in metrics:
    print(f"\n{metric_name}:")
    if friedman_p < 0.05:
        print(f"  ✓ Overall difference EXISTS (Friedman p={friedman_p:.6f})")
        print(f"  Significant pairwise differences:")
        for result in pairwise_results:
            if result['significant']:
                print(f"    • {result['comparison']}:")
                print(f"      - Mean diff: {result['mean_diff']:+.4f}")
                print(f"      - Win rate: {result['win_rate']:.1f}%")
                print(f"      - p-value: {result['p_value']:.6f}")
                print(f"      - Effect size: {result['effect_size']:.3f}")
    else:
        print(f"  ❌ No overall difference (Friedman p={friedman_p:.6f})")

print("\n" + "="*80)
print("6. FINAL VERDICT WITH PAIRED ANALYSIS")
print("="*80)

print("\n🎯 KEY FINDINGS (CORRECTED):")
print("-" * 80)

# Count significant advantages
sig_advantages_120b = 0
sig_advantages_counts = {
    'faith': 0,
    'relev': 0,
    'overall': 0,
    'contexts': 0
}

# Faithfulness
if faith_results[0]['significant'] or faith_results[1]['significant']:
    if faith_results[1]['significant'] and faith_results[1]['mean_diff'] > 0:
        sig_advantages_120b += 1
        sig_advantages_counts['faith'] = 1
        print("\n1. FAITHFULNESS:")
        print(f"   ✓ GPT-OSS 120B SIGNIFICANTLY BETTER than GPT-4.1 Mini")
        print(f"     - Wins: {faith_results[1]['win_rate']:.1f}% of experiments")
        print(f"     - Mean advantage: +{faith_results[1]['mean_diff']:.4f}")
        print(f"     - p-value: {faith_results[1]['p_value']:.6f}")

# Relevancy
if relev_results[1]['significant']:
    if relev_results[1]['mean_diff'] < 0:
        print("\n2. RELEVANCY:")
        print(f"   ✓ GPT-4.1 Mini SIGNIFICANTLY BETTER than GPT-OSS 120B")
        print(f"     - Wins: {100 - relev_results[1]['win_rate']:.1f}% of experiments")
        print(f"     - Mean advantage: +{-relev_results[1]['mean_diff']:.4f}")
        print(f"     - p-value: {relev_results[1]['p_value']:.6f}")
    else:
        sig_advantages_120b += 1
        sig_advantages_counts['relev'] = 1

# Overall
if overall_results[0]['significant'] or overall_results[1]['significant']:
    if overall_results[1]['significant'] and overall_results[1]['mean_diff'] > 0:
        sig_advantages_120b += 1
        sig_advantages_counts['overall'] = 1
        print("\n3. OVERALL:")
        print(f"   ✓ GPT-OSS 120B SIGNIFICANTLY BETTER than GPT-4.1 Mini")
        print(f"     - Wins: {overall_results[1]['win_rate']:.1f}% of experiments")
        print(f"     - Mean advantage: +{overall_results[1]['mean_diff']:.4f}")
        print(f"     - p-value: {overall_results[1]['p_value']:.6f}")
    elif overall_results[1]['p_value'] < 0.05:
        print("\n3. OVERALL:")
        print(f"   ~ GPT-OSS 120B MARGINALLY BETTER than GPT-4.1 Mini")
        print(f"     - Wins: {overall_results[1]['win_rate']:.1f}% of experiments")
        print(f"     - Mean advantage: +{overall_results[1]['mean_diff']:.4f}")
        print(f"     - p-value: {overall_results[1]['p_value']:.6f} (marginal)")

# Contexts
if contexts_results[0]['significant'] and contexts_results[1]['significant']:
    sig_advantages_120b += 1
    sig_advantages_counts['contexts'] = 1
    print("\n4. EFFICIENCY (Contexts):")
    print(f"   ✓ GPT-OSS 120B SIGNIFICANTLY MORE EFFICIENT")
    print(f"     - vs GPT-4.1 Mini:")
    print(f"       • Wins: {contexts_results[1]['win_rate']:.1f}% of experiments")
    print(f"       • Mean advantage: {contexts_results[1]['mean_diff']:.1f} fewer contexts")
    print(f"       • p-value: {contexts_results[1]['p_value']:.6f}")

print("\n" + "="*80)
print("7. RECOMMENDATION (WITH PAIRED ANALYSIS)")
print("="*80)

print("\n🏆 FINAL VERDICT:")
print("-" * 80)

if sig_advantages_120b >= 2:
    print("\n✅ GPT-OSS 120B IS THE STATISTICALLY SIGNIFICANT WINNER!")
    print("\nWith paired analysis (correct method):")
    print("  • Controls for query difficulty")
    print("  • More powerful statistical tests")
    print("  • Accounts for matched experimental design")
    print(f"\n  Significant advantages: {sig_advantages_120b}/4 metrics")
    print("\n  The performance differences are REAL and NOT due to random chance.")
else:
    print("\n⚠️  RESULTS ARE LESS CLEAR WITH PAIRED ANALYSIS")
    print("\n  Paired tests are more conservative but also more accurate.")
    print("  They account for the fact that the same queries are used across models.")

print("\n" + "="*80)
print("\nComparison saved to: paired_analysis_visualization.png")
print("="*80)