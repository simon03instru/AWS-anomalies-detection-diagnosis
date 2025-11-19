import numpy as np
import pandas as pd

diagnostic_quality = [
    [3, 4, 3, 4, 5, 5],  # Case 1
    [4, 5, 4, 4, 5, 5],  # Case 2
    [4, 4, 4, 5, 5, 5],  # Case 3
    [4, 5, 5, 3, 5, 5],  # Case 4
    [4, 4, 5, 5, 5, 5],  # Case 5
    [4, 4, 5, 5, 5, 5],  # Case 6
    [4, 5, 4, 5, 5, 5],  # Case 7
    [5, 5, 5, 4, 5, 5],  # Case 8
]

maintenance_quality = [
    [5, 4, 4, 1, 5, 4],  # Case 1
    [4, 5, 4, 5, 5, 4],  # Case 2
    [5, 5, 5, 5, 5, 4],  # Case 3
    [5, 5, 4, 5, 5, 4],  # Case 4
    [5, 4, 4, 5, 5, 4],  # Case 5
    [5, 4, 5, 5, 5, 4],  # Case 6
    [5, 5, 5, 5, 5, 4],  # Case 7
    [5, 5, 5, 5, 5, 5],  # Case 8
]

def calculate_rwg(ratings, num_response_options=5):
    """
    Calculate rWG (within-group agreement) for each case.
    
    Parameters:
    - ratings: list of lists, each sublist is ratings for one case
    - num_response_options: number of points on Likert scale (default: 5)
    
    Returns:
    - Dictionary with rWG values and statistics
    """
    
    # Expected variance under null hypothesis (uniform distribution)
    # For rectangular distribution: σ²_EU = (A² - 1) / 12
    # where A is number of response options
    expected_variance = (num_response_options ** 2 - 1) / 12
    
    rwg_values = []
    
    for case_idx, case_ratings in enumerate(ratings, 1):
        # Observed variance
        observed_variance = np.var(case_ratings, ddof=1)  # Sample variance
        
        # rWG formula: 1 - (observed variance / expected variance)
        rwg = 1 - (observed_variance / expected_variance)
        
        # rWG can be negative if disagreement is worse than random
        # Some researchers set floor at 0, others report negative values
        rwg_values.append(rwg)
    
    return rwg_values

def calculate_rwg_j(ratings, num_response_options=5):
    """
    Calculate rWG(j) - multi-item version of rWG.
    This treats multiple cases as multiple items measuring the same construct.
    
    Returns:
    - Single rWG(j) value for overall agreement
    """
    ratings_array = np.array(ratings)
    n_cases = ratings_array.shape[0]
    
    # Expected variance for uniform distribution
    expected_variance = (num_response_options ** 2 - 1) / 12
    
    # Mean observed variance across all cases
    mean_observed_variance = np.mean([np.var(case, ddof=1) for case in ratings])
    
    # rWG(j) formula
    rwg_j = 1 - (mean_observed_variance / expected_variance)
    
    return rwg_j

def interpret_rwg(rwg):
    """Interpret rWG value"""
    if rwg < 0:
        return "Poor (Worse than random)"
    elif rwg < 0.50:
        return "Poor"
    elif rwg < 0.70:
        return "Weak"
    elif rwg < 0.80:
        return "Acceptable"
    elif rwg < 0.90:
        return "Good"
    else:
        return "Excellent"

def analyze_agreement_rwg(ratings, dimension_name, num_options=5):
    """
    Complete rWG analysis for inter-rater agreement
    """
    print("=" * 70)
    print(f"\n{dimension_name.upper()} - rWG Analysis")
    print("-" * 70)
    
    # Calculate rWG for each case
    rwg_values = calculate_rwg(ratings, num_options)
    
    # Calculate overall rWG(j)
    rwg_j = calculate_rwg_j(ratings, num_options)
    
    print("\nCase-by-case rWG values:")
    print("-" * 70)
    for i, (case_ratings, rwg) in enumerate(zip(ratings, rwg_values), 1):
        mean = np.mean(case_ratings)
        std = np.std(case_ratings, ddof=1)
        print(f"Case {i}: {case_ratings}")
        print(f"         Mean={mean:.2f}, SD={std:.2f}, rWG={rwg:.3f} ({interpret_rwg(rwg)})")
    
    # Overall statistics
    mean_rwg = np.mean(rwg_values)
    median_rwg = np.median(rwg_values)
    min_rwg = np.min(rwg_values)
    max_rwg = np.max(rwg_values)
    
    print(f"\n{'Overall Agreement Statistics':^70}")
    print("-" * 70)
    print(f"rWG(j) [Overall Agreement]:     {rwg_j:.3f} ({interpret_rwg(rwg_j)})")
    print(f"Mean rWG across cases:          {mean_rwg:.3f} ({interpret_rwg(mean_rwg)})")
    print(f"Median rWG:                     {median_rwg:.3f}")
    print(f"Min rWG:                        {min_rwg:.3f}")
    print(f"Max rWG:                        {max_rwg:.3f}")
    
    # Count cases by agreement level
    excellent = sum(1 for rwg in rwg_values if rwg >= 0.90)
    good = sum(1 for rwg in rwg_values if 0.80 <= rwg < 0.90)
    acceptable = sum(1 for rwg in rwg_values if 0.70 <= rwg < 0.80)
    weak = sum(1 for rwg in rwg_values if 0.50 <= rwg < 0.70)
    poor = sum(1 for rwg in rwg_values if rwg < 0.50)
    
    n_cases = len(ratings)
    
    print(f"\nAgreement Distribution:")
    print(f"  Excellent (rWG ≥ 0.90):   {excellent}/{n_cases} ({excellent/n_cases*100:.0f}%)")
    print(f"  Good (0.80 ≤ rWG < 0.90): {good}/{n_cases} ({good/n_cases*100:.0f}%)")
    print(f"  Acceptable (0.70-0.80):   {acceptable}/{n_cases} ({acceptable/n_cases*100:.0f}%)")
    print(f"  Weak (0.50-0.70):         {weak}/{n_cases} ({weak/n_cases*100:.0f}%)")
    print(f"  Poor (rWG < 0.50):        {poor}/{n_cases} ({poor/n_cases*100:.0f}%)")
    
    return {
        'rwg_values': rwg_values,
        'rwg_j': rwg_j,
        'mean_rwg': mean_rwg,
        'median_rwg': median_rwg
    }

# Analyze both dimensions
diag_results = analyze_agreement_rwg(diagnostic_quality, "Diagnostic Quality", num_options=5)
maint_results = analyze_agreement_rwg(maintenance_quality, "Maintenance Quality", num_options=5)

# Summary Comparison
print("\n" + "=" * 70)
print(f"\n{'SUMMARY COMPARISON':^70}")
print("-" * 70)
print(f"{'Metric':<35} {'Diagnostic':<17} {'Maintenance':<17}")
print("-" * 70)
print(f"{'rWG(j) Overall Agreement':<35} {diag_results['rwg_j']:<17.3f} {maint_results['rwg_j']:<17.3f}")
print(f"{'Mean rWG':<35} {diag_results['mean_rwg']:<17.3f} {maint_results['mean_rwg']:<17.3f}")
print(f"{'Median rWG':<35} {diag_results['median_rwg']:<17.3f} {maint_results['median_rwg']:<17.3f}")
print(f"{'Interpretation':<35} {interpret_rwg(diag_results['rwg_j']):<17} {interpret_rwg(maint_results['rwg_j']):<17}")

# Identify problematic cases (rWG < 0.70)
print("\n" + "=" * 70)
print(f"\n{'CASES NEEDING ATTENTION (rWG < 0.70)':^70}")
print("-" * 70)

print("\nDiagnostic Quality:")
problem_cases = False
for i, (ratings, rwg) in enumerate(zip(diagnostic_quality, diag_results['rwg_values']), 1):
    if rwg < 0.70:
        print(f"  Case {i}: {ratings} (rWG={rwg:.3f})")
        problem_cases = True
if not problem_cases:
    print("  None - all cases have acceptable agreement (rWG ≥ 0.70)")

print("\nMaintenance Quality:")
problem_cases = False
for i, (ratings, rwg) in enumerate(zip(maintenance_quality, maint_results['rwg_values']), 1):
    if rwg < 0.70:
        print(f"  Case {i}: {ratings} (rWG={rwg:.3f})")
        problem_cases = True
if not problem_cases:
    print("  None - all cases have acceptable agreement (rWG ≥ 0.70)")

# Rater consistency analysis
print("\n" + "=" * 70)
print(f"\n{'RATER CONSISTENCY ANALYSIS':^70}")
print("-" * 70)

def analyze_rater_consistency(ratings, dimension_name):
    ratings_array = np.array(ratings)
    n_raters = ratings_array.shape[1]
    
    print(f"\n{dimension_name}:")
    case_means = ratings_array.mean(axis=1)
    
    for rater_idx in range(n_raters):
        rater_scores = ratings_array[:, rater_idx]
        deviations = rater_scores - case_means
        mean_dev = deviations.mean()
        abs_mean_dev = np.abs(deviations).mean()
        
        if abs_mean_dev < 0.3:
            consistency = "Excellent"
        elif abs_mean_dev < 0.5:
            consistency = "Good"
        elif abs_mean_dev < 0.75:
            consistency = "Moderate"
        else:
            consistency = "Poor - Possible outlier"
        
        print(f"  Rater {rater_idx + 1}: Avg deviation = {mean_dev:+.2f}, "
              f"Avg abs deviation = {abs_mean_dev:.2f} ({consistency})")

analyze_rater_consistency(diagnostic_quality, "Diagnostic Quality")
analyze_rater_consistency(maintenance_quality, "Maintenance Quality")

print("\n" + "=" * 70)
print("\nINTERPRETATION GUIDE")
print("-" * 70)
print("rWG (Within-Group Agreement Index):")
print("  rWG ≥ 0.90: Excellent agreement")
print("  rWG ≥ 0.80: Good agreement")
print("  rWG ≥ 0.70: Acceptable agreement (minimum threshold)")
print("  rWG ≥ 0.50: Weak agreement")
print("  rWG < 0.50: Poor agreement")
print("  rWG < 0.00: Agreement worse than random chance")
print("\nNote: rWG compares observed variance to expected variance")
print("      under uniform random distribution (null hypothesis)")