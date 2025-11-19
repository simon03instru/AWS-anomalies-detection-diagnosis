import matplotlib.pyplot as plt
import numpy as np

# Data structure: 6 experts × 8 cases
cases = ['Case 1', 'Case 2', 'Case 3', 'Case 4', 'Case 5', 'Case 6', 'Case 7', 'Case 8']

# Each row is one case, each column is one expert's rating
diagnostic_quality = [
    [3, 4, 3, 4, 5, 5],  # Case 1: ratings from 6 experts
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
    Calculate rWG (within-group agreement) for a single case.
    
    Parameters:
    - ratings: list of ratings for one case
    - num_response_options: number of points on Likert scale (default: 5)
    
    Returns:
    - rWG value
    """
    # Expected variance under null hypothesis (uniform distribution)
    # For rectangular distribution: σ²_EU = (A² - 1) / 12
    expected_variance = (num_response_options ** 2 - 1) / 12
    
    # Observed variance
    observed_variance = np.var(ratings, ddof=1)  # Sample variance
    
    # rWG formula: 1 - (observed variance / expected variance)
    rwg = 1 - (observed_variance / expected_variance)
    
    return rwg

def interpret_rwg(rwg):
    """Interpret rWG value and return agreement level"""
    if rwg >= 0.90:
        return 'Excellent'
    elif rwg >= 0.80:
        return 'Good'
    elif rwg >= 0.70:
        return 'Acceptable'
    elif rwg >= 0.50:
        return 'Weak'
    else:
        return 'Poor'

# Create figure
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('tight')
ax.axis('off')

# Prepare table data with statistics
headers = ['Case ID', 'Diagnostic Quality\n(Mean ± SD)', 'Maintenance Quality\n(Mean ± SD)', 
           'Diagnostic\nrWG', 'Maintenance\nrWG', 'Overall\nAgreement']

table_data = []
for i, case in enumerate(cases):
    diag_mean = np.mean(diagnostic_quality[i])
    diag_std = np.std(diagnostic_quality[i], ddof=1)
    maint_mean = np.mean(maintenance_quality[i])
    maint_std = np.std(maintenance_quality[i], ddof=1)
    
    # Calculate rWG for each dimension
    diag_rwg = calculate_rwg(diagnostic_quality[i])
    maint_rwg = calculate_rwg(maintenance_quality[i])
    
    # Overall agreement based on average rWG
    avg_rwg = (diag_rwg + maint_rwg) / 2
    overall_agreement = interpret_rwg(avg_rwg)
    
    row = [
        case,
        f'{diag_mean:.2f} ± {diag_std:.2f}',
        f'{maint_mean:.2f} ± {maint_std:.2f}',
        f'{diag_rwg:.3f}',
        f'{maint_rwg:.3f}',
        f'{overall_agreement}\n(rWG={avg_rwg:.3f})'
    ]
    table_data.append(row)

# Create table
table = ax.table(cellText=table_data, colLabels=headers,
                cellLoc='center', loc='center',
                colWidths=[0.10, 0.18, 0.18, 0.12, 0.12, 0.18])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.8)

# Style header
for i in range(len(headers)):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white', ha='center')

# Style data rows with color coding
for i in range(1, len(table_data) + 1):
    for j in range(len(headers)):
        cell = table[(i, j)]
        
        # Alternate row colors
        if i % 2 == 0:
            cell.set_facecolor('#E7E6E6')
        else:
            cell.set_facecolor('#FFFFFF')
        
        # Highlight rWG values
        if j == 3:  # Diagnostic rWG column
            diag_rwg = calculate_rwg(diagnostic_quality[i-1])
            if diag_rwg >= 0.80:
                cell.set_facecolor('#C6EFCE')  # Green for good
            elif diag_rwg >= 0.70:
                cell.set_facecolor('#FFEB9C')  # Yellow for acceptable
            elif diag_rwg < 0.70:
                cell.set_facecolor('#FFC7CE')  # Red for poor
                
        if j == 4:  # Maintenance rWG column
            maint_rwg = calculate_rwg(maintenance_quality[i-1])
            if maint_rwg >= 0.80:
                cell.set_facecolor('#C6EFCE')  # Green for good
            elif maint_rwg >= 0.70:
                cell.set_facecolor('#FFEB9C')  # Yellow for acceptable
            elif maint_rwg < 0.70:
                cell.set_facecolor('#FFC7CE')  # Red for poor
        
        # Highlight overall agreement level
        if j == 5:  # Overall Agreement column
            agreement_text = table_data[i-1][j].split('\n')[0]  # Get just the text part
            if agreement_text in ['Excellent', 'Good']:
                cell.set_facecolor('#C6EFCE')  # Green
            elif agreement_text == 'Acceptable':
                cell.set_facecolor('#FFEB9C')  # Yellow
            elif agreement_text in ['Weak', 'Poor']:
                cell.set_facecolor('#FFC7CE')  # Red

plt.title('Expert Evaluation Summary with rWG Agreement Analysis\n(N=6 experts per case, 48 total ratings)', 
          fontsize=14, fontweight='bold', pad=20)

# Add legend/note
note_text = ('rWG Interpretation: ≥0.90=Excellent, ≥0.80=Good, ≥0.70=Acceptable, ≥0.50=Weak, <0.50=Poor\n'
             'Color coding: Green=Good/Excellent, Yellow=Acceptable, Red=Weak/Poor')
plt.text(0.5, -0.05, note_text, ha='center', va='top', transform=ax.transAxes,
         fontsize=9, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('expert_evaluation_summary_rwg_table.png', dpi=300, bbox_inches='tight')
print("Summary table with rWG saved")

# Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

diag_rwg_values = [calculate_rwg(case) for case in diagnostic_quality]
maint_rwg_values = [calculate_rwg(case) for case in maintenance_quality]

print(f"\nDiagnostic Quality:")
print(f"  Mean rWG: {np.mean(diag_rwg_values):.3f}")
print(f"  Median rWG: {np.median(diag_rwg_values):.3f}")
print(f"  Min rWG: {np.min(diag_rwg_values):.3f}")
print(f"  Max rWG: {np.max(diag_rwg_values):.3f}")

print(f"\nMaintenance Quality:")
print(f"  Mean rWG: {np.mean(maint_rwg_values):.3f}")
print(f"  Median rWG: {np.median(maint_rwg_values):.3f}")
print(f"  Min rWG: {np.min(maint_rwg_values):.3f}")
print(f"  Max rWG: {np.max(maint_rwg_values):.3f}")

# Count agreement levels
print(f"\nDiagnostic Quality Agreement Distribution:")
for level in ['Excellent', 'Good', 'Acceptable', 'Weak', 'Poor']:
    count = sum(1 for rwg in diag_rwg_values if interpret_rwg(rwg) == level)
    print(f"  {level}: {count}/{len(cases)} ({count/len(cases)*100:.0f}%)")

print(f"\nMaintenance Quality Agreement Distribution:")
for level in ['Excellent', 'Good', 'Acceptable', 'Weak', 'Poor']:
    count = sum(1 for rwg in maint_rwg_values if interpret_rwg(rwg) == level)
    print(f"  {level}: {count}/{len(cases)} ({count/len(cases)*100:.0f}%)")

plt.show()