import matplotlib.pyplot as plt
import numpy as np

# Data
stations = ['DIY Pakem', 'DIY Stageof', 'DIY UGM', 'DKI Kebun Bibit', 
            'DKI UI', 'Jabar Cikancung', 'Jabar Cimalaka', 'Jabar Cisurupan', 
            'Jabar Pangalengan', 'Jabar Stageof']

fpr = [0.0951, 0.041, 0.0485, 0.0363, 0.0318, 0.0023, 0.0573, 0.0325, 0.0134, 0.0474]
dr = [1.000, 0.981, 0.9925, 0.9008, 0.9654, 0.9467, 0.9919, 0.9490, 0.8835, 0.9444]
f1 = [0.7776, 0.845, 0.8207, 0.7492, 0.8428, 0.9595, 0.8646, 0.8474, 0.8494, 0.7916]

# Create figure
plt.figure(figsize=(10, 8))

# Create scatter plot with size based on F1-score
sizes = np.array(f1) * 500  # Scale up for visibility

scatter = plt.scatter(fpr, dr, s=sizes, alpha=0.6, c=f1, 
                     cmap='RdYlGn', edgecolors='black', linewidth=1.5)

# Add colorbar for F1-score
cbar = plt.colorbar(scatter)
cbar.set_label('F1-Score', rotation=270, labelpad=20, fontsize=12)

# Add station labels
for i, station in enumerate(stations):
    plt.annotate(station, (fpr[i], dr[i]), 
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, alpha=0.8)

# Add ideal region shading (top-left corner)
plt.axhspan(0.95, 1.0, alpha=0.1, color='green', label='High DR Region')
plt.axvspan(0, 0.05, alpha=0.1, color='green')

# Labels and title
plt.xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
plt.ylabel('Detection Rate (DR)', fontsize=12, fontweight='bold')
plt.title('Detection Rate vs False Positive Rate Trade-off Analysis\n(Bubble size represents F1-Score)', 
         fontsize=14, fontweight='bold')

# Grid
plt.grid(True, alpha=0.3, linestyle='--')

# Set axis limits with some padding
plt.xlim(-0.01, max(fpr) + 0.02)
plt.ylim(min(dr) - 0.02, 1.01)

# Add reference lines
plt.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='DR = 0.95')
plt.axvline(x=0.05, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='FPR = 0.05')

plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()

# Save figure
plt.savefig('tradeoff_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('tradeoff_analysis.pdf', bbox_inches='tight')
plt.show()

# Print summary statistics
print("Summary Statistics:")
print(f"Mean DR: {np.mean(dr):.4f} ± {np.std(dr):.4f}")
print(f"Mean FPR: {np.mean(fpr):.4f} ± {np.std(fpr):.4f}")
print(f"Mean F1: {np.mean(f1):.4f} ± {np.std(f1):.4f}")
print(f"\nBest performing station (highest F1): {stations[np.argmax(f1)]} (F1={max(f1):.4f})")
print(f"Station with lowest FPR: {stations[np.argmin(fpr)]} (FPR={min(fpr):.4f})")
print(f"Station with highest DR: {stations[np.argmax(dr)]} (DR={max(dr):.4f})")