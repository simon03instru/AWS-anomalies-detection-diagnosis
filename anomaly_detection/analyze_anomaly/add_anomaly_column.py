import pandas as pd

# Read CSV file
df = pd.read_csv('diy_pakem/test_dataset.csv')

# Add is_anomaly column with all values set to 0
df['is_anomaly'] = 0

# Save to CSV
df.to_csv('diy_pakem/test_dataset.csv', index=False)

print("is_anomaly column added!")
print(df.head())