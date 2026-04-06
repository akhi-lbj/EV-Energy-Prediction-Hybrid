import pandas as pd

# ====================== Load the dataset ======================
file_path = "acn_enhanced_2019_final.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# ====================== Task 2: Transform grid_impact_proxy to Ordinal Values ======================

print("Original grid_impact_proxy value counts:")
print(df['grid_impact_proxy'].value_counts(dropna=False))

# Define the ordinal mapping as specified
ordinal_mapping = {
    'low': 0,
    'medium': 1,
    'high': 2
}

# Apply the mapping
df['grid_impact_proxy'] = df['grid_impact_proxy'].map(ordinal_mapping)

# Verify the transformation
print("\nAfter transformation - grid_impact_proxy value counts:")
print(df['grid_impact_proxy'].value_counts(dropna=False))

# Optional: Check first few rows to confirm
print("\nFirst 5 rows after transformation:")
print(df[['grid_impact_proxy']].head())

# ====================== Save the updated dataset ======================
output_file = "acn_enhanced_2019_final_ordinal.csv"
df.to_csv(output_file, index=False)
print(f"\n✅ Transformation complete! Updated file saved as: {output_file}")