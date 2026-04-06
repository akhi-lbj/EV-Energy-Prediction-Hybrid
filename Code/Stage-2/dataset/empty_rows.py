import pandas as pd

# ==================== CONFIGURATION ====================
# Change this to your actual file path if needed
input_file = 'acn_enhanced_2019_final_ordinal.csv'
output_file = 'acn_enhanced_2019_cleaned.csv'

# =======================================================

print("Loading the dataset... This may take a moment as the file is large (~6MB).")

# Load the CSV file
df = pd.read_csv(input_file)

print(f"Original dataset shape: {df.shape} rows and columns")

# Step 1: Check how many missing/empty doneChargingTime values exist
missing_before = df['doneChargingTime'].isna().sum()
empty_before = (df['doneChargingTime'] == '').sum()

print(f"Rows with NaN in doneChargingTime: {missing_before}")
print(f"Rows with empty string in doneChargingTime: {empty_before}")

# Step 2: Remove rows where doneChargingTime is NaN or empty string
df_clean = df[
    df['doneChargingTime'].notna() & 
    (df['doneChargingTime'] != '') &
    (df['doneChargingTime'].str.strip() != '')   # also handle whitespace-only strings
].copy()

print(f"Dataset shape after cleaning: {df_clean.shape}")

# Step 3: Reset index for clean sequential numbering
df_clean = df_clean.reset_index(drop=True)

# Step 4: Save the cleaned dataset
df_clean.to_csv(output_file, index=False)

print(f"\n✅ Cleaning completed successfully!")
print(f"✅ Cleaned file saved as: **{output_file}**")
print(f"✅ Removed {df.shape[0] - df_clean.shape[0]} rows with empty/missing doneChargingTime")

# Optional: Show summary of the cleaned column
print("\nSummary of doneChargingTime in cleaned data:")
print(df_clean['doneChargingTime'].describe())

# Optional: Check first few rows to verify
print("\nFirst 3 rows of cleaned data:")
print(df_clean[['_id', 'doneChargingTime', 'kWhDelivered']].head(3))