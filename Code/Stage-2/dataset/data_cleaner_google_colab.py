import pandas as pd
import json
from datetime import datetime
import numpy as np

# ====================== LOAD DATA ======================
df = pd.read_csv('acn_timeseries_jan1_to_dec_31_2019_with_formatted_user_input.csv', low_memory=False)

print(f"Original shape: {df.shape}")

# Drop rows where userInputs is missing or empty
df = df.dropna(subset=['userInputs'])
# Filter out common string representations of empty values
empty_vals = ['', '[]', '{}', 'nan', 'None']
df = df[~df['userInputs'].astype(str).str.strip().isin(empty_vals)]
# Reset index after dropping rows
df = df.reset_index(drop=True)

print(f"Shape after dropping empty userInputs: {df.shape}")

# ====================== PARSE userInputs (KEEP LATEST ONLY) ======================
def parse_latest_user_input(json_str):
    if pd.isna(json_str) or json_str.strip() == '' or json_str.strip() == '[]':
        return {}
    try:
        # Handle both string and already-parsed cases
        data = json.loads(json_str) if isinstance(json_str, str) else json_str
        if not isinstance(data, list):
            data = [data]
        if len(data) == 0:
            return {}
        # Take the LAST (most recent) revision
        latest = data[-1]
        # Ensure it's a dict
        return latest if isinstance(latest, dict) else {}
    except (json.JSONDecodeError, TypeError, Exception):
        return {}

# Apply parser
parsed_series = df['userInputs'].apply(parse_latest_user_input)

# Expand to columns
user_cols = pd.json_normalize(parsed_series)

# Rename to avoid conflicts
user_cols = user_cols.add_prefix('userInput_')

# ====================== COMBINE ======================
final_df = pd.concat([df, user_cols], axis=1)

# Optional: Convert datetime strings
for col in ['userInput_modifiedAt', 'userInput_requestedDeparture']:
    if col in final_df.columns:
        final_df[col] = pd.to_datetime(final_df[col], errors='coerce', utc=True)

# Add revision count for auditing
final_df['userInput_revisionCount'] = parsed_series.apply(lambda x: len(x) if isinstance(x, list) else 0)

print(f"Final shape: {final_df.shape}")
print("New columns added:")
print([c for c in final_df.columns if c.startswith('userInput_')])

# ====================== SAVE ======================
final_df.to_csv('acn_timeseries_cleaned_with_userinputs.csv', index=False)
final_df.to_parquet('acn_timeseries_cleaned_with_userinputs.parquet', index=False)  # faster for later ML

print("✅ Saved cleaned dataset with flattened latest user inputs!")