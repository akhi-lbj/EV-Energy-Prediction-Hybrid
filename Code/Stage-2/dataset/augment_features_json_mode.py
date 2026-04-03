# augment_features_json_mode.py
# Uses response_format=json_object + minimal columns to FORCE pure JSON

import os
import json
import time
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("❌ API_KEY not found in .env")

URL = "https://api.k2think.ai/v1/chat/completions"
INPUT_CSV = "acn_timeseries_cleaned_with_userinputs.csv"
OUTPUT_CSV = "acn_enhanced_2019_final.csv"
BATCH_SIZE = 4          # balanced speed + token safety
TEST_MODE = True

# Key columns only (reduces input tokens dramatically)
RELEVANT_COLS = [
    'kWhDelivered', 'parsed_milesRequested', 'parsed_WhPerMile',
    'parsed_minutesAvailable', 'parsed_kWhRequested', 'revisionCount',
    'connectionTime', 'disconnectTime', 'userID'
]

# ====================== K2 CALL WITH JSON MODE ======================
def call_k2_think(batch_json: str, is_test: bool = False):
    system_prompt = (
        "You are a 2026 EV charging demand expert. "
        "For every row, add exactly these 4 fields:\n"
        "1. urgency_score: integer 0-100\n"
        "2. flexibility_index: float 0.0-1.0\n"
        "3. habit_stability: float 0.0-1.0\n"
        "4. grid_impact_proxy: string 'low'|'medium'|'high'\n"
        "Output ONLY a JSON object with key 'augmented_rows' containing the full array. "
        "No text, no <think>, no explanations, no markdown."
    )

    payload = {
        "model": "MBZUAI-IFM/K2-Think-v2",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": batch_json}
        ],
        "temperature": 0.2,
        "max_tokens": 8000,
        "stream": False,
        "response_format": {"type": "json_object"}   # FORCES pure JSON
    }

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(URL, json=payload, headers=headers, timeout=30)
        print(f"   → Status: {r.status_code}")

        if r.status_code != 200:
            print(f"   ❌ ERROR: {r.text[:1000]}")
            raise Exception(f"HTTP {r.status_code}")

        content = r.json()['choices'][0]['message']['content'].strip()
        data = json.loads(content)
        return data.get("augmented_rows")   # the list we need

    except Exception as e:
        print(f"   ❌ API error: {e}")
        if 'r' in locals():
            print(f"   Raw response preview:\n{r.text[:800]}")
        return None

# ====================== MAIN ======================
print("🚀 Loading dataset...")
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df):,} sessions")

# Use only relevant columns for LLM to keep input short
df_minimal = df[RELEVANT_COLS].copy()

if TEST_MODE:
    print("\n=== RUNNING 1-ROW TEST WITH JSON MODE ===")
    test_batch = df_minimal.head(1).to_json(orient="records", date_format="iso")
    test_result = call_k2_think(test_batch, is_test=True)
    if test_result and isinstance(test_result, list):
        print("✅ TEST PASSED! Pure JSON mode works.")
        TEST_MODE = False
    else:
        print("❌ TEST FAILED - copy the entire console output (especially Raw response) and send it to me.")
        exit()

# ====================== FULL AUGMENTATION ======================
enhanced_rows = []

for i in tqdm(range(0, len(df_minimal), BATCH_SIZE), desc="K2-Think-v2 Feature Generation"):
    batch_minimal = df_minimal.iloc[i:i + BATCH_SIZE]
    batch_json = batch_minimal.to_json(orient="records", date_format="iso")
    
    result = call_k2_think(batch_json)
    
    if result and isinstance(result, list):
        # Merge new features back to original full rows
        batch_full = df.iloc[i:i + BATCH_SIZE].copy().reset_index(drop=True)
        result_df = pd.DataFrame(result)
        for col in ['urgency_score', 'flexibility_index', 'habit_stability', 'grid_impact_proxy']:
            batch_full[col] = result_df[col]
        enhanced_rows.extend(batch_full.to_dict('records'))
    else:
        # Safe fallback
        batch_full = df.iloc[i:i + BATCH_SIZE].copy()
        batch_full['urgency_score'] = None
        batch_full['flexibility_index'] = None
        batch_full['habit_stability'] = None
        batch_full['grid_impact_proxy'] = None
        enhanced_rows.extend(batch_full.to_dict('records'))
    
    # Rate-limit safety
    call_count = (i // BATCH_SIZE) + 1
    if call_count % 30 == 0:
        print(f"   ⏳ Sleeping 62 seconds (call #{call_count})...")
        time.sleep(62)

# ====================== SAVE ======================
enhanced_df = pd.DataFrame(enhanced_rows)
enhanced_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n🎉 SUCCESS! Enhanced dataset saved → {OUTPUT_CSV}")
print(f"Total rows: {len(enhanced_df):,}")
print("New columns added: urgency_score, flexibility_index, habit_stability, grid_impact_proxy")

print("\nPreview:")
print(enhanced_df[['kWhDelivered', 'parsed_milesRequested', 'parsed_minutesAvailable',
                   'revisionCount', 'urgency_score', 'flexibility_index',
                   'habit_stability', 'grid_impact_proxy']].head(10))