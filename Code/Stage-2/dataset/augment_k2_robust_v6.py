# augment_k2_robust_v6.py
import os
import json
import time
import random
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ====================== YOUR 3 K2-Think KEYS (cyclic) ======================
keys_str = os.getenv("K2THINK_API_KEYS", "")
API_KEYS = [k.strip() for k in keys_str.split(",") if k.strip()] if keys_str else [
    os.getenv("K2THINK_API_KEY"),
    os.getenv("K2THINK_API_KEY_2", ""),
    os.getenv("K2THINK_API_KEY_3", "")
]
API_KEYS = [k for k in API_KEYS if k]
if not API_KEYS:
    raise ValueError("❌ No K2-Think API keys found.")

print(f"🔑 Loaded {len(API_KEYS)} keys → cyclic rotation on failure")

URL = "https://api.k2think.ai/v1/chat/completions"
INPUT_CSV = "acn_timeseries_cleaned_with_userinputs.csv"
OUTPUT_CSV = "acn_enhanced_2019_final.csv"
BATCH_SIZE = 1
MAX_RETRIES_PER_KEY = 3

# ====================== HELPERS ======================
def add_user_message(messages, text):
    messages.append({"role": "user", "content": text})

def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": text})

def extract_json_from_content(content: str):
    if '</think>' in content:
        content = content.split('</think>')[-1].strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].strip()
    try:
        return json.loads(content)
    except:
        return None

# ====================== RULE-BASED FALLBACK ======================
def fallback_features(row):
    minutes = float(row.get('parsed_minutesAvailable', 60) or 60)
    kwh_req = float(row.get('parsed_kWhRequested', 10) or 10)
    revisions = int(row.get('revisionCount', 0) or 0)
    urgency = max(0, min(100, int(100 * (1 - minutes / 1440))))
    flex = max(0.0, min(1.0, minutes / 300.0))
    stability = max(0.0, min(1.0, 1.0 - revisions / 5.0))
    impact = 'high' if kwh_req > 30 else 'medium' if kwh_req > 10 else 'low'
    return {
        'urgency_score': urgency,
        'flexibility_index': flex,
        'habit_stability': stability,
        'grid_impact_proxy': impact
    }

# ====================== ROBUST CALL ======================
def call_k2_think(batch_json: str, current_key_idx=0):
    messages = []
    system_prompt = (
        "You are a 2026 EV charging demand expert. "
        "For every row in the JSON array, add exactly these 4 fields:\n"
        "1. urgency_score: integer 0-100\n"
        "2. flexibility_index: float 0.0-1.0\n"
        "3. habit_stability: float 0.0-1.0\n"
        "4. grid_impact_proxy: string 'low'|'medium'|'high'\n"
        "Respond in only JSON format with key 'augmented_rows' containing the array."
    )
    messages.append({"role": "system", "content": system_prompt})
    add_assistant_message(messages, "```json")
    add_user_message(messages, batch_json)

    payload = {
        "model": "MBZUAI-IFM/K2-Think-v2",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 8000,
        "stream": False
    }
    headers = {"accept": "application/json", "Content-Type": "application/json"}

    for attempt in range(MAX_RETRIES_PER_KEY):
        key = API_KEYS[current_key_idx]
        print(f"   🔑 Using Key {current_key_idx+1}/{len(API_KEYS)} ...", end=" ")
        headers["Authorization"] = f"Bearer {key}"

        try:
            r = requests.post(URL, json=payload, headers=headers, timeout=60)
            if r.status_code == 500:
                raise requests.exceptions.HTTPError("500 Server Error")
            r.raise_for_status()
            content = r.json()['choices'][0]['message']['content']
            result = extract_json_from_content(content)
            print("✅ SUCCESS")
            return result, current_key_idx
        except Exception as e:
            print(f"❌ FAILED: {e}")
            current_key_idx = (current_key_idx + 1) % len(API_KEYS)
            time.sleep(1 + random.uniform(0, 1))

    print("   ❌ All keys exhausted → using fallback")
    return None, current_key_idx

# ====================== MAIN (RESUME + ROBUST) ======================
print("🚀 Loading dataset...")
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df):,} sessions")

RELEVANT_COLS = ['kWhDelivered', 'parsed_milesRequested', 'parsed_WhPerMile',
                 'parsed_minutesAvailable', 'parsed_kWhRequested', 'revisionCount',
                 'connectionTime', 'disconnectTime', 'userID']

# Resume
if os.path.exists(OUTPUT_CSV):
    enhanced = pd.read_csv(OUTPUT_CSV)
    start_idx = len(enhanced)
    print(f"✅ Resuming from row {start_idx}")
else:
    start_idx = 0

df_minimal = df[RELEVANT_COLS].copy().iloc[start_idx:]
enhanced_rows = [] if start_idx == 0 else enhanced.to_dict('records')
key_idx = 0

target_cols = ['urgency_score', 'flexibility_index', 'habit_stability', 'grid_impact_proxy']

for i in tqdm(range(0, len(df_minimal), BATCH_SIZE), desc="K2-Think-v2 v6 (Robust)"):
    batch_df = df_minimal.iloc[i:i + BATCH_SIZE]
    batch_json = batch_df.to_json(orient="records", date_format="iso")
    
    result, key_idx = call_k2_think(batch_json, key_idx)
    
    batch_full = df.iloc[start_idx + i : start_idx + i + BATCH_SIZE].copy().reset_index(drop=True)
    
    used_fallback = False
    if result:
        # Try to extract as list of dicts
        if isinstance(result, dict) and "augmented_rows" in result:
            rows = result["augmented_rows"]
        elif isinstance(result, list):
            rows = result
        else:
            rows = None
        
        if rows and isinstance(rows, list) and len(rows) > 0:
            try:
                res_df = pd.DataFrame(rows)
                # Flexible column mapping
                for col in target_cols:
                    if col in res_df.columns:
                        batch_full[col] = res_df[col].values
                    else:
                        # Try common variations
                        alt = col.replace('_score', '').replace('_index', '').replace('_stability', '')
                        if alt in res_df.columns:
                            batch_full[col] = res_df[alt].values
                        else:
                            batch_full[col] = None
                            used_fallback = True
            except:
                used_fallback = True
        else:
            used_fallback = True
    else:
        used_fallback = True

    if used_fallback:
        print("   ⚠️ Using rule-based fallback for this batch")
        for idx, row in batch_full.iterrows():
            feats = fallback_features(row)
            for k, v in feats.items():
                batch_full.loc[idx, k] = v

    enhanced_rows.extend(batch_full.to_dict('records'))
    
    # SAVE AFTER EVERY BATCH
    pd.DataFrame(enhanced_rows).to_csv(OUTPUT_CSV, index=False)
    
    # PRINT JSON
    print(f"\n   📤 Batch {start_idx + i} JSON output:")
    print(json.dumps(result if result else "FALLBACK USED", indent=2))
    print("   💾 CSV updated\n")
    
    time.sleep(0.5)

print(f"\n🎉 AUGMENTATION v6 COMPLETE → {OUTPUT_CSV}")
print(f"Total rows: {len(enhanced_rows):,}")
print("✅ All 4 behavioral features added reliably")