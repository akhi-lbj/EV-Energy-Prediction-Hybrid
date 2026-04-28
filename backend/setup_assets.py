"""
setup_assets.py — run once before deploying to copy model files into backend/
Usage:  python backend/setup_assets.py
"""
import shutil, os

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SRC_BASE     = os.path.join(SCRIPT_DIR, "..", "Code", "Stage-2", "ev_sota")

DATA_SRC     = os.path.join(SRC_BASE, "acn_enhanced_final_2019_data.csv")
DATA_DST     = os.path.join(SCRIPT_DIR, "data", "acn_enhanced_final_2019_data.csv")

MODELS_V2_SRC = os.path.join(SRC_BASE, "sota_models_v2")
MODELS_V3_SRC = os.path.join(SRC_BASE, "sota_models_v3")
MODELS_V2_DST = os.path.join(SCRIPT_DIR, "models", "sota_models_v2")
MODELS_V3_DST = os.path.join(SCRIPT_DIR, "models", "sota_models_v3")

def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  OK  {os.path.relpath(src)} -> {os.path.relpath(dst)}")

def copy_dir(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"  OK  {os.path.relpath(src)}/ -> {os.path.relpath(dst)}/")

print("Copying assets into backend/ ...")
copy(DATA_SRC, DATA_DST)
copy_dir(MODELS_V2_SRC, MODELS_V2_DST)
copy_dir(MODELS_V3_SRC, MODELS_V3_DST)
print("Done!")
