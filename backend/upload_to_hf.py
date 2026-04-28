"""
upload_to_hf.py — run ONCE locally to push model files to Hugging Face Hub.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login        (or set HF_TOKEN env var)

Usage:
    python backend/upload_to_hf.py --repo YOUR_HF_USERNAME/ev-energy-models

The script uploads everything under backend/models/ and backend/data/ as a
private HuggingFace dataset/model repo that Render can pull from at build time.
"""

import argparse
import os
from huggingface_hub import HfApi, create_repo

parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True,
                    help="HuggingFace repo id, e.g. akhil123/ev-energy-models")
parser.add_argument("--private", action="store_true", default=True,
                    help="Create a private repo (default: True)")
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
DATA_DIR   = os.path.join(SCRIPT_DIR, "data")

api = HfApi()

print(f"Creating repo {args.repo} (private={args.private}) ...")
create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

for root, _, files in os.walk(MODELS_DIR):
    for fname in files:
        local_path  = os.path.join(root, fname)
        path_in_repo = os.path.relpath(local_path, SCRIPT_DIR).replace("\\", "/")
        print(f"  Uploading {path_in_repo} ...")
        api.upload_file(path_or_fileobj=local_path,
                        path_in_repo=path_in_repo,
                        repo_id=args.repo,
                        repo_type="model")

for root, _, files in os.walk(DATA_DIR):
    for fname in files:
        local_path  = os.path.join(root, fname)
        path_in_repo = os.path.relpath(local_path, SCRIPT_DIR).replace("\\", "/")
        print(f"  Uploading {path_in_repo} ...")
        api.upload_file(path_or_fileobj=local_path,
                        path_in_repo=path_in_repo,
                        repo_id=args.repo,
                        repo_type="model")

print()
print("Done! Now set these env vars on Render:")
print(f"  HF_REPO_ID = {args.repo}")
print("  HF_TOKEN   = <your HuggingFace read token>")
