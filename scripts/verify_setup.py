#!/usr/bin/env python3
"""
PhishGuard AI - Quick Setup Verification
==========================================
Run this after pip install -r requirements.txt
to verify everything is correctly installed.

Usage:
    python scripts/verify_setup.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def check(label, fn):
    try:
        fn()
        print(f"  ✅ {label}")
        return True
    except Exception as e:
        print(f"  ❌ {label}: {e}")
        return False

print("\n" + "=" * 55)
print("  PhishGuard AI — Setup Verification")
print("=" * 55)

failures = 0

# Python version
print("\n[1] Python environment")
if sys.version_info < (3, 11):
    print(f"  ⚠️  Python {sys.version_info.major}.{sys.version_info.minor} detected. 3.11+ recommended.")
else:
    print(f"  ✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# Core dependencies
print("\n[2] Core dependencies")
failures += 0 if check("FastAPI", lambda: __import__("fastapi")) else 1
failures += 0 if check("Uvicorn", lambda: __import__("uvicorn")) else 1
failures += 0 if check("Pydantic", lambda: __import__("pydantic")) else 1
failures += 0 if check("Motor (MongoDB async)", lambda: __import__("motor")) else 1
failures += 0 if check("Beanie ODM", lambda: __import__("beanie")) else 1

# ML dependencies
print("\n[3] ML dependencies")
failures += 0 if check("scikit-learn", lambda: __import__("sklearn")) else 1
failures += 0 if check("XGBoost", lambda: __import__("xgboost")) else 1
failures += 0 if check("LightGBM", lambda: __import__("lightgbm")) else 1
failures += 0 if check("pandas", lambda: __import__("pandas")) else 1
failures += 0 if check("numpy", lambda: __import__("numpy")) else 1
failures += 0 if check("joblib", lambda: __import__("joblib")) else 1

# NLP dependencies
print("\n[4] NLP dependencies")
failures += 0 if check("NLTK", lambda: __import__("nltk")) else 1
failures += 0 if check("BeautifulSoup4", lambda: __import__("bs4")) else 1

# NLTK resources
print("\n[5] NLTK resources")
import nltk
resources = [
    ("punkt_tab", "tokenizers/punkt_tab"),
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
]
for name, path in resources:
    try:
        nltk.data.find(path)
        print(f"  ✅ NLTK {name}")
    except LookupError:
        print(f"  ⬇️  Downloading NLTK {name}...")
        nltk.download(name, quiet=True)

# Config
print("\n[6] Configuration")
failures += 0 if check("Settings load", lambda: __import__("config.settings", fromlist=["settings"])) else 1

# Dataset
print("\n[7] Datasets")
dataset_dir = Path("./datasets")
csvs = list(dataset_dir.glob("*.csv")) if dataset_dir.exists() else []
if csvs:
    print(f"  ✅ Found {len(csvs)} dataset files in ./datasets/")
    for f in csvs:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"     - {f.name} ({size_mb:.1f} MB)")
else:
    print("  ⚠️  No CSV datasets found in ./datasets/")
    print("     Copy your CSV files to the datasets/ directory.")

# Model
print("\n[8] Trained model")
model_path = Path("./models/saved/model.joblib")
if model_path.exists():
    print(f"  ✅ Trained model found: {model_path}")
else:
    print("  ℹ️  No trained model yet.")
    print("     Run: python scripts/train.py")

print("\n" + "=" * 55)
if failures == 0:
    print("  🎉 All checks passed! Ready to go.")
    print("\n  Next steps:")
    print("  1. python scripts/train.py   ← Train the model")
    print("  2. python backend/main.py    ← Start the API")
    print("  3. Open http://localhost:8000/api/docs")
else:
    print(f"  ⚠️  {failures} check(s) failed.")
    print("     Run: pip install -r requirements.txt")
print("=" * 55 + "\n")
