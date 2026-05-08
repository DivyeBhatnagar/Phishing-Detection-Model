#!/usr/bin/env python3
"""
PhishGuard AI - Inference Script
==================================
Test the model with a sample email from the command line.

Usage:
    python scripts/predict.py "Congratulations! You won a prize, click here!"
    echo "Dear customer, your invoice is attached." | python scripts/predict.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.detector import PhishingDetectorService


def main():
    if len(sys.argv) > 1:
        email_text = " ".join(sys.argv[1:])
    else:
        print("Reading email from stdin...")
        email_text = sys.stdin.read().strip()

    if not email_text:
        print("Error: No email text provided.")
        sys.exit(1)

    detector = PhishingDetectorService()
    if not detector.load():
        print("ERROR: No trained model found. Run scripts/train.py first.")
        sys.exit(1)

    result = detector.predict(email_text, include_shap=False)

    print("\n" + "─" * 50)
    print(f"  📧 Email Preview: {email_text[:80]}...")
    print("─" * 50)
    print(f"  🎯 Prediction:    {result['prediction'].upper()}")
    print(f"  📊 Confidence:    {result['confidence']:.1f}%")
    print(f"  ⚠️  Risk Level:    {result['risk_level'].upper()}")
    print(f"  🔤 Keywords:      {', '.join(result['phishing_keywords']) or 'None'}")
    print(f"  ⏱️  Time:          {result['processing_time_ms']:.1f}ms")
    print("─" * 50 + "\n")


if __name__ == "__main__":
    main()
