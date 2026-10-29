#!/usr/bin/env python3
"""
Run G65 Phishing Detection Project
This script runs the complete project from the organized folder.
"""

import os
import sys
import subprocess

def main():
    print("=" * 80)
    print("G65 Phishing Detection Project")
    print("=" * 80)
    print("\nAvailable scripts:")
    print("1. data_collection.py - Create datasets")
    print("2. feature_extraction.py - Extract features from URLs")
    print("3. model_training.py - Train ML models")
    print("4. evaluation.py - Evaluate model performance")
    print("5. shap_analysis.py - Generate explainability plots")
    print("6. prototype_app.py - Launch web application")
    print("\nTo run the prototype:")
    print("  streamlit run prototype_app.py")
    print("\nTo install requirements:")
    print("  pip install -r requirements.txt")
    print("\nProject files are organized in:")
    for dir_name in ["data/", "models/", "evaluation_plots/", "evaluation_reports/", "shap_analysis/"]:
        if os.path.exists(dir_name):
            print(f"  • {dir_name}")
    
    print("\nSee implementation_summary.md for detailed project information.")

if __name__ == "__main__":
    main()
