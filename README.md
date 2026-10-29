# PufferPhish

AI-assisted URL risk assessment and phishing detection framework for online gaming and account security.

## Overview

**PufferPhish** is a lightweight, interactive machine-learning-powered security application built with Streamlit. It inspects submitted HTTP(S) URLs to identify potential phishing threats and credential-harvesting pages targeting online services and gaming accounts.

The application features:
- **Lexical & Domain Feature Engineering**: 130+ extracted lexical, domain, and gaming-brand security indicators (typosquatting, suspicious subdomains, suspicious tokens, credential-solicitation keywords).
- **Multi-Model Inference & Ensembling**: Real-time predictions powered by Logistic Regression, Random Forest, Support Vector Machines (SVM), and XGBoost, alongside a weighted probability ensemble.
- **Interactive UI with Dynamic Risk Mascot**: Clear low/medium/high risk tiers, granular confidence scores, explainable signals, and a visual mascot indicating classification state.

> **Note**: PufferPhish performs static URL and domain-level feature analysis without sending HTTP requests or fetching live website content, keeping analysis fast and privacy-safe.

---

## Repository Structure

```text
PufferPhish/
├── data/
│   ├── raw/                  # URL collections (legitimate and phishing)
│   └── processed/            # Train and holdout datasets
├── models/                   # Pre-trained models (.pkl) & training metadata
├── prototype_app.py          # Main Streamlit web application
├── launch_app.py             # Desktop browser launcher script
├── feature_extraction.py     # URL feature extraction engine
├── model_training.py         # Model training pipeline
├── evaluation.py             # Model evaluation utilities
├── shap_analysis.py          # Explainability & feature attribution analysis
├── data_collection.py        # Dataset compilation utilities
├── requirements.txt          # Python dependencies
├── setup.bat                 # Setup script for Windows environments
└── README.md                 # Project documentation
```

---

## Getting Started

### 1. Prerequisites
- Python 3.9+ installed
- Recommended: A Python virtual environment

### 2. Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/Nilabjo21/PufferPhish.git
cd PufferPhish

# (Optional) Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

*(On Windows, you can also double-click `setup.bat` to install dependencies automatically).*

---

## Running the Application

Launch the Streamlit web dashboard:

```bash
streamlit run prototype_app.py
```

Or execute the launcher:

```bash
python launch_app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Pipeline & Architecture

1. **Feature Extraction (`feature_extraction.py`)**:
   Analyzes structural, statistical, and semantic URL characteristics, including entropy, delimiter counts, digit ratios, IP usage, deceptive subdomains, and gaming-brand targeted keywords.
2. **Model Training & Inference (`model_training.py`, `models/`)**:
   Trained across diverse classifiers (Random Forest, XGBoost, SVM, and Logistic Regression) with feature normalization and serialized model weights.
3. **Detection Engine (`prototype_app.py`)**:
   Validates inputs, extracts feature vectors on the fly, computes model probabilities, and presents interpretable security indicators.

---

## Retraining the Models

To re-run the end-to-end data processing and model training pipelines:

```bash
# 1. Prepare/process datasets
python data_collection.py

# 2. Extract features and train models
python model_training.py

# 3. Evaluate models
python evaluation.py
```

---

## License

This project is licensed under the MIT License - feel free to adapt and expand it.
