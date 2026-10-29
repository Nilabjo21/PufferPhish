#!/usr/bin/env python3
"""
Streamlit Prototype for G65 Phishing Detection Framework
Interactive web application for phishing URL detection.
"""

import streamlit as st
import os
import html
import pandas as pd
import numpy as np
import pickle
import re
from typing import Dict, List, Tuple, Any
from urllib.parse import urlsplit
import warnings
warnings.filterwarnings('ignore')

# Import feature extractors from our modules
try:
    from feature_extraction import FeaturePipeline
    from feature_extraction import get_numeric_feature_columns
except ImportError:
    st.error("⚠️ Feature extraction module not found. Please run feature extraction first.")
    FeaturePipeline = None
    get_numeric_feature_columns = None

# Page configuration
st.set_page_config(
    page_title="PufferPhish | Link Security",
    page_icon="🐡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom styling
st.markdown("""
<style>
    :root {
        --ink: #f4f7fb;
        --muted: #c3cede;
        --line: #33445c;
        --teal: #59d4cf;
    }
    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Aptos', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 50% -15%, rgba(20, 166, 168, .2), transparent 48%),
            #0b1220;
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] { background: transparent; }
    [data-testid="stMainBlockContainer"] {
        max-width: 820px;
        padding: 3rem 1.5rem 2rem;
    }
    .brand {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: .65rem;
        color: #e3ebf5;
        font-size: .87rem;
        font-weight: 700;
        letter-spacing: .04em;
        margin-bottom: 2.25rem;
    }
    .brand-mark {
        display: inline-grid;
        place-items: center;
        width: 2rem;
        height: 2rem;
        border-radius: .7rem;
        color: white;
        background: linear-gradient(145deg, #14a3a4, #087f8c);
        font-size: 1rem;
    }
    .hero-layout {
        margin: 0 auto 2rem;
        max-width: 680px;
        text-align: center;
    }
    .mascot {
        height: 122px;
        width: 142px;
    }
    .mascot .fish {
        animation: swim 4.2s ease-in-out infinite;
        transform-box: fill-box;
        transform-origin: center;
    }
    .mascot .fish-body {
        transform-box: fill-box;
        transform-origin: center;
        transition: transform .35s ease;
    }
    .mascot .spikes {
        opacity: 0;
        transform: scale(.7);
        transform-box: fill-box;
        transform-origin: center;
        transition: opacity .25s ease, transform .35s ease;
    }
    .mascot.danger .fish-body {
        animation: puff .65s ease-in-out infinite alternate;
    }
    .mascot.danger .spikes { opacity: 1; transform: scale(1); }
    .mascot.danger .body-highlight { stop-color: #ffd18a; }
    .mascot.danger .body-shadow { stop-color: #f05b67; }
    .mascot.danger .fin-highlight { stop-color: #ffad73; }
    .mascot.danger .fin-shadow { stop-color: #c83c58; }
    .mascot.danger .fish-body ellipse {
        stroke: #ffe0a8;
        transition: stroke .3s ease;
    }
    .mascot-widget.danger { border-color: #bd5860; }
    .mascot-widget.danger .mascot-status {
        background: #762f3b;
        color: #ffe0d2;
    }
    .mascot.danger .eyes-calm,
    .mascot.danger .mouth-calm { display: none; }
    .mascot.danger .eyes-alert,
    .mascot.danger .mouth-alert { display: block; }
    .mascot.safe .fish-body { transform: scale(1); }
    .mascot.safe .eyes-calm,
    .mascot.safe .mouth-calm { display: block; }
    .mascot.safe .eyes-alert,
    .mascot.safe .mouth-alert { display: none; }
    .mascot.idle .eyes-calm,
    .mascot.idle .mouth-calm { display: block; }
    .mascot.idle .eyes-alert,
    .mascot.idle .mouth-alert { display: none; }
    @keyframes swim {
        0%, 100% { transform: translate(0, 0) rotate(-2deg); }
        25% { transform: translate(-9px, -7px) rotate(2deg); }
        55% { transform: translate(5px, -11px) rotate(-1deg); }
        78% { transform: translate(10px, -3px) rotate(3deg); }
    }
    @keyframes puff { from { scale: 1; } to { scale: 1.12; } }
    .mascot-widget {
        align-items: center;
        background: rgba(15, 29, 46, .96);
        border: 1px solid #315366;
        border-radius: 1.1rem;
        bottom: 22px;
        box-shadow: 0 12px 36px rgba(0, 0, 0, .48);
        display: flex;
        flex-direction: column;
        gap: .25rem;
        padding: .45rem .7rem .65rem;
        position: fixed;
        right: 22px;
        z-index: 999;
    }
    .mascot-status {
        border-radius: 999px;
        color: #bfe9e6;
        font-size: .65rem;
        font-weight: 800;
        letter-spacing: .08em;
        padding: .27rem .65rem;
        text-align: center;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .mascot-status.idle { background: #193e50; color: #c3f4f0; }
    .mascot-status.safe { background: #174838; color: #b9f3d5; }
    .mascot-status.danger { background: #632a36; color: #ffd0cc; }
    @media (prefers-reduced-motion: reduce) {
        .mascot .fish, .mascot.danger .fish-body { animation: none; }
    }
    .hero { text-align: center; margin: 0 auto 2rem; }
    .eyebrow {
        color: var(--teal);
        font-size: .73rem;
        font-weight: 700;
        letter-spacing: .15em;
        text-transform: uppercase;
        margin-bottom: .75rem;
    }
    .hero h1 {
        color: var(--ink);
        font-family: 'Segoe UI', 'Aptos', sans-serif;
        font-size: clamp(2rem, 6vw, 3.2rem);
        letter-spacing: -.055em;
        line-height: 1.08;
        margin: 0 0 .8rem;
    }
    .hero p {
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.6;
        margin: 0 auto;
        max-width: 560px;
    }
    .url-label {
        color: #e5edf7;
        font-size: .88rem;
        font-weight: 700;
        margin: 0 0 .45rem;
    }
    div[data-testid="stTextInput"] input {
        min-height: 3.2rem;
        border: 1px solid #60718a;
        border-radius: .8rem;
        background: #f8fafc;
        color: #101b2b;
        font-size: .98rem;
        box-shadow: 0 4px 14px rgba(18, 33, 58, .035);
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #007f83;
        box-shadow: 0 0 0 .2rem rgba(0, 127, 131, .23);
    }
    div[data-testid="stTextInput"] input::placeholder { color: #536276; opacity: 1; }
    div.stButton > button {
        min-height: 3.1rem;
        border: 1px solid #39c7c1;
        border-radius: .8rem;
        color: white;
        background: linear-gradient(100deg, #08777d, #006b70);
        font-weight: 700;
        box-shadow: 0 7px 18px rgba(0, 0, 0, .3);
        transition: transform .15s ease, box-shadow .15s ease;
    }
    div.stButton > button:hover {
        color: white;
        border: 1px solid #76e4df;
        background: #075f64;
        box-shadow: 0 10px 22px rgba(0, 0, 0, .38);
        transform: translateY(-1px);
    }
    div.stButton > button:focus { color: white; outline: 3px solid #83e5e0; }
    .hint {
        color: #c1ccda;
        font-size: .79rem;
        margin-top: .45rem;
    }
    .result-card {
        background: #152238;
        border: 1px solid var(--line);
        border-radius: 1rem;
        box-shadow: 0 12px 35px rgba(0, 0, 0, .28);
        margin: 1.7rem 0 1rem;
        padding: 1.4rem 1.5rem;
    }
    .result-card.risk-high { background: #301c27; border-color: #8c4655; }
    .result-card.risk-medium { background: #302719; border-color: #8d6a31; }
    .result-card.risk-low { background: #142c29; border-color: #347767; }
    .result-top {
        align-items: center;
        display: flex;
        gap: .85rem;
        margin-bottom: .65rem;
    }
    .result-icon {
        align-items: center;
        border-radius: .85rem;
        display: flex;
        font-size: 1.25rem;
        height: 2.8rem;
        justify-content: center;
        width: 2.8rem;
    }
    .result-title {
        color: var(--ink);
        font-family: 'Segoe UI', 'Aptos', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
    }
    .result-subtitle { color: #d4deeb; font-size: .86rem; margin-top: .15rem; }
    .risk-high .result-icon { background: #5b2933; color: #ffd3d1; }
    .risk-medium .result-icon { background: #59401f; color: #ffe1a8; }
    .risk-low .result-icon { background: #1d4b40; color: #c4f4df; }
    .result-url {
        background: #0d1728;
        border-radius: .6rem;
        color: #e3ebf5;
        font-family: monospace;
        font-size: .8rem;
        margin-top: 1rem;
        overflow-wrap: anywhere;
        padding: .7rem .8rem;
    }
    .section-title {
        color: var(--ink);
        font-family: 'Segoe UI', 'Aptos', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        margin: 1.6rem 0 .7rem;
    }
    .indicator {
        background: #152238;
        border: 1px solid var(--line);
        border-left: 3px solid #f2bd62;
        border-radius: .65rem;
        color: #e0e8f2;
        font-size: .88rem;
        line-height: 1.5;
        margin: .5rem 0;
        padding: .75rem .9rem;
    }
    .indicator.high { border-left-color: #ff7777; }
    .indicator.info { border-left-color: #59d4cf; }
    .indicator strong { color: var(--ink); }
    .recommendation {
        background: #142d3a;
        border: 1px solid #346173;
        border-radius: .8rem;
        color: #e2f0f4;
        line-height: 1.65;
        margin: .65rem 0 1.5rem;
        padding: 1rem 1.1rem;
    }
    .footer {
        border-top: 1px solid #33445c;
        color: #c0ccda;
        font-size: .78rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
        text-align: center;
    }
    [data-testid="stMetric"] {
        background: #121e31;
        border: 1px solid #33445c;
        border-radius: .7rem;
        padding: .8rem;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p { color: #c9d5e4 !important; }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div { color: #f4f7fb !important; }
    [data-testid="stAlert"] { color: #101b2b; }
    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] { padding: 2rem 1rem 1.5rem; }
        .brand { margin-bottom: 1.7rem; }
        .result-card { padding: 1.1rem; }
        .mascot-widget { bottom: 12px; right: 12px; padding: .25rem .45rem .45rem; }
        .mascot { height: 90px; width: 105px; }
        .mascot-status { font-size: .58rem; }
    }
</style>
""", unsafe_allow_html=True)

class PhishingDetectionApp:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_pipeline = None
        self.feature_names = []
        self.load_error = None
        self.load_resources()

    def load_resources(self):
        """Load models, scalers, and feature pipeline"""
        try:
            # Load models
            model_files = {
                'logistic_regression': './models/logistic_regression.pkl',
                'random_forest': './models/random_forest.pkl',
                'xgboost': './models/xgboost.pkl',
                'fusion': None  # Fusion model is combination of others
            }

            for model_name, model_path in model_files.items():
                if model_path and os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)

            # Load scalers
            scaler_files = {
                'logistic': './models/scaler_logistic.pkl',
                'svm': './models/scaler_svm.pkl'
            }

            for scaler_name, scaler_path in scaler_files.items():
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scalers[scaler_name] = pickle.load(f)

            # Load feature pipeline
            if FeaturePipeline:
                self.feature_pipeline = FeaturePipeline()

            # Load feature names (from training data)
            if os.path.exists('./data/processed/train_dataset.csv'):
                train_df = pd.read_csv('./data/processed/train_dataset.csv')
                self.feature_names = get_numeric_feature_columns(train_df)

        except Exception as e:
            self.load_error = str(e)

    def extract_features(self, url: str) -> Dict[str, float]:
        """Extract features from URL"""
        if self.feature_pipeline:
            try:
                features = self.feature_pipeline.extract_features(url)
                return features
            except Exception as e:
                st.warning(f"Feature extraction failed: {e}")
                return self.get_basic_features(url)
        else:
            return self.get_basic_features(url)

    def get_basic_features(self, url: str) -> Dict[str, float]:
        """Extract basic features when feature pipeline is not available"""
        features = {}

        # Basic URL features
        features['url_length'] = len(url)
        features['has_https'] = 1.0 if url.startswith('https://') else 0.0
        features['num_digits'] = sum(c.isdigit() for c in url)

        # Check for suspicious patterns
        url_lower = url.lower()
        phishing_keywords = ['verify', 'account', 'login', 'secure', 'update', 'suspend']
        gaming_keywords = ['steam', 'game', 'skin', 'trade', 'gaming', 'valve']

        features['phishing_keyword_count'] = sum(1 for keyword in phishing_keywords if keyword in url_lower)
        features['gaming_keyword_count'] = sum(1 for keyword in gaming_keywords if keyword in url_lower)
        features['has_ip_address'] = 1.0 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0.0

        # Fake Steam detection
        fake_steam_patterns = ['steamcommmunity', 'steamcomunnity', 'steam-comunnity']
        features['has_fake_steam'] = 1.0 if any(pattern in url_lower for pattern in fake_steam_patterns) else 0.0

        return features

    def predict_with_model(self, model_name: str, features: Dict[str, float]) -> Tuple[float, float]:
        """Make prediction using a specific model"""
        if model_name not in self.models:
            raise RuntimeError(f"The selected model '{model_name}' is not loaded.")

        model = self.models[model_name]

        # Convert features to array
        if self.feature_names:
            # Align features with training feature order
            X = np.array([features.get(feat, 0.0) for feat in self.feature_names]).reshape(1, -1)
        else:
            # Use available features
            X = np.array(list(features.values())).reshape(1, -1)

        # Scale features if needed
        if model_name in ['logistic_regression']:
            scaler_name = 'logistic'
            if scaler_name in self.scalers:
                X = self.scalers[scaler_name].transform(X)

        if hasattr(model, 'predict_proba'):
            probability = float(model.predict_proba(X)[0, 1])
        else:
            prediction = model.predict(X)[0]
            probability = float(prediction)

        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise RuntimeError(f"The {model_name} model returned an invalid probability.")

        confidence = abs(probability - 0.5) * 2
        return probability, confidence

    def get_fusion_prediction(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Get fusion prediction (weighted ensemble)"""
        predictions = []
        weights = []

        # Logistic Regression (weight: 0.25)
        if 'logistic_regression' in self.models:
            prob_lr, _ = self.predict_with_model('logistic_regression', features)
            predictions.append(prob_lr)
            weights.append(0.25)

        # Random Forest (weight: 0.35)
        if 'random_forest' in self.models:
            prob_rf, _ = self.predict_with_model('random_forest', features)
            predictions.append(prob_rf)
            weights.append(0.35)

        # XGBoost (weight: 0.40)
        if 'xgboost' in self.models:
            prob_xgb, _ = self.predict_with_model('xgboost', features)
            predictions.append(prob_xgb)
            weights.append(0.40)

        if not predictions:
            raise RuntimeError("No trained models are available for the ensemble.")

        # Normalize weights
        weights = np.array(weights) / sum(weights)
        fused_probability = np.dot(predictions, weights)
        confidence = abs(fused_probability - 0.5) * 2

        return float(fused_probability), float(confidence)

    def analyze_risk_indicators(self, url: str, features: Dict[str, float]) -> List[Dict[str, Any]]:
        """Analyze specific risk indicators"""
        indicators = []
        url_lower = url.lower()

        # URL structure indicators
        if features.get('has_ip_address', 0) > 0:
            indicators.append({
                'type': 'warning',
                'title': 'IP Address in URL',
                'description': 'URL contains an IP address instead of a domain name.',
                'risk': 'medium'
            })

        if features.get('url_length', 0) > 100:
            indicators.append({
                'type': 'warning',
                'title': 'Long URL',
                'description': f'URL is {features["url_length"]} characters long.',
                'risk': 'low'
            })

        if features.get('has_fake_steam', 0) > 0:
            indicators.append({
                'type': 'danger',
                'title': 'Fake Steam Domain',
                'description': 'URL uses a misspelled Steam domain (typosquatting).',
                'risk': 'high'
            })

        # Content indicators
        phishing_count = features.get('phishing_keyword_count', 0)
        if phishing_count > 2:
            indicators.append({
                'type': 'danger',
                'title': 'Multiple Phishing Keywords',
                'description': f'URL contains {phishing_count} suspicious keywords.',
                'risk': 'high'
            })
        elif phishing_count > 0:
            indicators.append({
                'type': 'warning',
                'title': 'Phishing Keywords Detected',
                'description': f'URL contains {phishing_count} suspicious keyword(s).',
                'risk': 'medium'
            })

        # Protocol indicators
        if features.get('has_https', 0) == 0:
            indicators.append({
                'type': 'warning',
                'title': 'No HTTPS',
                'description': 'URL uses HTTP instead of HTTPS (less secure).',
                'risk': 'medium'
            })

        # Gaming context indicators
        gaming_count = features.get('gaming_keyword_count', 0)
        if gaming_count > 0:
            indicators.append({
                'type': 'info',
                'title': 'Gaming Context',
                'description': f'URL contains {gaming_count} gaming-related keyword(s).',
                'risk': 'info'
            })

        return indicators

    def get_recommendation(self, probability: float, indicators: List[Dict[str, Any]]) -> str:
        """Generate recommendation based on risk assessment"""
        if probability > 0.7:
            return "⚠️ **HIGH RISK - Do not proceed**\n\nThis URL shows strong phishing indicators. Do not enter any credentials or personal information."
        elif probability > 0.4:
            high_risk_count = sum(1 for ind in indicators if ind['risk'] == 'high')
            if high_risk_count > 0:
                return "⚠️ **MEDIUM-HIGH RISK - Exercise extreme caution**\n\nMultiple risk indicators detected. Verify the URL through official channels."
            else:
                return "⚠️ **MEDIUM RISK - Proceed with caution**\n\nSome suspicious elements detected. Double-check the website's authenticity."
        else:
            return "✅ **LOW RISK - Likely safe**\n\nNo significant phishing indicators detected. The URL appears legitimate."

    @staticmethod
    def normalize_url(raw_url: str) -> str:
        """Normalize and validate an HTTP(S) URL without making a network request."""
        url = raw_url.strip()
        if not url:
            raise ValueError("Enter a URL to run the security check.")
        if re.search(r"\s", url):
            raise ValueError("The URL cannot contain spaces or line breaks.")
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
            url = "https://" + url

        try:
            parsed = urlsplit(url)
        except ValueError as error:
            raise ValueError("The URL format is invalid.") from error

        if parsed.scheme.lower() not in {"http", "https"}:
            raise ValueError("Only HTTP and HTTPS URLs can be checked.")
        if not parsed.hostname:
            raise ValueError("Enter a valid URL with a domain name.")
        try:
            port = parsed.port
        except ValueError as error:
            raise ValueError("The URL contains an invalid port.") from error
        if port is not None and not 1 <= port <= 65535:
            raise ValueError("The URL contains an invalid port.")

        return url

    @staticmethod
    def mascot_svg(state: str) -> str:
        """Return the local, CSS-animated pufferfish illustration."""
        if state not in {"idle", "danger", "safe"}:
            state = "idle"
        status = {
            "idle": "Ready to analyze",
            "danger": "Phishing detected",
            "safe": "Classified legitimate",
        }[state]
        return f"""
        <div class="mascot-widget {state}">
          <div class="mascot {state}" role="img" aria-label="Pufferfish: {state}">
            <svg class="fish" viewBox="0 0 180 155" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <linearGradient id="pufferBody" x1="0" y1="0" x2="1" y2="1">
                  <stop class="body-highlight" offset="0" stop-color="#77f0d2"/>
                  <stop class="body-shadow" offset="1" stop-color="#20b9bd"/>
                </linearGradient>
                <linearGradient id="pufferFin" x1="0" y1="0" x2="1" y2="1">
                  <stop class="fin-highlight" offset="0" stop-color="#3bd3c7"/>
                  <stop class="fin-shadow" offset="1" stop-color="#128e9a"/>
                </linearGradient>
              </defs>
              <g class="spikes" fill="none" stroke="#a4fff0" stroke-linecap="round" stroke-width="5">
                <animate attributeName="stroke" values="#a4fff0;#ff987d;#a4fff0" dur="1.4s" repeatCount="indefinite"/>
                <path d="M90 39V22 M62 49 49 36 M46 76H27 M60 103 46 117 M90 117V137 M120 103 135 117 M134 76H153 M119 49 133 35"/>
                <path d="m75 42-5-16 m30 16 5-16 M52 61 36 55 m96 6 16-6 M53 91l-16 7 m90-7 16 7 M74 110l-5 16 m37-16 5 16"/>
              </g>
              <g class="fish-body">
                <path d="M47 79c-17-13-21-7-24 0 9 2 12 8 23 10" fill="url(#pufferFin)"/>
                <path d="M69 111c2 13-4 19-10 23 12 2 22-5 25-18M106 111c-2 13 4 19 10 23-12 2-22-5-25-18" fill="url(#pufferFin)"/>
                <ellipse cx="90" cy="78" rx="45" ry="39" fill="url(#pufferBody)" stroke="#a4fff0" stroke-width="2"/>
                <g fill="#0c5264" opacity=".45">
                  <circle cx="59" cy="70" r="2.2"/><circle cx="69" cy="48" r="2"/>
                  <circle cx="111" cy="48" r="2"/><circle cx="121" cy="70" r="2.2"/>
                  <circle cx="62" cy="91" r="2"/><circle cx="118" cy="91" r="2"/>
                </g>
                <g class="eyes-calm">
                  <ellipse cx="76" cy="72" rx="5" ry="7" fill="#092638"/>
                  <ellipse cx="104" cy="72" rx="5" ry="7" fill="#092638"/>
                  <circle cx="77.5" cy="69.5" r="1.8" fill="white"/>
                  <circle cx="105.5" cy="69.5" r="1.8" fill="white"/>
                </g>
                <g class="eyes-alert">
                  <circle cx="76" cy="71" r="8" fill="white"/>
                  <circle cx="104" cy="71" r="8" fill="white"/>
                  <circle cx="77" cy="73" r="3.8" fill="#092638"/>
                  <circle cx="105" cy="73" r="3.8" fill="#092638"/>
                </g>
                <path class="mouth-calm" d="M82 88q8 8 16 0" fill="none" stroke="#12576a" stroke-linecap="round" stroke-width="3"/>
                <ellipse class="mouth-alert" cx="90" cy="91" rx="5" ry="7" fill="#12576a"/>
                <circle cx="61" cy="83" r="5" fill="#ff9d9a" opacity=".75"/>
                <circle cx="119" cy="83" r="5" fill="#ff9d9a" opacity=".75"/>
              </g>
              <circle cx="35" cy="31" r="4" fill="#58d9d0" opacity=".75"/>
              <circle cx="143" cy="103" r="3" fill="#58d9d0" opacity=".65"/>
            </svg>
          </div>
          <div class="mascot-status {state}">{status}</div>
        </div>
        """

    def run(self):
        """Render the URL checker and its prediction-driven mascot."""
        header_slot = st.empty()
        result = None
        error_message = None
        mascot_state = "idle"
        with st.form("url_check_form"):
            st.markdown('<div class="url-label">Website URL</div>', unsafe_allow_html=True)
            input_url = st.text_input(
                "Website URL",
                placeholder="https://example.com/login",
                label_visibility="collapsed",
            )
            available_models = [
                name for name in ("logistic_regression", "random_forest", "xgboost")
                if name in self.models
            ]
            selected_model = None
            if available_models:
                model_options = ["fusion", *available_models]
                selected_model = st.selectbox(
                    "Detection model",
                    options=model_options,
                    format_func=lambda name: (
                        "Weighted ensemble" if name == "fusion"
                        else name.replace("_", " ").title()
                    ),
                )
            analyze_button = st.form_submit_button(
                "Analyze URL  →",
                use_container_width=True,
            )
        st.markdown(
            '<div class="hint">Enter a web address to check it locally. No website is opened or contacted.</div>',
            unsafe_allow_html=True,
        )
        result_slot = st.empty()

        if analyze_button:
            try:
                normalized_url = self.normalize_url(input_url)
                if self.load_error:
                    raise RuntimeError(
                        f"Detection resources could not be loaded: {self.load_error}"
                    )
                if not selected_model:
                    raise RuntimeError("No trained detection models are available.")

                with st.spinner("Analyzing URL features..."):
                    features = self.extract_features(normalized_url)
                    if selected_model == "fusion":
                        probability, confidence = self.get_fusion_prediction(features)
                    else:
                        probability, confidence = self.predict_with_model(
                            selected_model, features
                        )
                if not np.isfinite(probability) or not np.isfinite(confidence):
                    raise RuntimeError("The selected model did not return a valid result.")

                indicators = self.analyze_risk_indicators(normalized_url, features)
                prediction = "Phishing" if probability >= 0.5 else "Legitimate"
                risk_level = (
                    "High" if probability > 0.7
                    else "Medium" if probability > 0.4
                    else "Low"
                )
                result = {
                    "url": normalized_url,
                    "features": features,
                    "probability": probability,
                    "confidence": confidence,
                    "indicators": indicators,
                    "prediction": prediction,
                    "risk_level": risk_level,
                    "recommendation": self.get_recommendation(probability, indicators),
                }
                mascot_state = "danger" if prediction == "Phishing" else "safe"
            except (ValueError, RuntimeError) as error:
                error_message = str(error)
            except Exception as error:
                error_message = f"Analysis failed. Check the selected model and try again. ({error})"

        header_slot.markdown(
            '<div class="brand">PUFFERPHISH</div>'
            '<div class="hero-layout">'
            '<div class="eyebrow">Gaming link protection</div>'
            '<div class="hero"><h1>Know before<br>you click.</h1>'
            '<p>Check a URL for phishing signals using your trained machine-learning models.</p></div>'
            '</div>'
            f'{self.mascot_svg(mascot_state)}',
            unsafe_allow_html=True,
        )

        if error_message:
            result_slot.error(error_message)
        elif result:
            probability = result["probability"]
            risk_class = (
                "high" if result["risk_level"] == "High"
                else "medium" if result["risk_level"] == "Medium"
                else "low"
            )
            icon = "!" if result["prediction"] == "Phishing" else "✓"
            subtitle = (
                "Phishing indicators detected by the selected model."
                if result["prediction"] == "Phishing"
                else "The selected model classified this URL as legitimate."
            )
            safe_url = html.escape(result["url"])
            with result_slot.container():
                st.markdown(
                    f'<div class="result-card risk-{risk_class}">'
                    f'<div class="result-top"><div class="result-icon">{icon}</div>'
                    f'<div><div class="result-title">{result["prediction"]} · '
                    f'{result["risk_level"]} risk</div>'
                    f'<div class="result-subtitle">{subtitle}</div></div></div>'
                    f'<div class="result-url">{safe_url}</div></div>',
                    unsafe_allow_html=True,
                )
                score_col, confidence_col, risk_col = st.columns(3)
                score_col.metric("Phishing probability", f"{probability:.1%}")
                confidence_col.metric("Model confidence", f'{result["confidence"]:.1%}')
                risk_col.metric("Risk level", result["risk_level"])

                st.markdown('<div class="section-title">Detection signals</div>', unsafe_allow_html=True)
                if result["indicators"]:
                    for indicator in result["indicators"]:
                        indicator_class = (
                            "high" if indicator["risk"] == "high"
                            else "info" if indicator["risk"] == "info"
                            else ""
                        )
                        st.markdown(
                            f'<div class="indicator {indicator_class}">'
                            f'<strong>{html.escape(indicator["title"])}</strong> &nbsp; '
                            f'{html.escape(indicator["description"])}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        '<div class="indicator info"><strong>No obvious warning signs</strong> '
                        'The URL did not trigger the risk checks used by this prototype.</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown('<div class="section-title">Recommendation</div>', unsafe_allow_html=True)
                recommendation = result["recommendation"].replace("**", "")
                st.markdown(
                    f'<div class="recommendation">{html.escape(recommendation).replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown(
            '<div class="footer">PufferPhish · AI-assisted URL risk assessment · '
            'A low-risk result does not guarantee that a website is safe.</div>',
            unsafe_allow_html=True,
        )


# Main execution
if __name__ == "__main__":
    import os

    # Check if required files exist
    required_dirs = ['./models', './data']
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]

    if missing_dirs:
        st.warning(f"⚠️ Missing directories: {', '.join(missing_dirs)}")
        st.info("Running in demo mode with basic features.")

    # Initialize and run app
    app = PhishingDetectionApp()
    app.run()
