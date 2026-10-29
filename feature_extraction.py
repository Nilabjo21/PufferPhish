#!/usr/bin/env python3
"""
Feature Extraction for G65 Phishing Detection Framework
Extracts URL, textual, and gaming-specific features for phishing detection.
"""

import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse, parse_qs
from typing import List, Dict, Tuple
import logging
from tldextract import extract
import math
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NON_FEATURE_COLUMNS = {'url', 'is_phishing', 'domain', 'source', 'category', 'suffix'}


def get_numeric_feature_columns(frame: pd.DataFrame) -> List[str]:
    """Return the shared numeric model-input columns from a feature dataset."""
    return [
        column for column in frame.select_dtypes(include=[np.number]).columns
        if column not in NON_FEATURE_COLUMNS
    ]


class URLFeatureExtractor:
    def __init__(self):
        """Initialize feature extractor with phishing keyword lists"""
        self.phishing_keywords = [
            'verify', 'confirm', 'account', 'login', 'password', 'secure',
            'bank', 'update', 'unlock', 'suspend', 'suspended', 'alert',
            'security', 'click', 'urgent', 'immediately', 'now', 'free',
            'gift', 'giveaway', 'win', 'claim', 'offer', 'limited', 'time',
            'verify-account', 'confirm-identity', 'reset-password'
        ]

        self.gaming_keywords = [
            'steam', 'valve', 'gaming', 'game', 'csgo', 'dota', 'tf2',
            'skin', 'skins', 'item', 'items', 'trade', 'trading', 'inventory',
            'market', 'wallet', 'gems', 'coins', 'points', 'vbucks',
            'tournament', 'league', 'esports', 'competitive', 'rank',
            'mmr', 'elo', 'kdr', 'frag', 'clutch', 'ace', 'headshot',
            'legendary', 'epic', 'rare', 'common', 'unusual', 'strange'
        ]

        self.urgency_keywords = [
            'now', 'today', 'immediately', 'urgent', 'hurry', 'limited',
            'expire', 'expires', 'ending', 'final', 'last', 'warning',
            'alert', 'important', 'critical', 'action', 'required'
        ]

        self.suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top']

    def extract_all_features(self, url: str) -> Dict[str, float]:
        """Extract all features from a URL"""
        features = {}

        # Basic URL features
        features.update(self._extract_basic_features(url))

        # URL structure features
        features.update(self._extract_structure_features(url))

        # Content-based features
        features.update(self._extract_content_features(url))

        # Domain-based features
        features.update(self._extract_domain_features(url))

        # Gaming-specific features
        features.update(self._extract_gaming_features(url))

        # Risk indicators
        features.update(self._extract_risk_indicators(url))

        return features

    def _extract_basic_features(self, url: str) -> Dict[str, float]:
        """Extract basic URL features"""
        features = {}

        # URL length
        features['url_length'] = len(url)

        # Protocol
        features['has_https'] = 1.0 if url.startswith('https://') else 0.0
        features['has_http'] = 1.0 if url.startswith('http://') else 0.0

        # Number of digits in URL
        features['num_digits'] = sum(c.isdigit() for c in url)

        # Number of special characters
        special_chars = r'[!@#$%^&*()\-_=+{}\[\]|\\;:\'",.<>?`~]'
        features['num_special_chars'] = len(re.findall(special_chars, url))

        return features

    def _extract_structure_features(self, url: str) -> Dict[str, float]:
        """Extract URL structure features"""
        features = {}

        try:
            parsed = urlparse(url)

            # Path length
            path = parsed.path
            features['path_length'] = len(path)

            # Number of directories in path
            features['num_directories'] = len([p for p in path.split('/') if p])

            # Query parameters
            query = parsed.query
            features['has_query'] = 1.0 if query else 0.0

            if query:
                params = parse_qs(query)
                features['num_params'] = len(params)
                features['param_length'] = len(query)
            else:
                features['num_params'] = 0
                features['param_length'] = 0

            # Fragment
            features['has_fragment'] = 1.0 if parsed.fragment else 0.0

            # Subdomain analysis
            domain = parsed.netloc
            if domain:
                parts = domain.split('.')
                features['num_subdomains'] = max(0, len(parts) - 2)  # Remove domain and TLD

        except:
            # Default values if parsing fails
            features['path_length'] = 0
            features['num_directories'] = 0
            features['has_query'] = 0.0
            features['num_params'] = 0
            features['param_length'] = 0
            features['has_fragment'] = 0.0
            features['num_subdomains'] = 0

        return features

    def _extract_content_features(self, url: str) -> Dict[str, float]:
        """Extract content-based features"""
        features = {}
        url_lower = url.lower()

        # Keyword presence
        for keyword in self.phishing_keywords:
            features[f'has_phishing_{keyword}'] = 1.0 if keyword in url_lower else 0.0

        # Count of phishing keywords
        phishing_count = sum(1 for keyword in self.phishing_keywords if keyword in url_lower)
        features['phishing_keyword_count'] = phishing_count
        features['phishing_keyword_density'] = phishing_count / max(1, len(url_lower.split()))

        # Urgency keywords
        urgency_count = sum(1 for keyword in self.urgency_keywords if keyword in url_lower)
        features['urgency_keyword_count'] = urgency_count

        # Suspicious patterns
        features['has_ip_address'] = 1.0 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0.0
        features['has_at_symbol'] = 1.0 if '@' in url else 0.0
        features['has_double_slash'] = 1.0 if '//' in url[8:] else 0.0  # After protocol

        # URL encoding
        encoded_patterns = ['%20', '%2F', '%3D', '%3F', '%26', '%25']
        features['has_encoding'] = 1.0 if any(pattern in url for pattern in encoded_patterns) else 0.0

        return features

    def _extract_domain_features(self, url: str) -> Dict[str, float]:
        """Extract domain-based features"""
        features = {}

        try:
            # Extract domain components
            parsed = urlparse(url)
            domain = parsed.netloc

            if domain:
                # Domain length
                features['domain_length'] = len(domain)

                # TLD analysis
                tld_extract = extract(url)
                features['domain'] = tld_extract.domain
                features['suffix'] = tld_extract.suffix

                # Check for suspicious TLDs
                features['suspicious_tld'] = 1.0 if tld_extract.suffix in self.suspicious_tlds else 0.0

                # Hyphens in domain
                features['hyphens_in_domain'] = domain.count('-')

                # Subdomain ratio
                if '.' in domain:
                    parts = domain.split('.')
                    features['subdomain_ratio'] = len(parts[:-2]) / max(1, len(parts))
                else:
                    features['subdomain_ratio'] = 0.0

            else:
                features['domain_length'] = 0
                features['suspicious_tld'] = 0.0
                features['hyphens_in_domain'] = 0
                features['subdomain_ratio'] = 0.0

        except:
            features['domain_length'] = 0
            features['suspicious_tld'] = 0.0
            features['hyphens_in_domain'] = 0
            features['subdomain_ratio'] = 0.0

        return features

    def _extract_gaming_features(self, url: str) -> Dict[str, float]:
        """Extract gaming-specific features"""
        features = {}
        url_lower = url.lower()

        # Gaming keyword presence
        gaming_keyword_count = 0
        for keyword in self.gaming_keywords:
            feature_name = f'has_gaming_{keyword}'
            has_keyword = 1.0 if keyword in url_lower else 0.0
            features[feature_name] = has_keyword
            if has_keyword:
                gaming_keyword_count += 1

        features['gaming_keyword_count'] = gaming_keyword_count
        features['gaming_keyword_density'] = gaming_keyword_count / max(1, len(url_lower.split()))

        # Steam-specific patterns
        steam_patterns = [
            'steamcommunity', 'steampowered', 'steam-gifts', 'steam-trade',
            'steam-wallet', 'steam-guard', 'steam-market', 'steam-inventory'
        ]

        steam_count = sum(1 for pattern in steam_patterns if pattern in url_lower)
        features['steam_pattern_count'] = steam_count

        # Check for fake Steam domains
        fake_steam_patterns = [
            'steamcommmunity', 'steamcomunnity', 'steam-comunnity',
            'steampowred', 'steam-powred', 'steamcommuity'
        ]

        features['has_fake_steam'] = 1.0 if any(pattern in url_lower for pattern in fake_steam_patterns) else 0.0

        # Trading-related terms
        trading_terms = ['trade', 'trading', 'market', 'inventory', 'skin', 'skins', 'item', 'items']
        trading_count = sum(1 for term in trading_terms if term in url_lower)
        features['trading_term_count'] = trading_count

        return features

    def _extract_risk_indicators(self, url: str) -> Dict[str, float]:
        """Extract risk indicators for phishing detection"""
        features = {}
        url_lower = url.lower()

        # 1. Suspicious protocol-port combinations
        suspicious_ports = [':8080', ':80', ':443', ':3000']
        features['suspicious_port'] = 1.0 if any(port in url for port in suspicious_ports) else 0.0

        # 2. Excessive subdomains
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            if domain:
                subdomain_count = len(domain.split('.')) - 2
                features['excessive_subdomains'] = 1.0 if subdomain_count > 3 else 0.0
            else:
                features['excessive_subdomains'] = 0.0
        except:
            features['excessive_subdomains'] = 0.0

        # 3. Short URL indicators
        url_shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 't.co', 'is.gd', 'buff.ly', 'adf.ly']
        features['is_url_shortener'] = 1.0 if any(shortener in url_lower for shortener in url_shorteners) else 0.0

        # 4. Homograph attack indicators (IDN)
        features['has_idn_chars'] = 1.0 if any(ord(c) > 127 for c in url) else 0.0

        # 5. Redirect chains
        redirect_params = ['redirect', 'redirect_to', 'redirect_url', 'url', 'u', 'link', 'next']
        features['has_redirect_param'] = 1.0 if any(param in url_lower for param in redirect_params) else 0.0

        # 6. Credential harvesting indicators
        credential_params = ['login', 'password', 'pass', 'pwd', 'username', 'user', 'email']
        features['has_credential_param'] = 1.0 if any(param in url_lower for param in credential_params) else 0.0

        # 7. BitB attack indicators (iframe/embed)
        bitb_patterns = ['iframe', 'embed', 'popup', 'modal', 'overlay', 'window']
        features['has_bitb_pattern'] = 1.0 if any(pattern in url_lower for pattern in bitb_patterns) else 0.0

        # 8. JavaScript execution
        js_patterns = ['javascript:', 'js:', 'onclick=', 'onload=', 'onerror=']
        features['has_js_pattern'] = 1.0 if any(pattern in url_lower for pattern in js_patterns) else 0.0

        # 9. Form submission
        form_patterns = ['form', 'submit', 'post', 'action=']
        features['has_form_pattern'] = 1.0 if any(pattern in url_lower for pattern in form_patterns) else 0.0

        # 10. Overall risk score (composite)
        risk_factors = [
            features.get('suspicious_port', 0),
            features.get('excessive_subdomains', 0),
            features.get('is_url_shortener', 0),
            features.get('has_idn_chars', 0),
            features.get('has_redirect_param', 0),
            features.get('has_credential_param', 0),
            features.get('has_bitb_pattern', 0),
            features.get('has_js_pattern', 0),
            features.get('has_form_pattern', 0),
            features.get('has_fake_steam', 0)
        ]

        features['risk_score'] = sum(risk_factors) / max(1, len(risk_factors))

        return features


class TextFeatureExtractor:
    """Extract textual features from URL/message content"""
    def __init__(self):
        self.sentiment_words = {
            'urgent': ['urgent', 'immediately', 'now', 'hurry', 'quick', 'fast'],
            'threatening': ['suspended', 'locked', 'banned', 'terminated', 'closed'],
            'rewarding': ['free', 'gift', 'prize', 'win', 'reward', 'bonus'],
            'authoritative': ['official', 'secure', 'verified', 'trusted', 'authentic']
        }

    def extract_text_features(self, url: str) -> Dict[str, float]:
        """Extract text-based features"""
        features = {}
        url_lower = url.lower()

        # Sentiment analysis
        for sentiment, words in self.sentiment_words.items():
            count = sum(1 for word in words if word in url_lower)
            features[f'{sentiment}_word_count'] = count

        # Character distributions
        features['letter_ratio'] = sum(c.isalpha() for c in url) / max(1, len(url))
        features['digit_ratio'] = sum(c.isdigit() for c in url) / max(1, len(url))
        features['special_char_ratio'] = sum(not c.isalnum() for c in url) / max(1, len(url))

        # Vocabulary richness (simplified)
        words = url_lower.split()
        if words:
            features['avg_word_length'] = sum(len(w) for w in words) / len(words)
            features['unique_word_ratio'] = len(set(words)) / len(words)
        else:
            features['avg_word_length'] = 0
            features['unique_word_ratio'] = 0

        # Entropy (measure of randomness)
        features['entropy'] = self._calculate_entropy(url)

        return features

    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of the text"""
        import math
        if not text:
            return 0

        # Count frequency of each character
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1

        # Calculate entropy
        entropy = 0
        length = len(text)
        for count in freq.values():
            p = count / length
            entropy -= p * math.log2(p)

        return entropy


class GamingContextExtractor:
    """Extract gaming-specific contextual features"""
    def __init__(self):
        self.gaming_platforms = {
            'steam': ['steam', 'valve'],
            'epic': ['epic', 'fortnite'],
            'riot': ['riot', 'league', 'valorant'],
            'blizzard': ['blizzard', 'overwatch', 'wow'],
            'ubisoft': ['ubisoft', 'assassin', 'far cry'],
            'ea': ['ea', 'electronic arts', 'fifa', 'madden']
        }

        self.gaming_genres = [
            'fps', 'rpg', 'mmo', 'moba', 'battle', 'royale', 'strategy',
            'simulation', 'adventure', 'puzzle', 'arcade', 'sports'
        ]

    def extract_gaming_context(self, url: str) -> Dict[str, float]:
        """Extract gaming context features"""
        features = {}
        url_lower = url.lower()

        # Platform detection
        for platform, keywords in self.gaming_platforms.items():
            platform_count = sum(1 for keyword in keywords if keyword in url_lower)
            features[f'platform_{platform}'] = 1.0 if platform_count > 0 else 0.0

        # Genre detection
        genre_count = 0
        for genre in self.gaming_genres:
            if genre in url_lower:
                genre_count += 1
                features[f'genre_{genre}'] = 1.0
            else:
                features[f'genre_{genre}'] = 0.0

        features['genre_count'] = genre_count

        # Esports indicators
        esports_terms = ['tournament', 'championship', 'league', 'esports', 'competitive', 'pro', 'professional']
        esports_count = sum(1 for term in esports_terms if term in url_lower)
        features['esports_indicator'] = esports_count

        # Trading indicators
        trading_terms = ['trade', 'trading', 'market', 'auction', 'sell', 'buy', 'price', 'value']
        trading_count = sum(1 for term in trading_terms if term in url_lower)
        features['trading_indicator'] = trading_count

        # Account security indicators
        security_terms = ['account', 'login', 'password', 'verify', 'secure', 'guard', 'auth', 'authenticator']
        security_count = sum(1 for term in security_terms if term in url_lower)
        features['security_indicator'] = security_count

        return features


class FeaturePipeline:
    """Complete feature extraction pipeline"""
    def __init__(self):
        self.url_extractor = URLFeatureExtractor()
        self.text_extractor = TextFeatureExtractor()
        self.gaming_extractor = GamingContextExtractor()

    def extract_features(self, url: str) -> Dict[str, float]:
        """Extract all features from URL"""
        features = {}

        # URL features
        features.update(self.url_extractor.extract_all_features(url))

        # Text features
        features.update(self.text_extractor.extract_text_features(url))

        # Gaming context features
        features.update(self.gaming_extractor.extract_gaming_context(url))

        return features

    def extract_features_batch(self, urls: List[str]) -> List[Dict[str, float]]:
        """Extract features for multiple URLs"""
        features_list = []
        for i, url in enumerate(urls):
            if i % 100 == 0:
                logger.info(f"Processing URL {i+1}/{len(urls)}")
            features = self.extract_features(url)
            features_list.append(features)

        return features_list

    def create_feature_dataframe(self, urls: List[str], labels: List[int] = None) -> pd.DataFrame:
        """Create DataFrame with extracted features"""
        features_list = self.extract_features_batch(urls)
        df = pd.DataFrame(features_list)

        # Add URL and labels if provided
        df['url'] = urls
        if labels is not None:
            df['is_phishing'] = labels

        return df


def main():
    """Main execution function"""
    logger.info("Starting G65 Feature Extraction")

    # Create sample URLs for testing
    sample_urls = [
        "https://steamcommunity.com/login",
        "https://steamcommmunity.com/verify-account",
        "https://free-csgo-skins-trade.com/steam-auth",
        "https://store.steampowered.com/app/730",
        "https://steam-account-suspended.com/recover",
        "https://help.steampowered.com/en/faqs",
        "https://steam-gifts-free.com/claim",
        "https://discord.com",
        "https://twitch.tv",
        "https://steam-trading-bot.com/connect"
    ]

    sample_labels = [0, 1, 1, 0, 1, 0, 1, 0, 0, 1]

    # Initialize pipeline
    pipeline = FeaturePipeline()

    # Extract features
    logger.info("Extracting features...")
    df = pipeline.create_feature_dataframe(sample_urls, sample_labels)

    # Display results
    logger.info(f"Extracted {df.shape[1]} features for {df.shape[0]} URLs")
    logger.info("\nFeature summary:")
    print(f"Total features: {len(df.columns)}")
    print(f"Sample features (first 20): {list(df.columns[:20])}")
    print(f"\nSample data (first 3 rows):")
    print(df.head(3))

    # Save features
    output_path = "extracted_features.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Features saved to: {output_path}")

    # Show some statistics
    phishing_mask = df['is_phishing'] == 1
    phishing_df = df[phishing_mask]
    legit_df = df[~phishing_mask]

    logger.info(f"\nDataset Statistics:")
    logger.info(f"  - Phishing URLs: {len(phishing_df)}")
    logger.info(f"  - Legitimate URLs: {len(legit_df)}")
    logger.info(f"  - Feature columns: {len(df.columns)}")

    # Show top 10 features by variance
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    variance = df[numeric_cols].var().sort_values(ascending=False)
    logger.info(f"\nTop 10 features by variance:")
    for i, (feature, var_value) in enumerate(variance.head(10).items()):
        logger.info(f"  {i+1:2d}. {feature}: {var_value:.4f}")

    return df


if __name__ == "__main__":
    df = main()
