#!/usr/bin/env python3
"""
Data Collection for G65 Phishing Detection Framework
Builds and prepares a synthetic educational dataset for phishing detection.
"""

import pandas as pd
import numpy as np
import requests
from io import StringIO
import zipfile
import os
from pathlib import Path
from typing import Tuple, List
import logging
import random
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhishingDataCollector:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def fetch_phishing_datasets(self) -> pd.DataFrame:
        """
        Assemble simulated seed examples and deterministic synthetic URLs.
        No external threat-intelligence source is downloaded.
        """
        logger.info("Fetching phishing datasets...")

        phishing_samples = []

        # 1. PhishTank-like seed examples, simulated for educational use
        try:
            phishtank_urls = self._get_phishtank_samples()
            phishing_samples.extend(phishtank_urls)
            logger.info(f"✓ Simulated phishing seed patterns: {len(phishtank_urls)} samples")
        except Exception as e:
            logger.warning(f"PhishTank failed: {e}")

        # 2. Synthetic gaming phishing URLs (based on patterns)
        synthetic_urls = self._create_synthetic_gaming_phishing()
        phishing_samples.extend(synthetic_urls)
        logger.info(f"✓ Synthetic gaming phishing: {len(synthetic_urls)} samples")

        augmented_urls = self._create_augmented_phishing()
        phishing_samples.extend(augmented_urls)
        logger.info(f"✓ Synthetic phishing augmentation: {len(augmented_urls)} samples")
        augmented_url_set = set(augmented_urls)

        # 3. Research-inspired seed examples; no external dataset is downloaded
        research_urls = self._get_research_phishing_samples()
        phishing_samples.extend(research_urls)
        logger.info(f"✓ Research-inspired synthetic patterns: {len(research_urls)} samples")

        # Create DataFrame
        phishing_df = pd.DataFrame({
            'url': phishing_samples,
            'is_phishing': 1,
            'source': [
                'synthetic_augmentation' if url in augmented_url_set else 'synthetic_seed'
                for url in phishing_samples
            ],
            'category': 'gaming_phishing'
        }).drop_duplicates(subset='url')

        # Save raw data
        phishing_df.to_csv(self.raw_dir / "phishing_urls.csv", index=False)

        return phishing_df

    def fetch_legitimate_datasets(self) -> pd.DataFrame:
        """
        Assemble known-domain seeds and synthetic legitimate-path variants.
        """
        logger.info("Fetching legitimate URLs...")

        legitimate_samples = []

        # 1. Legitimate gaming domains
        gaming_urls = self._get_legitimate_gaming_urls()
        legitimate_samples.extend(gaming_urls)
        logger.info(f"✓ Legitimate gaming: {len(gaming_urls)} samples")

        # 2. General legitimate domains
        general_urls = self._get_general_legitimate_urls()
        legitimate_samples.extend(general_urls)
        logger.info(f"✓ General legitimate: {len(general_urls)} samples")

        # 3. Steam community URLs
        steam_urls = self._get_steam_urls()
        legitimate_samples.extend(steam_urls)
        logger.info(f"✓ Steam URLs: {len(steam_urls)} samples")

        augmented_urls = self._create_augmented_legitimate()
        legitimate_samples.extend(augmented_urls)
        logger.info(f"✓ Synthetic legitimate augmentation: {len(augmented_urls)} samples")
        augmented_url_set = set(augmented_urls)

        # Create DataFrame
        legit_df = pd.DataFrame({
            'url': legitimate_samples,
            'is_phishing': 0,
            'source': [
                'synthetic_augmentation' if url in augmented_url_set else 'known_domain_seed'
                for url in legitimate_samples
            ],
            'category': 'legitimate'
        }).drop_duplicates(subset='url')

        # Save raw data
        legit_df.to_csv(self.raw_dir / "legitimate_urls.csv", index=False)

        return legit_df

    def combine_and_prepare(self, phishing_df: pd.DataFrame, legitimate_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine phishing and legitimate datasets, balance, and split.
        """
        logger.info("Combining and preparing dataset...")

        # Combine and remove repeated URLs before balancing.
        combined_df = pd.concat([phishing_df, legitimate_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset='url').reset_index(drop=True)
        combined_df['domain'] = combined_df['url'].apply(self._extract_domain)

        # Balance the generated corpus without duplicating records.
        phishing_count = combined_df['is_phishing'].sum()
        legit_count = len(combined_df) - phishing_count
        min_count = min(phishing_count, legit_count)

        balanced_phishing = combined_df[combined_df['is_phishing'] == 1].sample(
            min_count, random_state=42
        )
        balanced_legit = combined_df[combined_df['is_phishing'] == 0].sample(
            min_count, random_state=42
        )

        balanced_df = pd.concat([balanced_phishing, balanced_legit], ignore_index=True)

        from feature_extraction import FeaturePipeline

        extractor = FeaturePipeline()
        feature_df = pd.DataFrame(
            extractor.extract_features_batch(balanced_df['url'].tolist())
        ).drop(columns=['domain', 'suffix'], errors='ignore')
        balanced_df = pd.concat(
            [balanced_df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1
        )
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"✓ Combined dataset: {len(balanced_df)} total samples")
        logger.info(f"  - Phishing: {balanced_df['is_phishing'].sum()}")
        logger.info(f"  - Legitimate: {len(balanced_df) - balanced_df['is_phishing'].sum()}")

        # Save processed data
        balanced_df.to_csv(self.processed_dir / "full_dataset.csv", index=False)

        return balanced_df

    def create_train_test_split(self, df: pd.DataFrame, test_size: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split with stratification.
        """
        from sklearn.model_selection import GroupShuffleSplit

        logger.info(f"Creating train/test split (test_size={test_size})...")
        train_parts = []
        test_parts = []
        for label in sorted(df['is_phishing'].unique()):
            class_df = df[df['is_phishing'] == label]
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=42
            )
            train_indices, test_indices = next(
                splitter.split(class_df, groups=class_df['domain'])
            )
            train_parts.append(class_df.iloc[train_indices])
            test_parts.append(class_df.iloc[test_indices])

        train_df = pd.concat(train_parts).sample(frac=1, random_state=42).reset_index(drop=True)
        test_df = pd.concat(test_parts).sample(frac=1, random_state=42).reset_index(drop=True)
        overlap = set(train_df['domain']) & set(test_df['domain'])
        if overlap:
            raise ValueError(f"Domain leakage between train and test splits: {len(overlap)} domains")

        logger.info(f"✓ Training set: {len(train_df)} samples")
        logger.info(f"✓ Test set: {len(test_df)} samples")

        # Save splits
        train_df.to_csv(self.processed_dir / "train_dataset.csv", index=False)
        test_df.to_csv(self.processed_dir / "test_dataset.csv", index=False)

        return train_df, test_df

    def _get_phishtank_samples(self) -> List[str]:
        """Get sample phishing URLs (simulated for educational purposes)"""
        # In a real implementation, download from PhishTank API
        # For this educational project, use simulated data
        sample_phishing_patterns = [
            "https://steamcommmunity.com/login",
            "https://steam-gifts-free.com/verify",
            "https://steamtrade-offers.com/confirm",
            "http://steamwallet-codes.com/claim",
            "https://steam-account-verify.net/secure",
            "https://steamcomunnity-profile.com/update",
            "http://steam-trading-cards-free.com/redeem",
            "https://steampowered-giveaway.com/win",
            "http://steam-support-help.com/reset",
            "https://steam-inventory-trade.com/accept",
            "http://valve-steam-security.com/check",
            "https://steam-market-skins.com/transfer",
            "http://steam-guard-reset.com/recover",
            "https://steam-trading-bot.com/connect",
            "http://steam-community-market.com/buy"
        ]
        return sample_phishing_patterns

    def _create_synthetic_gaming_phishing(self) -> List[str]:
        """Create synthetic gaming phishing URLs based on common patterns"""
        gaming_phishing_patterns = []

        # Browser-in-Browser (BitB) style URLs
        bitb_patterns = [
            "https://steamcommunity.com{malicious-domain}/login",
            "https://store.steampowered.com{fake}/verify",
            "https://help.steampowered.com{hack}/reset"
        ]

        # Trading scam URLs
        trading_patterns = [
            "https://free-csgo-skins-trade.com/steam-login",
            "https://dota2-items-giveaway.com/connect-steam",
            "https://tf2-unusual-hats.com/steam-auth",
            "https://valorant-points-free.com/steam-verify",
            "https://fortnite-vbucks-steam.com/link"
        ]

        # Account verification scams
        verification_patterns = [
            "https://steam-account-suspended.com/recover",
            "https://steam-security-alert.com/confirm",
            "https://valve-anti-cheat.com/verify-identity",
            "https://steam-guard-disabled.com/reactivate",
            "https://steam-trade-ban.com/appeal"
        ]

        # Tournament scam URLs
        tournament_patterns = [
            "https://esports-tournament-steam.com/register",
            "https://csgo-pro-league.com/steam-connect",
            "https://dota2-championship.com/participate",
            "https://valorant-tourney-steam.com/signup"
        ]

        gaming_phishing_patterns.extend(bitb_patterns)
        gaming_phishing_patterns.extend(trading_patterns)
        gaming_phishing_patterns.extend(verification_patterns)
        gaming_phishing_patterns.extend(tournament_patterns)

        return gaming_phishing_patterns

    def _create_augmented_phishing(self, count: int = 600) -> List[str]:
        """Create clearly synthetic, deterministic URL examples for training."""
        rng = random.Random(6501)
        brands = ['steam', 'valve', 'steamcommunity', 'steampowered', 'cs2', 'dota2']
        lures = [
            'login', 'account', 'verify', 'trade', 'gift', 'wallet', 'security',
            'market', 'inventory', 'support', 'reward', 'tournament'
        ]
        paths = [
            'login', 'account/verify', 'trade/accept', 'claim/reward',
            'security/confirm', 'auth/steam', 'market/offer', 'support/appeal',
            'user/session', 'connect'
        ]
        tlds = ['com', 'net', 'org', 'info', 'site', 'online', 'xyz']
        params = ['redirect', 'next', 'continue', 'return', 'session', 'token']
        urls = []
        for i in range(count):
            brand = brands[i % len(brands)]
            lure = lures[(i // len(brands)) % len(lures)]
            token = ''.join(rng.choice('abcdefghjkmnpqrstuvwxyz23456789') for _ in range(5))
            tld = tlds[(i // 3) % len(tlds)]
            path = paths[(i // 7) % len(paths)]
            host_style = i % 4
            if host_style == 0:
                host = f'{brand}-{lure}-{token}.{tld}'
            elif host_style == 1:
                host = f'{lure}-service-{token}.{tld}'
            elif host_style == 2:
                host = f'{brand}.{lure}-{token}.{tld}'
            else:
                host = f'{brand}{token}.{tld}'
            protocol = 'http' if i % 5 == 0 else 'https'
            url = f'{protocol}://{host}/{path}'
            if host_style in (1, 3):
                url += f'/steam/{rng.choice(["community", "account", "market", "support"])}'
            if i % 4 == 0:
                url += f'?{params[i % len(params)]}={rng.choice(paths).replace("/", "%2F")}'
            urls.append(url)
        return list(dict.fromkeys(urls))

    def _create_augmented_legitimate(self, count: int = 600) -> List[str]:
        """Create synthetic URL-path variants on known public service domains."""
        rng = random.Random(6502)
        domains = [
            'store.steampowered.com', 'steamcommunity.com', 'help.steampowered.com',
            'steam.tv', 'steamdb.info', 'steamcharts.com', 'discord.com', 'twitch.tv',
            'epicgames.com', 'ea.com', 'ubisoft.com', 'blizzard.com', 'riotgames.com',
            'playstation.com', 'xbox.com', 'nintendo.com', 'gog.com', 'itch.io',
            'youtube.com', 'reddit.com', 'github.com', 'google.com', 'microsoft.com',
            'apple.com', 'amazon.com', 'wikipedia.org', 'linkedin.com', 'netflix.com',
            'spotify.com', 'adobe.com', 'mozilla.org', 'cloudflare.com', 'zoom.us',
            'un.org', 'bbc.com', 'theguardian.com', 'nytimes.com', 'stackoverflow.com',
            'python.org', 'npmjs.com', 'docker.com', 'khanacademy.org', 'coursera.org',
            'paypal.com', 'dropbox.com', 'notion.so', 'canva.com', 'figma.com',
            'roblox.com', 'minecraft.net', 'valorant.com', 'playvalorant.com',
            'battle.net', 'riotgames.com', 'store.epicgames.com', 'support.xbox.com',
            'support.playstation.com', 'community.cloudflare.com', 'developer.mozilla.org',
            'support.google.com', 'learn.microsoft.com', 'steamcommunity.com'
        ]
        paths = [
            'community', 'account', 'help', 'support', 'games', 'store', 'news',
            'profile', 'discussion', 'market', 'trade', 'library', 'downloads',
            'products', 'updates', 'events', 'security', 'guides', 'articles',
            'settings', 'faq', 'contact', 'wishlist', 'reviews', 'about',
            'login', 'verify', 'account/security', 'trade/offer', 'auth/session',
            'market/listings', 'steam/community', 'support/account'
        ]
        urls = []
        for i in range(count):
            domain = domains[i % len(domains)]
            path = paths[(i // len(domains) + i) % len(paths)]
            slug = ''.join(rng.choice('abcdefghjkmnpqrstuvwxyz23456789') for _ in range(4))
            url = f'https://{domain}/{path}/{slug}'
            if i % 3 == 0:
                url += f'?page={i % 40 + 1}'
            urls.append(url)
        return list(dict.fromkeys(urls))

    def _get_research_phishing_samples(self) -> List[str]:
        """Get phishing URLs from research datasets (simulated)"""
        # These would come from UCI Phishing Websites, Kaggle, etc.
        research_samples = [
            "http://paypal-verify-account.com/login",
            "https://apple-id-confirm.com/secure",
            "http://bank-account-update.com/auth",
            "https://netflix-payment-info.com/verify",
            "http://amazon-security-alert.com/signin",
            "https://microsoft-account-locked.com/unlock",
            "http://google-verification.com/confirm",
            "https://facebook-security-check.com/login",
            "http://instagram-recovery.com/reset",
            "https://twitter-account-suspended.com/appeal"
        ]
        return research_samples

    def _get_legitimate_gaming_urls(self) -> List[str]:
        """Get legitimate gaming-related URLs"""
        legitimate_gaming = [
            "https://store.steampowered.com",
            "https://steamcommunity.com",
            "https://help.steampowered.com",
            "https://steam.tv",
            "https://steamcharts.com",
            "https://steamdb.info",
            "https://steamstat.us",
            "https://discord.com",
            "https://twitch.tv",
            "https://epicgames.com",
            "https://ea.com",
            "https://ubisoft.com",
            "https://blizzard.com",
            "https://riotgames.com",
            "https://valvesoftware.com",
            "https://reddit.com/r/steam",
            "https://steamcommunity.com/discussions",
            "https://steamcommunity.com/market",
            "https://steamcommunity.com/workshop",
            "https://steamcommunity.com/groups"
        ]
        return legitimate_gaming

    def _get_general_legitimate_urls(self) -> List[str]:
        """Get general legitimate URLs"""
        general_legitimate = [
            "https://google.com",
            "https://youtube.com",
            "https://github.com",
            "https://wikipedia.org",
            "https://amazon.com",
            "https://facebook.com",
            "https://twitter.com",
            "https://instagram.com",
            "https://linkedin.com",
            "https://netflix.com",
            "https://microsoft.com",
            "https://apple.com",
            "https://stackoverflow.com",
            "https://reddit.com",
            "https://medium.com",
            "https://bbc.com",
            "https://cnn.com",
            "https://nytimes.com",
            "https://washingtonpost.com",
            "https://theguardian.com"
        ]
        return general_legitimate

    def _get_steam_urls(self) -> List[str]:
        """Get specific Steam platform URLs"""
        steam_urls = [
            "https://steamcommunity.com/id/",
            "https://steamcommunity.com/profiles/",
            "https://steamcommunity.com/games/",
            "https://steamcommunity.com/sharedfiles/filedetails/",
            "https://steamcommunity.com/workshop/browse/",
            "https://steamcommunity.com/market/listings/",
            "https://steamcommunity.com/tradeoffer/",
            "https://steamcommunity.com/inventory/",
            "https://steamcommunity.com/chat/",
            "https://steamcommunity.com/discussions/forum/",
            "https://store.steampowered.com/app/",
            "https://store.steampowered.com/bundle/",
            "https://store.steampowered.com/sale/",
            "https://store.steampowered.com/wishlist/",
            "https://store.steampowered.com/cart/",
            "https://help.steampowered.com/en/faqs/",
            "https://help.steampowered.com/en/accountdata/",
            "https://help.steampowered.com/en/contact/",
            "https://steam.tv/watch/",
            "https://steam.tv/broadcast/"
        ]
        return steam_urls

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            # Simple domain extraction
            if '://' in url:
                domain = url.split('://')[1].split('/')[0]
            else:
                domain = url.split('/')[0]
            return domain
        except:
            return url

def main():
    """Main execution function"""
    logger.info("Starting G65 Phishing Detection Data Collection")

    # Initialize collector
    collector = PhishingDataCollector(data_dir="./data")

    try:
        # Fetch datasets
        phishing_df = collector.fetch_phishing_datasets()
        legitimate_df = collector.fetch_legitimate_datasets()

        # Combine and prepare
        combined_df = collector.combine_and_prepare(phishing_df, legitimate_df)

        # Create train/test split
        train_df, test_df = collector.create_train_test_split(combined_df, test_size=0.3)

        logger.info("=" * 60)
        logger.info("DATA COLLECTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total samples: {len(combined_df)}")
        logger.info(f"  - Training: {len(train_df)}")
        logger.info(f"  - Testing: {len(test_df)}")
        logger.info(f"Class distribution:")
        logger.info(f"  - Phishing: {combined_df['is_phishing'].sum()}")
        logger.info(f"  - Legitimate: {len(combined_df) - combined_df['is_phishing'].sum()}")
        logger.info("")
        logger.info("Files created:")
        logger.info(f"  - data/raw/phishing_urls.csv")
        logger.info(f"  - data/raw/legitimate_urls.csv")
        logger.info(f"  - data/processed/full_dataset.csv")
        logger.info(f"  - data/processed/train_dataset.csv")
        logger.info(f"  - data/processed/test_dataset.csv")

        # Show sample of data
        logger.info("")
        logger.info("Sample data (first 5 rows):")
        print(combined_df.head())

        return combined_df, train_df, test_df

    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        raise

if __name__ == "__main__":
    df, train_df, test_df = main()
