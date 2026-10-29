#!/usr/bin/env python3
"""
Model Training for G65 Phishing Detection Framework
Trains and evaluates multiple models for phishing detection.
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
from typing import Dict, Tuple, List, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from feature_extraction import get_numeric_feature_columns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0

    def prepare_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare feature matrices and labels"""
        # Drop non-feature columns
        feature_columns = get_numeric_feature_columns(train_df)
        if not feature_columns:
            raise ValueError("No numeric URL features found in the processed training data.")
        missing_columns = set(feature_columns) - set(test_df.columns)
        if missing_columns:
            raise ValueError(f"Test dataset is missing model features: {sorted(missing_columns)}")

        # Separate features and labels
        X_train = train_df[feature_columns].values
        y_train = train_df['is_phishing'].values
        X_test = test_df[feature_columns].values
        y_test = test_df['is_phishing'].values

        logger.info(f"Training features: {X_train.shape}")
        logger.info(f"Training labels: {y_train.shape}")
        logger.info(f"Testing features: {X_test.shape}")
        logger.info(f"Testing labels: {y_test.shape}")

        return X_train, y_train, X_test, y_test

    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Standardize features for models that need it"""
        if model_name not in self.scalers:
            self.scalers[model_name] = StandardScaler()
            X_train_scaled = self.scalers[model_name].fit_transform(X_train)
            X_test_scaled = self.scalers[model_name].transform(X_test)
        else:
            X_train_scaled = self.scalers[model_name].transform(X_train)
            X_test_scaled = self.scalers[model_name].transform(X_test)

        return X_train_scaled, X_test_scaled

    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train Logistic Regression model"""
        logger.info("Training Logistic Regression...")

        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, 'logistic')

        # Define model with optimized hyperparameters
        model = LogisticRegression(
            C=1.0,  # Regularization strength
            penalty='l2',  # L2 regularization
            solver='lbfgs',  # Suitable for small datasets
            max_iter=1000,
            random_state=self.random_state,
            n_jobs=-1
        )

        start_time = datetime.now()

        # Fit preprocessing within each fold to avoid cross-validation leakage.
        cv_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, penalty='l2', solver='lbfgs', max_iter=1000,
                random_state=self.random_state, n_jobs=-1
            )
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring='f1')

        # Final training
        model.fit(X_train_scaled, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        # Predictions
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['training_time'] = training_time
        metrics['cv_scores_mean'] = cv_scores.mean()
        metrics['cv_scores_std'] = cv_scores.std()
        metrics['model_size'] = self.get_model_size(model)

        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = metrics

        logger.info(f"✓ Logistic Regression trained (Accuracy: {metrics['accuracy']:.4f})")
        return metrics

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train Random Forest model"""
        logger.info("Training Random Forest...")

        # Define model with optimized hyperparameters
        model = RandomForestClassifier(
            n_estimators=100,  # Number of trees
            max_depth=10,  # Maximum depth of trees
            min_samples_split=5,  # Minimum samples to split
            min_samples_leaf=2,  # Minimum samples in leaf
            max_features='sqrt',  # Features considered for split
            bootstrap=True,
            random_state=self.random_state,
            n_jobs=-1
        )

        start_time = datetime.now()

        # Train with cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')

        # Final training
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['training_time'] = training_time
        metrics['cv_scores_mean'] = cv_scores.mean()
        metrics['cv_scores_std'] = cv_scores.std()
        metrics['model_size'] = self.get_model_size(model)
        metrics['feature_importance'] = model.feature_importances_

        self.models['random_forest'] = model
        self.results['random_forest'] = metrics

        logger.info(f"✓ Random Forest trained (Accuracy: {metrics['accuracy']:.4f})")
        return metrics

    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train SVM model"""
        logger.info("Training SVM...")

        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, 'svm')

        # Define model with optimized hyperparameters
        model = SVC(
            kernel='rbf',  # Radial basis function kernel
            C=1.0,  # Regularization parameter
            gamma='scale',  # Kernel coefficient
            probability=True,  # Enable probability estimates
            random_state=self.random_state
        )

        start_time = datetime.now()

        # Fit preprocessing within each fold to avoid cross-validation leakage.
        cv_model = make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
                random_state=self.random_state)
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring='f1')

        # Final training
        model.fit(X_train_scaled, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        # Predictions
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['training_time'] = training_time
        metrics['cv_scores_mean'] = cv_scores.mean()
        metrics['cv_scores_std'] = cv_scores.std()
        metrics['model_size'] = self.get_model_size(model)

        self.models['svm'] = model
        self.results['svm'] = metrics

        logger.info(f"✓ SVM trained (Accuracy: {metrics['accuracy']:.4f})")
        return metrics

    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train XGBoost model"""
        logger.info("Training XGBoost...")

        # Define model with optimized hyperparameters
        model = XGBClassifier(
            n_estimators=100,  # Number of trees
            max_depth=6,  # Maximum depth of trees
            learning_rate=0.1,  # Step size shrinkage
            subsample=0.8,  # Subsample ratio of training instances
            colsample_bytree=0.8,  # Subsample ratio of features
            objective='binary:logistic',  # Binary classification
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=self.random_state,
            n_jobs=-1
        )

        start_time = datetime.now()

        # Train with cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')

        # Final training
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['training_time'] = training_time
        metrics['cv_scores_mean'] = cv_scores.mean()
        metrics['cv_scores_std'] = cv_scores.std()
        metrics['model_size'] = self.get_model_size(model)
        metrics['feature_importance'] = model.feature_importances_

        self.models['xgboost'] = model
        self.results['xgboost'] = metrics

        logger.info(f"✓ XGBoost trained (Accuracy: {metrics['accuracy']:.4f})")
        return metrics

    def train_distilbert(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Train DistilBERT model for text-based phishing detection"""
        logger.info("Training DistilBERT (This may take several minutes)...")

        raise NotImplementedError(
            "DistilBERT is not trained by this URL-feature pipeline; no text model is reported."
        )

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate all performance metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0,
        }

        # Confusion matrix components
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            metrics['tn'] = int(cm[0, 0])
            metrics['fp'] = int(cm[0, 1])
            metrics['fn'] = int(cm[1, 0])
            metrics['tp'] = int(cm[1, 1])
            metrics['fpr'] = metrics['fp'] / (metrics['fp'] + metrics['tn']) if (metrics['fp'] + metrics['tn']) > 0 else 0
            metrics['fnr'] = metrics['fn'] / (metrics['fn'] + metrics['tp']) if (metrics['fn'] + metrics['tp']) > 0 else 0
        else:
            metrics.update({'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0, 'fpr': 0, 'fnr': 0})

        return metrics

    def get_model_size(self, model: Any) -> float:
        """Calculate approximate model size in KB"""
        try:
            # Try to serialize and check size
            import pickle
            import io
            buffer = io.BytesIO()
            pickle.dump(model, buffer)
            size_kb = len(buffer.getvalue()) / 1024
            return round(size_kb, 2)
        except:
            return 0.0

    def train_fusion_model(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train a simple fusion model (weighted ensemble)"""
        logger.info("Training Fusion Model (weighted ensemble)...")

        # Train individual models if not already trained
        if 'logistic_regression' not in self.models:
            self.train_logistic_regression(X_train, y_train, X_test, y_test)
        if 'random_forest' not in self.models:
            self.train_random_forest(X_train, y_train, X_test, y_test)
        if 'xgboost' not in self.models:
            self.train_xgboost(X_train, y_train, X_test, y_test)

        # Collect predictions from individual models
        predictions = []
        weights = []

        # Logistic Regression predictions (weight: 0.25)
        X_test_scaled_lr = self.scalers['logistic'].transform(X_test)
        pred_lr = self.models['logistic_regression'].predict_proba(X_test_scaled_lr)[:, 1]
        predictions.append(pred_lr)
        weights.append(0.25)

        # Random Forest predictions (weight: 0.35)
        pred_rf = self.models['random_forest'].predict_proba(X_test)[:, 1]
        predictions.append(pred_rf)
        weights.append(0.35)

        # XGBoost predictions (weight: 0.40)
        pred_xgb = self.models['xgboost'].predict_proba(X_test)[:, 1]
        predictions.append(pred_xgb)
        weights.append(0.40)

        # Weighted average fusion
        weights = np.array(weights) / sum(weights)  # Normalize
        fused_predictions = np.zeros_like(pred_lr)
        for pred, weight in zip(predictions, weights):
            fused_predictions += pred * weight

        # Convert to binary predictions
        y_pred = (fused_predictions >= 0.5).astype(int)

        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, fused_predictions)
        metrics['training_time'] = 0.0  # No training time for fusion
        metrics['model_size'] = 0.0
        metrics['weights'] = weights.tolist()
        metrics['component_models'] = ['logistic_regression', 'random_forest', 'xgboost']

        self.results['fusion'] = metrics

        logger.info(f"✓ Fusion Model trained (Accuracy: {metrics['accuracy']:.4f})")
        return metrics

    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Train all models"""
        logger.info("=" * 60)
        logger.info("Training All Models")
        logger.info("=" * 60)

        # Train traditional models
        self.train_logistic_regression(X_train, y_train, X_test, y_test)
        self.train_random_forest(X_train, y_train, X_test, y_test)
        self.train_svm(X_train, y_train, X_test, y_test)
        self.train_xgboost(X_train, y_train, X_test, y_test)

        # Train fusion model
        self.train_fusion_model(X_train, y_train, X_test, y_test)

        # Determine best model
        self.find_best_model()

        return self.results

    def find_best_model(self):
        """Find the best model based on F1 score"""
        best_score = 0
        best_model_name = None

        for model_name, metrics in self.results.items():
            if metrics['f1'] > best_score:
                best_score = metrics['f1']
                best_model_name = model_name

        self.best_model = best_model_name
        self.best_score = best_score

        logger.info(f"Best model: {best_model_name} (F1: {best_score:.4f})")

    def save_models(self, output_dir: str = "./models"):
        """Save trained models to disk"""
        import os
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save each model
        for model_name, model in self.models.items():
            model_path = output_path / f"{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"✓ Saved {model_name} to {model_path}")

        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = output_path / f"scaler_{scaler_name}.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)

        # Save results
        results_path = output_path / "training_results.json"
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for model_name, metrics in self.results.items():
            json_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    json_metrics[key] = value.tolist()
                elif isinstance(value, np.integer):
                    json_metrics[key] = int(value)
                elif isinstance(value, np.floating):
                    json_metrics[key] = float(value)
                else:
                    json_metrics[key] = value
            json_results[model_name] = json_metrics

        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        logger.info(f"✓ Saved results to {results_path}")

    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate comparison table of all models"""
        table_data = []

        for model_name, metrics in self.results.items():
            row = {
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'ROC-AUC': metrics['roc_auc'],
                'Training Time (s)': metrics['training_time'],
                'Model Size (KB)': metrics.get('model_size', 0),
                'CV F1 Mean': metrics.get('cv_scores_mean', 0),
                'CV F1 Std': metrics.get('cv_scores_std', 0),
            }
            table_data.append(row)

        df = pd.DataFrame(table_data)

        # Format numeric columns
        float_columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC',
                        'CV F1 Mean', 'CV F1 Std']
        # Use map instead of applymap for newer pandas versions
        for col in float_columns:
            if col in df.columns:
                df[col] = df[col].map(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

        return df

    def plot_comparison(self, output_path: str = "./model_comparison.png"):
        """Create comparison plot of model performances"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'training_time']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Training Time (s)']

        for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
            ax = axes[idx]
            model_names = []
            metric_values = []

            for model_name, metrics in self.results.items():
                model_names.append(model_name.replace('_', ' ').title())
                metric_values.append(metrics[metric])

            bars = ax.bar(model_names, metric_values)
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.set_ylabel('Score' if metric != 'training_time' else 'Seconds')
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}' if metric != 'training_time' else f'{height:.2f}s',
                       ha='center', va='bottom', fontsize=8)

            # Highlight best model for accuracy metric
            if metric == 'accuracy' and self.best_model:
                best_idx = model_names.index(self.best_model.replace('_', ' ').title())
                bars[best_idx].set_color('green')

        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved comparison plot to {output_path}")


def main():
    """Main execution function"""
    logger.info("Starting G65 Model Training")

    try:
        # Load the processed data
        logger.info("Loading processed data...")
        train_df = pd.read_csv("./data/processed/train_dataset.csv")
        test_df = pd.read_csv("./data/processed/test_dataset.csv")

        logger.info(f"Training data shape: {train_df.shape}")
        logger.info(f"Testing data shape: {test_df.shape}")

        # Initialize trainer
        trainer = ModelTrainer(random_state=42)

        # Prepare data
        X_train, y_train, X_test, y_test = trainer.prepare_data(train_df, test_df)

        # Train all models
        results = trainer.train_all_models(X_train, y_train, X_test, y_test)

        # Generate comparison table
        comparison_table = trainer.generate_comparison_table()
        print("\n" + "=" * 80)
        print("MODEL COMPARISON TABLE")
        print("=" * 80)
        print(comparison_table.to_string(index=False))

        # Print detailed results
        print("\n" + "=" * 80)
        print("DETAILED RESULTS")
        print("=" * 80)
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"  Training Time: {metrics['training_time']:.2f}s")
            if 'tp' in metrics:
                print(f"  TP/FP/TN/FN: {metrics['tp']}/{metrics['fp']}/{metrics['tn']}/{metrics['fn']}")
                print(f"  FPR/FNR:     {metrics['fpr']:.4f}/{metrics['fnr']:.4f}")

        # Save models and results
        trainer.save_models(output_dir="./models")

        # Generate comparison plot
        trainer.plot_comparison(output_path="./model_comparison.png")

        # Display best model
        print("\n" + "=" * 80)
        print("BEST MODEL SUMMARY")
        print("=" * 80)
        print(f"Best Model: {trainer.best_model}")
        print(f"Best F1 Score: {trainer.best_score:.4f}")

        # Create results summary for report
        summary = {
            "timestamp": datetime.now().isoformat(),
            "best_model": trainer.best_model,
            "best_f1_score": float(trainer.best_score),
            "total_samples": len(train_df) + len(test_df),
            "training_samples": len(train_df),
            "testing_samples": len(test_df),
            "feature_count": X_train.shape[1],
            "data_provenance": "Synthetic educational URL corpus; no verified live phishing feed was downloaded.",
            "split_strategy": "70/30 class-wise grouped split by exact hostname; no hostname overlaps across train and test.",
            "results": results
        }

        with open("./training_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("✓ Model training complete!")
        logger.info("✓ Results saved to ./training_summary.json")
        logger.info("✓ Models saved to ./models/ directory")

        return results, trainer

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise


if __name__ == "__main__":
    results, trainer = main()
