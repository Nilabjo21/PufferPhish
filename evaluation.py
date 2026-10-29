#!/usr/bin/env python3
"""
Evaluation and Analysis for G65 Phishing Detection Framework
Generates comprehensive evaluation metrics, confusion matrices, and error analysis.
"""

import pandas as pd
import numpy as np
import json
import pickle
import logging
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from feature_extraction import get_numeric_feature_columns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.load_models()

    def load_models(self):
        """Load trained models and scalers"""
        try:
            import pickle
            import os

            # Load scalers
            for scaler_file in os.listdir(self.models_dir):
                if scaler_file.startswith('scaler_'):
                    scaler_name = scaler_file.replace('scaler_', '').replace('.pkl', '')
                    with open(os.path.join(self.models_dir, scaler_file), 'rb') as f:
                        self.scalers[scaler_name] = pickle.load(f)
                    logger.info(f"✓ Loaded scaler: {scaler_name}")

            # Load models
            for model_file in os.listdir(self.models_dir):
                if model_file.endswith('.pkl') and not model_file.startswith('scaler_'):
                    model_name = model_file.replace('.pkl', '')
                    with open(os.path.join(self.models_dir, model_file), 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    logger.info(f"✓ Loaded model: {model_name}")

            # Load results
            results_file = os.path.join(self.models_dir, "training_results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    self.results = json.load(f)
                logger.info("✓ Loaded training results")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")

    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray,
                      feature_names: List[str] = None) -> Dict[str, Any]:
        """Evaluate a single model"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return {}

        model = self.models[model_name]
        logger.info(f"Evaluating {model_name}...")

        # Scale features if needed
        if model_name in ['logistic_regression', 'svm']:
            scaler_name = 'logistic' if model_name == 'logistic_regression' else 'svm'
            if scaler_name in self.scalers:
                X_test_scaled = self.scalers[scaler_name].transform(X_test)
                X_eval = X_test_scaled
            else:
                X_eval = X_test
        else:
            X_eval = X_test

        # Generate predictions
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_eval)[:, 1]
            else:
                y_pred_proba = model.predict(X_eval)

            y_pred = (y_pred_proba >= 0.5).astype(int)

            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Calculate detailed metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            tn, fp, fn, tp = cm.ravel()

            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)

            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                feature_importance = np.abs(model.coef_[0])

            # Compile evaluation results
            eval_results = {
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
                'roc_curve': {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': float(roc_auc)
                },
                'pr_curve': {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'auc': float(pr_auc)
                },
                'metrics': {
                    'accuracy': float(report['accuracy']),
                    'precision_0': float(report['0']['precision']),
                    'recall_0': float(report['0']['recall']),
                    'f1_0': float(report['0']['f1-score']),
                    'precision_1': float(report['1']['precision']),
                    'recall_1': float(report['1']['recall']),
                    'f1_1': float(report['1']['f1-score']),
                    'tp': int(tp),
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn),
                    'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
                    'fnr': float(fn / (fn + tp)) if (fn + tp) > 0 else 0
                },
                'feature_importance': feature_importance.tolist() if feature_importance is not None else None
            }

            # Save sample predictions for error analysis
            eval_results['sample_predictions'] = self.get_error_samples(
                model_name, X_test, y_test, y_pred, y_pred_proba, feature_names
            )

            logger.info(f"✓ Evaluated {model_name} (Accuracy: {report['accuracy']:.4f})")
            return eval_results

        except Exception as e:
            logger.error(f"Evaluation failed for {model_name}: {e}")
            return {}

    def get_error_samples(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray,
                         y_pred: np.ndarray, y_pred_proba: np.ndarray,
                         feature_names: List[str] = None) -> Dict[str, Any]:
        """Get examples of false positives and false negatives"""
        # Get indices of errors
        fp_indices = np.where((y_pred == 1) & (y_test == 0))[0]
        fn_indices = np.where((y_pred == 0) & (y_test == 1))[0]

        # Sample up to 3 of each error type
        fp_samples = []
        fn_samples = []

        if len(fp_indices) > 0:
            sample_size = min(3, len(fp_indices))
            for idx in fp_indices[:sample_size]:
                sample = {
                    'index': int(idx),
                    'predicted': int(y_pred[idx]),
                    'actual': int(y_test[idx]),
                    'probability': float(y_pred_proba[idx]),
                    'features': X_test[idx].tolist() if len(X_test.shape) > 1 else [float(X_test[idx])]
                }
                if feature_names is not None:
                    sample['feature_names'] = feature_names
                fp_samples.append(sample)

        if len(fn_indices) > 0:
            sample_size = min(3, len(fn_indices))
            for idx in fn_indices[:sample_size]:
                sample = {
                    'index': int(idx),
                    'predicted': int(y_pred[idx]),
                    'actual': int(y_test[idx]),
                    'probability': float(y_pred_proba[idx]),
                    'features': X_test[idx].tolist() if len(X_test.shape) > 1 else [float(X_test[idx])]
                }
                if feature_names is not None:
                    sample['feature_names'] = feature_names
                fn_samples.append(sample)

        return {
            'false_positives': fp_samples,
            'false_negatives': fn_samples,
            'fp_count': len(fp_indices),
            'fn_count': len(fn_indices)
        }

    def plot_confusion_matrices(self, evaluations: Dict[str, Dict[str, Any]],
                               output_dir: str = "./evaluation_plots"):
        """Plot confusion matrices for all models"""
        import os
        import matplotlib

        output_path = os.path.join(output_dir, "confusion_matrices.png")
        os.makedirs(output_dir, exist_ok=True)

        n_models = len(evaluations)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_models == 1:
            axes = np.array([axes])

        axes = axes.flatten()

        for idx, (model_name, eval_data) in enumerate(evaluations.items()):
            ax = axes[idx]
            cm = np.array(eval_data['confusion_matrix'])

            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       cbar_kws={'label': 'Count'})

            ax.set_title(f"{model_name.replace('_', ' ').title()}", fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_xticklabels(['Legit (0)', 'Phishing (1)'])
            ax.set_yticklabels(['Legit (0)', 'Phishing (1)'])

            # Add accuracy to title
            accuracy = eval_data['metrics']['accuracy']
            ax.set_title(f"{model_name.replace('_', ' ').title()}\nAccuracy: {accuracy:.4f}",
                        fontsize=12, fontweight='bold')

        # Hide empty subplots
        for idx in range(len(evaluations), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved confusion matrices to {output_path}")

    def plot_roc_curves(self, evaluations: Dict[str, Dict[str, Any]],
                       output_dir: str = "./evaluation_plots"):
        """Plot ROC curves for all models"""
        import os

        output_path = os.path.join(output_dir, "roc_curves.png")
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(10, 8))

        # Plot ROC curve for each model
        for model_name, eval_data in evaluations.items():
            roc_data = eval_data['roc_curve']
            fpr = np.array(roc_data['fpr'])
            tpr = np.array(roc_data['tpr'])
            auc_score = roc_data['auc']

            plt.plot(fpr, tpr, label=f"{model_name.replace('_', ' ').title()} (AUC = {auc_score:.4f})",
                    linewidth=2)

        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved ROC curves to {output_path}")

    def plot_precision_recall_curves(self, evaluations: Dict[str, Dict[str, Any]],
                                    output_dir: str = "./evaluation_plots"):
        """Plot Precision-Recall curves for all models"""
        import os

        output_path = os.path.join(output_dir, "precision_recall_curves.png")
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(10, 8))

        # Plot PR curve for each model
        for model_name, eval_data in evaluations.items():
            pr_data = eval_data['pr_curve']
            precision = np.array(pr_data['precision'])
            recall = np.array(pr_data['recall'])
            pr_auc = pr_data['auc']

            plt.plot(recall, precision, label=f"{model_name.replace('_', ' ').title()} (AUC = {pr_auc:.4f})",
                    linewidth=2)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold')
        plt.legend(loc="upper right", fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved Precision-Recall curves to {output_path}")

    def plot_metrics_comparison(self, evaluations: Dict[str, Dict[str, Any]],
                               output_dir: str = "./evaluation_plots"):
        """Create bar chart comparing key metrics across models"""
        import os

        output_path = os.path.join(output_dir, "metrics_comparison.png")
        os.makedirs(output_dir, exist_ok=True)

        metrics_to_plot = ['accuracy', 'precision_1', 'recall_1', 'f1_1', 'fpr', 'fnr']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'FPR', 'FNR']

        n_metrics = len(metrics_to_plot)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        model_names = [name.replace('_', ' ').title() for name in evaluations.keys()]
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))

        for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            ax = axes[idx]
            metric_values = []

            for model_name, eval_data in evaluations.items():
                metric_value = eval_data['metrics'][metric]
                metric_values.append(metric_value)

            bars = ax.bar(model_names, metric_values, color=colors)
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.set_ylabel('Score' if metric != 'training_time' else 'Seconds')
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=8)

            # Set y-axis limits
            if metric in ['accuracy', 'precision_1', 'recall_1', 'f1_1']:
                ax.set_ylim([0, 1.1])
            else:
                ax.set_ylim([0, max(metric_values) * 1.2])

        # Hide empty subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Model Metrics Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved metrics comparison to {output_path}")

    def generate_evaluation_report(self, evaluations: Dict[str, Dict[str, Any]],
                                  output_dir: str = "./evaluation_reports"):
        """Generate comprehensive evaluation report"""
        import os
        from datetime import datetime

        os.makedirs(output_dir, exist_ok=True)

        report_data = {
            'timestamp': datetime.now().isoformat(),
            'model_count': len(evaluations),
            'evaluations': evaluations
        }

        # Save detailed JSON report
        json_path = os.path.join(output_dir, "detailed_evaluation.json")
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        # Generate summary report
        summary_data = {}
        for model_name, eval_data in evaluations.items():
            summary_data[model_name] = {
                'accuracy': eval_data['metrics']['accuracy'],
                'precision': eval_data['metrics']['precision_1'],
                'recall': eval_data['metrics']['recall_1'],
                'f1_score': eval_data['metrics']['f1_1'],
                'roc_auc': eval_data['roc_curve']['auc'],
                'pr_auc': eval_data['pr_curve']['auc'],
                'fpr': eval_data['metrics']['fpr'],
                'fnr': eval_data['metrics']['fnr'],
                'tp': eval_data['metrics']['tp'],
                'tn': eval_data['metrics']['tn'],
                'fp': eval_data['metrics']['fp'],
                'fn': eval_data['metrics']['fn'],
                'error_samples': {
                    'fp_count': eval_data['sample_predictions']['fp_count'],
                    'fn_count': eval_data['sample_predictions']['fn_count']
                }
            }

        summary_path = os.path.join(output_dir, "evaluation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)

        # Create human-readable report
        txt_path = os.path.join(output_dir, "evaluation_report.txt")
        with open(txt_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("G65 PHISHING DETECTION EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models Evaluated: {len(evaluations)}\n\n")

            for model_name, eval_data in evaluations.items():
                f.write("-" * 60 + "\n")
                f.write(f"MODEL: {model_name.upper()}\n")
                f.write("-" * 60 + "\n\n")

                metrics = eval_data['metrics']
                f.write("Performance Metrics:\n")
                f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['precision_1']:.4f}\n")
                f.write(f"  Recall:    {metrics['recall_1']:.4f}\n")
                f.write(f"  F1-Score:  {metrics['f1_1']:.4f}\n")
                f.write(f"  ROC-AUC:   {eval_data['roc_curve']['auc']:.4f}\n")
                f.write(f"  PR-AUC:    {eval_data['pr_curve']['auc']:.4f}\n\n")

                f.write("Confusion Matrix:\n")
                cm = eval_data['confusion_matrix']
                f.write(f"  True Negatives:  {metrics['tn']}\n")
                f.write(f"  False Positives: {metrics['fp']}\n")
                f.write(f"  False Negatives: {metrics['fn']}\n")
                f.write(f"  True Positives:  {metrics['tp']}\n\n")

                f.write("Error Rates:\n")
                f.write(f"  False Positive Rate: {metrics['fpr']:.4f}\n")
                f.write(f"  False Negative Rate: {metrics['fnr']:.4f}\n\n")

                # Error samples
                samples = eval_data['sample_predictions']
                f.write(f"Error Analysis:\n")
                f.write(f"  False Positives: {samples['fp_count']} samples\n")
                f.write(f"  False Negatives: {samples['fn_count']} samples\n\n")

        logger.info(f"✓ Generated evaluation reports in {output_dir}")
        logger.info(f"  - Detailed JSON: {json_path}")
        logger.info(f"  - Summary JSON: {summary_path}")
        logger.info(f"  - Text Report: {txt_path}")

        return summary_data


def main():
    """Main execution function"""
    logger.info("Starting G65 Model Evaluation")

    try:
        # Load test data
        logger.info("Loading test data...")
        test_df = pd.read_csv("./data/processed/test_dataset.csv")

        # Get feature names
        feature_columns = get_numeric_feature_columns(test_df)

        X_test = test_df[feature_columns].values
        y_test = test_df['is_phishing'].values

        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Feature count: {len(feature_columns)}")

        # Initialize evaluator
        evaluator = ModelEvaluator(models_dir="./models")

        if not evaluator.models:
            logger.error("No models found. Please run model_training.py first.")
            return

        # Evaluate each model
        evaluations = {}
        for model_name in evaluator.models.keys():
            eval_results = evaluator.evaluate_model(
                model_name, X_test, y_test, feature_names=feature_columns
            )
            if eval_results:
                evaluations[model_name] = eval_results

        if not evaluations:
            logger.error("No evaluations completed successfully.")
            return

        # Generate plots
        evaluator.plot_confusion_matrices(evaluations)
        evaluator.plot_roc_curves(evaluations)
        evaluator.plot_precision_recall_curves(evaluations)
        evaluator.plot_metrics_comparison(evaluations)

        # Generate comprehensive report
        summary = evaluator.generate_evaluation_report(evaluations)

        # Print summary to console
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)

        for model_name, metrics in summary.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy:   {metrics['accuracy']:.4f}")
            print(f"  Precision:  {metrics['precision']:.4f}")
            print(f"  Recall:     {metrics['recall']:.4f}")
            print(f"  F1-Score:   {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
            print(f"  FPR/FNR:    {metrics['fpr']:.4f}/{metrics['fnr']:.4f}")
            print(f"  TP/TN/FP/FN: {metrics['tp']}/{metrics['tn']}/{metrics['fp']}/{metrics['fn']}")

        # Find best model
        best_model = max(summary.items(), key=lambda x: x[1]['f1_score'])
        print("\n" + "=" * 80)
        print("BEST MODEL")
        print("=" * 80)
        print(f"Model: {best_model[0]}")
        print(f"F1-Score: {best_model[1]['f1_score']:.4f}")
        print(f"Accuracy: {best_model[1]['accuracy']:.4f}")

        logger.info("✓ Evaluation complete!")
        logger.info("✓ Reports saved to ./evaluation_reports/")
        logger.info("✓ Plots saved to ./evaluation_plots/")

        return evaluations, summary

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    evaluations, summary = main()
