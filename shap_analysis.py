#!/usr/bin/env python3
"""
SHAP Analysis for G65 Phishing Detection Framework
Generates explainability visualizations using SHAP (SHapley Additive exPlanations).
"""

import pandas as pd
import numpy as np
import pickle
import json
import logging
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# SHAP imports
import shap

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.explainers = {}
        self.shap_values_cache = {}
        self.load_models()

    def load_models(self):
        """Load trained models and scalers"""
        try:
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

        except Exception as e:
            logger.error(f"Failed to load models: {e}")

    def prepare_data(self, test_df: pd.DataFrame, sample_size: int = 100) -> Tuple[np.ndarray, List[str]]:
        """Prepare test data for SHAP analysis"""
        # Get feature columns
        feature_columns = [col for col in test_df.columns
                          if col not in ['url', 'is_phishing', 'domain', 'source', 'category']]

        X_test = test_df[feature_columns].values

        # Sample data for faster computation
        if len(X_test) > sample_size:
            np.random.seed(42)
            indices = np.random.choice(len(X_test), sample_size, replace=False)
            X_sample = X_test[indices]
            logger.info(f"Sampled {sample_size} instances for SHAP analysis")
        else:
            X_sample = X_test
            logger.info(f"Using all {len(X_test)} instances for SHAP analysis")

        return X_sample, feature_columns

    def create_explainer(self, model_name: str, X_sample: np.ndarray) -> Any:
        """Create appropriate SHAP explainer for the model"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return None

        model = self.models[model_name]
        logger.info(f"Creating SHAP explainer for {model_name}...")

        try:
            # Choose explainer based on model type
            if model_name in ['logistic_regression', 'svm']:
                # Linear models
                scaler_name = 'logistic' if model_name == 'logistic_regression' else 'svm'
                if scaler_name in self.scalers:
                    X_scaled = self.scalers[scaler_name].transform(X_sample)
                    explainer = shap.LinearExplainer(model, X_scaled)
                else:
                    explainer = shap.LinearExplainer(model, X_sample)

            elif model_name in ['random_forest', 'xgboost']:
                # Tree-based models
                explainer = shap.TreeExplainer(model)

            else:
                # Default to KernelExplainer for other models
                def predict_fn(x):
                    return model.predict_proba(x)[:, 1]

                explainer = shap.KernelExplainer(predict_fn, X_sample[:10])

            self.explainers[model_name] = explainer
            logger.info(f"✓ Created {type(explainer).__name__} for {model_name}")

            return explainer

        except Exception as e:
            logger.error(f"Failed to create explainer for {model_name}: {e}")
            return None

    def calculate_shap_values(self, model_name: str, X_sample: np.ndarray) -> Tuple[Any, Any]:
        """Calculate SHAP values for a model"""
        if model_name in self.shap_values_cache:
            logger.info(f"Using cached SHAP values for {model_name}")
            return self.shap_values_cache[model_name]

        explainer = self.create_explainer(model_name, X_sample)
        if not explainer:
            return None, None

        try:
            # Scale features if needed
            if model_name in ['logistic_regression', 'svm']:
                scaler_name = 'logistic' if model_name == 'logistic_regression' else 'svm'
                if scaler_name in self.scalers:
                    X_scaled = self.scalers[scaler_name].transform(X_sample)
                    shap_values = explainer.shap_values(X_scaled)
                else:
                    shap_values = explainer.shap_values(X_sample)
            else:
                shap_values = explainer.shap_values(X_sample)

            # Get expected value
            expected_value = explainer.expected_value

            self.shap_values_cache[model_name] = (shap_values, expected_value)
            logger.info(f"✓ Calculated SHAP values for {model_name}")

            return shap_values, expected_value

        except Exception as e:
            logger.error(f"Failed to calculate SHAP values for {model_name}: {e}")
            return None, None

    def plot_summary(self, model_name: str, shap_values: Any, X_sample: np.ndarray,
                    feature_names: List[str], output_dir: str):
        """Create SHAP summary plot"""
        import os
        output_path = os.path.join(output_dir, f"{model_name}_summary.png")

        plt.figure(figsize=(12, 8))

        try:
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # For binary classification, use shap_values[1] for positive class
                shap_values_to_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_values_to_plot = shap_values

            shap.summary_plot(shap_values_to_plot, X_sample, feature_names=feature_names,
                             show=False, plot_size=None)

            plt.title(f"SHAP Summary - {model_name.replace('_', ' ').title()}", fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✓ Saved summary plot to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create summary plot for {model_name}: {e}")
            return False

    def plot_waterfall(self, model_name: str, shap_values: Any, expected_value: Any,
                      X_sample: np.ndarray, feature_names: List[str],
                      instance_idx: int, output_dir: str):
        """Create SHAP waterfall plot for a specific instance"""
        import os
        output_path = os.path.join(output_dir, f"{model_name}_waterfall_{instance_idx}.png")

        plt.figure(figsize=(10, 6))

        try:
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_values_instance = shap_values[1][instance_idx] if len(shap_values) > 1 else shap_values[0][instance_idx]
            else:
                shap_values_instance = shap_values[instance_idx]

            # Create waterfall plot
            shap.plots.waterfall(shap.Explanation(values=shap_values_instance,
                                                  base_values=expected_value,
                                                  data=X_sample[instance_idx],
                                                  feature_names=feature_names),
                                show=False, max_display=15)

            plt.title(f"Waterfall Plot - {model_name.replace('_', ' ').title()} (Instance {instance_idx})",
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✓ Saved waterfall plot to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create waterfall plot for {model_name}: {e}")
            return False

    def plot_force(self, model_name: str, shap_values: Any, expected_value: Any,
                  X_sample: np.ndarray, feature_names: List[str],
                  instance_idx: int, output_dir: str):
        """Create SHAP force plot for a specific instance"""
        import os
        output_path = os.path.join(output_dir, f"{model_name}_force_{instance_idx}.png")

        try:
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_values_instance = shap_values[1][instance_idx] if len(shap_values) > 1 else shap_values[0][instance_idx]
            else:
                shap_values_instance = shap_values[instance_idx]

            # Create force plot
            force_plot = shap.plots.force(base_value=expected_value,
                                         shap_values=shap_values_instance,
                                         features=X_sample[instance_idx],
                                         feature_names=feature_names,
                                         show=False, matplotlib=True)

            # Save the figure
            plt.title(f"Force Plot - {model_name.replace('_', ' ').title()} (Instance {instance_idx})",
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✓ Saved force plot to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create force plot for {model_name}: {e}")
            return False

    def plot_feature_importance(self, model_name: str, shap_values: Any,
                               feature_names: List[str], output_dir: str):
        """Create feature importance bar chart from SHAP values"""
        import os
        output_path = os.path.join(output_dir, f"{model_name}_feature_importance.png")

        plt.figure(figsize=(12, 8))

        try:
            # Calculate mean absolute SHAP values
            if isinstance(shap_values, list):
                shap_values_mean = np.abs(shap_values[1]).mean(axis=0) if len(shap_values) > 1 else np.abs(shap_values[0]).mean(axis=0)
            else:
                shap_values_mean = np.abs(shap_values).mean(axis=0)

            # Sort features by importance
            sorted_idx = np.argsort(shap_values_mean)[::-1]
            sorted_features = [feature_names[i] for i in sorted_idx[:20]]  # Top 20 features
            sorted_importance = shap_values_mean[sorted_idx[:20]]

            # Create bar plot
            bars = plt.barh(sorted_features, sorted_importance)
            plt.xlabel('Mean |SHAP value| (average impact on model output magnitude)', fontsize=12)
            plt.title(f'Feature Importance - {model_name.replace("_", " ").title()}', fontsize=16, fontweight='bold')
            plt.gca().invert_yaxis()  # Most important at top

            # Add value labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2,
                        f'{width:.4f}', ha='left', va='center', fontsize=9)

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✓ Saved feature importance plot to {output_path}")

            # Return feature importance data
            importance_data = {
                'features': sorted_features,
                'importance': sorted_importance.tolist()
            }
            return importance_data

        except Exception as e:
            logger.error(f"Failed to create feature importance plot for {model_name}: {e}")
            return None

    def plot_dependence(self, model_name: str, shap_values: Any, X_sample: np.ndarray,
                       feature_names: List[str], output_dir: str):
        """Create SHAP dependence plots for top features"""
        import os

        try:
            # Calculate feature importance
            if isinstance(shap_values, list):
                shap_values_for_dep = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_values_for_dep = shap_values

            shap_values_mean = np.abs(shap_values_for_dep).mean(axis=0)
            top_indices = np.argsort(shap_values_mean)[::-1][:5]  # Top 5 features

            # Create dependence plots for top features
            for idx in top_indices[:3]:  # Create plots for top 3 features
                feature_name = feature_names[idx]

                # Find interacting feature (second most correlated with SHAP values)
                correlation = []
                for j in range(len(feature_names)):
                    if j != idx:
                        corr = np.corrcoef(shap_values_for_dep[:, idx], X_sample[:, j])[0, 1]
                        correlation.append((j, abs(corr)))

                correlation.sort(key=lambda x: x[1], reverse=True)
                interaction_idx = correlation[0][0] if correlation else idx

                # Create dependence plot
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(idx, shap_values_for_dep, X_sample,
                                    feature_names=feature_names,
                                    interaction_index=interaction_idx,
                                    show=False)

                plt.title(f'Dependence Plot - {feature_name}\n({model_name.replace("_", " ").title()})',
                         fontsize=14, fontweight='bold')
                plt.tight_layout()

                output_path = os.path.join(output_dir, f"{model_name}_dependence_{feature_name}.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()

                logger.info(f"✓ Saved dependence plot to {output_path}")

        except Exception as e:
            logger.error(f"Failed to create dependence plots for {model_name}: {e}")

    def analyze_model(self, model_name: str, X_sample: np.ndarray,
                     feature_names: List[str], output_dir: str) -> Dict[str, Any]:
        """Perform complete SHAP analysis for a model"""
        import os

        logger.info(f"Performing SHAP analysis for {model_name}...")

        # Create output directory
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        # Calculate SHAP values
        shap_values, expected_value = self.calculate_shap_values(model_name, X_sample)
        if shap_values is None:
            logger.error(f"Failed to calculate SHAP values for {model_name}")
            return {}

        analysis_results = {
            'model_name': model_name,
            'expected_value': float(expected_value) if hasattr(expected_value, '__float__') else None,
            'plots_generated': [],
            'feature_importance': None
        }

        # Generate plots
        if self.plot_summary(model_name, shap_values, X_sample, feature_names, model_output_dir):
            analysis_results['plots_generated'].append('summary')

        # Generate feature importance plot
        importance_data = self.plot_feature_importance(model_name, shap_values, feature_names, model_output_dir)
        if importance_data:
            analysis_results['feature_importance'] = importance_data
            analysis_results['plots_generated'].append('feature_importance')

        # Generate sample waterfall and force plots (for 3 instances)
        for instance_idx in [0, 1, 2]:
            if instance_idx < len(X_sample):
                if self.plot_waterfall(model_name, shap_values, expected_value,
                                     X_sample, feature_names, instance_idx, model_output_dir):
                    analysis_results['plots_generated'].append(f'waterfall_{instance_idx}')

                if self.plot_force(model_name, shap_values, expected_value,
                                 X_sample, feature_names, instance_idx, model_output_dir):
                    analysis_results['plots_generated'].append(f'force_{instance_idx}')

        # Generate dependence plots
        self.plot_dependence(model_name, shap_values, X_sample, feature_names, model_output_dir)

        logger.info(f"✓ Completed SHAP analysis for {model_name}")
        return analysis_results

    def generate_comparison_report(self, all_results: Dict[str, Dict[str, Any]],
                                  output_dir: str):
        """Generate comparison report across all models"""
        import os
        from datetime import datetime

        report_data = {
            'timestamp': datetime.now().isoformat(),
            'model_count': len(all_results),
            'models_analyzed': list(all_results.keys()),
            'results': all_results
        }

        # Save JSON report
        json_path = os.path.join(output_dir, "shap_analysis_report.json")
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        # Create summary report
        summary_path = os.path.join(output_dir, "shap_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SHAP ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models Analyzed: {len(all_results)}\n\n")

            for model_name, results in all_results.items():
                f.write("-" * 60 + "\n")
                f.write(f"MODEL: {model_name.upper()}\n")
                f.write("-" * 60 + "\n\n")

                f.write("Plots Generated:\n")
                for plot in results.get('plots_generated', []):
                    f.write(f"  • {plot}\n")
                f.write("\n")

                if results.get('feature_importance'):
                    f.write("Top 5 Most Important Features:\n")
                    features = results['feature_importance']['features'][:5]
                    importance = results['feature_importance']['importance'][:5]
                    for feat, imp in zip(features, importance):
                        f.write(f"  • {feat}: {imp:.4f}\n")
                f.write("\n")

        # Create comparison plot of top features across models
        self.plot_comparison_feature_importance(all_results, output_dir)

        logger.info(f"✓ Generated SHAP reports in {output_dir}")
        logger.info(f"  - Detailed JSON: {json_path}")
        logger.info(f"  - Summary Text: {summary_path}")

    def plot_comparison_feature_importance(self, all_results: Dict[str, Dict[str, Any]],
                                          output_dir: str):
        """Create comparison plot of feature importance across models"""
        import os

        # Collect top 5 features from each model
        feature_importance_data = {}
        for model_name, results in all_results.items():
            if results.get('feature_importance'):
                features = results['feature_importance']['features'][:5]
                importance = results['feature_importance']['importance'][:5]

                for feat, imp in zip(features, importance):
                    if feat not in feature_importance_data:
                        feature_importance_data[feat] = {}
                    feature_importance_data[feat][model_name] = imp

        if not feature_importance_data:
            return

        # Create DataFrame for plotting
        df_data = []
        for feat, model_imps in feature_importance_data.items():
            for model_name, imp in model_imps.items():
                df_data.append({
                    'feature': feat,
                    'model': model_name.replace('_', ' ').title(),
                    'importance': imp
                })

        if not df_data:
            return

        df = pd.DataFrame(df_data)

        # Pivot for heatmap
        pivot_df = df.pivot(index='feature', columns='model', values='importance')
        pivot_df = pivot_df.fillna(0)

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='YlOrRd',
                   cbar_kws={'label': 'Mean |SHAP value|'})

        plt.title('Feature Importance Comparison Across Models', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()

        output_path = os.path.join(output_dir, "feature_importance_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved feature importance comparison to {output_path}")


def main():
    """Main execution function"""
    logger.info("Starting G65 SHAP Analysis")

    try:
        # Load test data
        logger.info("Loading test data...")
        test_df = pd.read_csv("./data/processed/test_dataset.csv")

        # Prepare data for SHAP analysis
        X_sample, feature_names = SHAPAnalyzer().prepare_data(test_df, sample_size=100)

        logger.info(f"Feature count: {len(feature_names)}")
        logger.info(f"Sample size for SHAP: {len(X_sample)}")

        # Initialize analyzer
        analyzer = SHAPAnalyzer(models_dir="./models")

        if not analyzer.models:
            logger.error("No models found. Please run model_training.py first.")
            return

        # Perform SHAP analysis for each model
        all_results = {}
        for model_name in analyzer.models.keys():
            results = analyzer.analyze_model(
                model_name, X_sample, feature_names, output_dir="./shap_analysis"
            )
            if results:
                all_results[model_name] = results

        if not all_results:
            logger.error("No SHAP analysis completed successfully.")
            return

        # Generate comparison report
        analyzer.generate_comparison_report(all_results, output_dir="./shap_analysis")

        # Print summary to console
        print("\n" + "=" * 80)
        print("SHAP ANALYSIS SUMMARY")
        print("=" * 80)

        for model_name, results in all_results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Plots generated: {len(results.get('plots_generated', []))}")

            if results.get('feature_importance'):
                top_features = results['feature_importance']['features'][:5]
                print(f"  Top features: {', '.join(top_features[:3])}...")

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print("SHAP visualizations saved to ./shap_analysis/")
        print("Each model has its own subdirectory with:")
        print("  • Summary plots")
        print("  • Feature importance charts")
        print("  • Waterfall plots (sample instances)")
        print("  • Force plots (sample instances)")
        print("  • Dependence plots (top features)")

        logger.info("✓ SHAP analysis complete!")
        logger.info("✓ Visualizations saved to ./shap_analysis/")

        return all_results

    except Exception as e:
        logger.error(f"SHAP analysis failed: {e}")
        raise


if __name__ == "__main__":
    all_results = main()
