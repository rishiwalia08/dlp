"""
SHAP explainability module for CNN predictions.

This module uses SHAP to explain CNN predictions with robust error handling.
"""

import numpy as np
import shap
import tensorflow as tf


class SHAPExplainer:
    """Handles SHAP-based explainability for IDS CNN model."""
    
    def __init__(self, model, background_data, feature_names):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained Keras model
            background_data: Background dataset for SHAP (subset of training data)
            feature_names: List of feature names
        """
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names
        self.explainer = None
        
        print("\n=== Initializing SHAP Explainer ===")
        print(f"Background data shape: {background_data.shape}")
        print(f"Number of features: {len(feature_names)}")
        
        # Initialize SHAP GradientExplainer
        self.explainer = shap.GradientExplainer(
            model,
            background_data
        )
        
        print("SHAP GradientExplainer initialized successfully")
    
    def explain_prediction(self, sample, top_k=10):
        """
        Generate SHAP explanation for a single prediction.
        
        Args:
            sample: Single sample to explain (shape: (1, features, 1))
            top_k: Number of top features to return
            
        Returns:
            dict: Explanation with top features and their SHAP values
        """
        # Get prediction first
        prediction = self.model.predict(sample, verbose=0)
        predicted_class = int(np.argmax(prediction[0]))
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(sample)
        
        # Handle different SHAP output formats robustly
        try:
            if isinstance(shap_values, list):
                # Multi-class output: list of arrays
                if len(shap_values) > predicted_class:
                    class_shap = shap_values[predicted_class]
                else:
                    # Fallback to first class
                    class_shap = shap_values[0]
            else:
                # Single array output
                class_shap = shap_values
            
            # Extract and flatten the SHAP values
            # Handle shape: (1, features, 1) or (features,) or (1, features)
            class_shap = np.array(class_shap).squeeze()
            class_shap = np.atleast_1d(class_shap).flatten()
            
            # Ensure we don't exceed feature count
            num_features = min(len(class_shap), len(self.feature_names))
            class_shap = class_shap[:num_features]
            
        except Exception as e:
            print(f"Warning: SHAP value extraction failed: {e}")
            # Fallback: create zero array
            class_shap = np.zeros(len(self.feature_names))
        
        # Get absolute values for ranking
        abs_shap = np.abs(class_shap)
        
        # Get top-k indices (ensure we don't request more than available)
        actual_top_k = min(top_k, len(abs_shap))
        top_indices = np.argsort(abs_shap)[-actual_top_k:][::-1]
        
        # Build top features list with safe indexing
        top_features = []
        for i, idx in enumerate(top_indices):
            idx = int(idx)  # Ensure scalar
            if idx < len(self.feature_names) and idx < len(class_shap):
                top_features.append({
                    'feature_name': self.feature_names[idx],
                    'feature_index': idx,
                    'shap_value': float(class_shap[idx]),
                    'abs_shap_value': float(abs_shap[idx])
                })
        
        # Compute total absolute SHAP
        total_abs_shap = float(np.sum(abs_shap))
        
        explanation = {
            'predicted_class': predicted_class,
            'confidence': float(prediction[0][predicted_class]),
            'top_features': top_features,
            'total_abs_shap': total_abs_shap,
            'shap_values_all': class_shap.tolist()
        }
        
        return explanation
    
    def explain_batch(self, samples, top_k=10):
        """
        Generate SHAP explanations for multiple samples.
        
        Args:
            samples: Batch of samples (shape: (batch_size, features, 1))
            top_k: Number of top features to return per sample
            
        Returns:
            list: List of explanation dictionaries
        """
        print(f"\nGenerating SHAP explanations for {samples.shape[0]} samples...")
        
        explanations = []
        for i in range(samples.shape[0]):
            sample = samples[i:i+1]  # Keep batch dimension
            explanation = self.explain_prediction(sample, top_k)
            explanations.append(explanation)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{samples.shape[0]} samples")
        
        print("SHAP explanations complete")
        return explanations
    
    def get_feature_importance_summary(self, samples, max_samples=100):
        """
        Get overall feature importance across multiple samples.
        
        Args:
            samples: Samples to analyze
            max_samples: Maximum number of samples to use
            
        Returns:
            dict: Feature importance summary
        """
        n_samples = min(len(samples), max_samples)
        sample_subset = samples[:n_samples]
        
        print(f"\nComputing feature importance summary for {n_samples} samples...")
        
        try:
            # Compute SHAP values
            shap_values = self.explainer.shap_values(sample_subset)
            
            # Average absolute SHAP values
            if isinstance(shap_values, list):
                all_abs_shap = []
                for class_shap in shap_values:
                    class_shap_squeezed = class_shap.squeeze(axis=-1)
                    all_abs_shap.append(np.abs(class_shap_squeezed))
                stacked_shap = np.stack(all_abs_shap, axis=0)
                mean_abs_shap = np.mean(stacked_shap, axis=(0, 1))
            else:
                shap_squeezed = shap_values.squeeze(axis=-1)
                mean_abs_shap = np.mean(np.abs(shap_squeezed), axis=0)
            
            # Ensure correct length
            num_features = min(len(mean_abs_shap), len(self.feature_names))
            mean_abs_shap = mean_abs_shap[:num_features]
            
            # Rank features
            feature_ranking = np.argsort(mean_abs_shap)[::-1]
            
            importance_summary = {
                'feature_importance': [
                    {
                        'rank': i + 1,
                        'feature_name': self.feature_names[int(idx)],
                        'feature_index': int(idx),
                        'mean_abs_shap': float(mean_abs_shap[int(idx)])
                    }
                    for i, idx in enumerate(feature_ranking) if int(idx) < len(self.feature_names)
                ]
            }
            
            print("Feature importance summary complete")
            return importance_summary
            
        except Exception as e:
            print(f"Warning: Feature importance computation failed: {e}")
            return {'feature_importance': []}


def create_shap_explainer(model, background_data, feature_names):
    """
    Convenience function to create SHAP explainer.
    
    Args:
        model: Trained Keras model
        background_data: Background dataset for SHAP
        feature_names: List of feature names
        
    Returns:
        SHAPExplainer instance
    """
    return SHAPExplainer(model, background_data, feature_names)


if __name__ == "__main__":
    print("SHAP explainer module loaded successfully")
