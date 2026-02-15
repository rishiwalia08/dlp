"""
HuggingFace LLM client for IDS explanations.
Uses DistilGPT-2 - small model that fits in Colab GPU.
"""

from transformers import pipeline
import torch


class HuggingFaceExplainer:
    """LLM explainer using HuggingFace models."""
    
    def __init__(self, model_name="distilgpt2", temperature=0.7):
        """
        Initialize HuggingFace LLM.
        
        Args:
            model_name: HuggingFace model name (default: distilgpt2 - 82MB)
            temperature: Sampling temperature
        """
        print(f"\nLoading HuggingFace model: {model_name}...")
        
        device = 0 if torch.cuda.is_available() else -1
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device=device,
            max_length=200,
            pad_token_id=50256  # GPT-2 EOS token
        )
        self.temperature = temperature
        
        device_name = "GPU" if device == 0 else "CPU"
        print(f"✓ Model loaded on {device_name}")
    
    def _build_prompt(self, attack_type, confidence, top_features, risk_score):
        """Build prompt for LLM."""
        features_str = ", ".join([f['feature_name'] for f in top_features[:3]])
        
        prompt = f"""Network Attack Analysis:
Type: {attack_type}
Confidence: {confidence:.1%}
Risk Score: {risk_score:.1f}/10
Key Indicators: {features_str}

Explanation:"""
        return prompt
    
    def explain_prediction(self, attack_type, confidence, top_features, risk_score):
        """
        Generate natural language explanation for a prediction.
        
        Args:
            attack_type: Predicted attack type
            confidence: Model confidence
            top_features: Top contributing features from SHAP
            risk_score: Computed risk score
            
        Returns:
            str: Natural language explanation
        """
        try:
            # Build prompt
            prompt = self._build_prompt(attack_type, confidence, top_features, risk_score)
            
            # Generate explanation
            result = self.generator(
                prompt,
                max_new_tokens=50,
                temperature=self.temperature,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=50256
            )
            
            explanation = result[0]['generated_text']
            
            # Extract only the generated part (after prompt)
            explanation = explanation.replace(prompt, '').strip()
            
            # Clean up
            if not explanation:
                explanation = f"Detected {attack_type} with {confidence:.0%} confidence based on {top_features[0]['feature_name']}."
            
            return explanation
            
        except Exception as e:
            # Fallback if LLM fails
            print(f"Warning: LLM generation failed ({str(e)[:50]}...), using fallback")
            return f"Detected {attack_type} attack with {confidence:.1%} confidence. Risk: {risk_score:.1f}/10. Key indicators: {', '.join([f['feature_name'] for f in top_features[:3]])}"
    
    def explain_batch(self, predictions, shap_explanations, risk_scores):
        """
        Generate explanations for multiple predictions.
        
        Args:
            predictions: List of (attack_type, confidence) tuples
            shap_explanations: List of SHAP explanation dicts
            risk_scores: List of risk scores
            
        Returns:
            list: List of explanation strings
        """
        explanations = []
        for (attack_type, confidence), shap_exp, risk_score in zip(predictions, shap_explanations, risk_scores):
            explanation = self.explain_prediction(
                attack_type, confidence,
                shap_exp['top_features'],
                risk_score
            )
            explanations.append(explanation)
        
        return explanations


def create_huggingface_explainer(model_name="distilgpt2", temperature=0.7):
    """
    Create HuggingFace explainer instance.
    
    Args:
        model_name: HuggingFace model name
        temperature: Sampling temperature
        
    Returns:
        HuggingFaceExplainer instance
    """
    return HuggingFaceExplainer(model_name, temperature)


if __name__ == "__main__":
    print("HuggingFace explainer module loaded")
