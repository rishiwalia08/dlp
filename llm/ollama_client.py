"""
Ollama LLM client for reasoning and explanation.

This module:
- Interfaces with Ollama for local LLM inference
- LLM does NOT predict attacks
- LLM ONLY explains and reasons about DL + SHAP outputs
"""

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from .prompts import (
    create_explanation_prompt,
    format_top_features,
    format_top_features_simple
)


class OllamaExplainer:
    """Handles LLM-based reasoning for IDS predictions using Ollama."""
    
    def __init__(self, model_name="llama3.2", temperature=0.3, detailed_prompts=True):
        """
        Initialize Ollama explainer.
        
        Args:
            model_name: Name of Ollama model to use
            temperature: LLM temperature (lower = more deterministic)
            detailed_prompts: If True, use detailed prompts; else use simple prompts
        """
        self.model_name = model_name
        self.temperature = temperature
        self.detailed_prompts = detailed_prompts
        
        print("\n=== Initializing Ollama LLM ===")
        print(f"Model: {model_name}")
        print(f"Temperature: {temperature}")
        
        # Initialize Ollama LLM
        try:
            self.llm = Ollama(
                model=model_name,
                temperature=temperature
            )
            print("Ollama LLM initialized successfully")
        except Exception as e:
            print(f"Error initializing Ollama: {e}")
            print("Make sure Ollama is running and the model is pulled:")
            print(f"  ollama pull {model_name}")
            raise
        
        # Create prompt template
        self.prompt = create_explanation_prompt(detailed=detailed_prompts)
        
        # Create chain using modern pipe operator (| instead of LLMChain)
        self.chain = self.prompt | self.llm
    
    def explain_prediction(self, attack_type, confidence, risk_score, 
                          severity, top_features):
        """
        Generate LLM explanation for a prediction.
        
        Args:
            attack_type: Predicted attack type name
            confidence: Model confidence (0-1)
            risk_score: Computed risk score (0-1)
            severity: Risk severity category
            top_features: List of top SHAP features
            
        Returns:
            dict: Parsed LLM explanation
        """
        # Format features for prompt
        if self.detailed_prompts:
            features_formatted = format_top_features(top_features)
            inputs = {
                "attack_type": attack_type,
                "confidence": confidence,
                "risk_score": risk_score,
                "severity": severity,
                "top_features": features_formatted
            }
        else:
            features_simple = format_top_features_simple(top_features)
            inputs = {
                "attack_type": attack_type,
                "confidence": confidence,
                "risk_score": risk_score,
                "severity": severity,
                "top_features_simple": features_simple
            }
        
        # Generate explanation
        try:
            response = self.chain.invoke(inputs)
            
            # Parse response
            parsed = self._parse_response(response)
            
            return {
                'raw_explanation': response,
                'parsed_explanation': parsed,
                'attack_type': attack_type,
                'confidence': confidence,
                'risk_score': risk_score,
                'severity': severity
            }
        
        except Exception as e:
            print(f"Error generating LLM explanation: {e}")
            return {
                'raw_explanation': f"Error: {str(e)}",
                'parsed_explanation': {
                    'explanation': 'LLM explanation unavailable',
                    'risk_assessment': severity,
                    'false_positive_analysis': 'Unable to assess',
                    'recommended_response': 'Manual review required'
                },
                'attack_type': attack_type,
                'confidence': confidence,
                'risk_score': risk_score,
                'severity': severity
            }
    
    def _parse_response(self, response):
        """
        Parse LLM response into structured components.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            dict: Parsed components
        """
        # Simple parsing based on numbered sections
        sections = {
            'explanation': '',
            'risk_assessment': '',
            'false_positive_analysis': '',
            'recommended_response': ''
        }
        
        # Try to extract sections
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Identify section headers
            if 'explanation' in line_lower or line_lower.startswith('1.'):
                current_section = 'explanation'
                continue
            elif 'risk' in line_lower and 'assessment' in line_lower or line_lower.startswith('2.'):
                current_section = 'risk_assessment'
                continue
            elif 'false positive' in line_lower or line_lower.startswith('3.'):
                current_section = 'false_positive_analysis'
                continue
            elif 'recommend' in line_lower or 'response' in line_lower or line_lower.startswith('4.'):
                current_section = 'recommended_response'
                continue
            
            # Add content to current section
            if current_section and line.strip():
                sections[current_section] += line.strip() + ' '
        
        # Clean up sections
        for key in sections:
            sections[key] = sections[key].strip()
        
        # If parsing failed, use full response as explanation
        if not any(sections.values()):
            sections['explanation'] = response.strip()
        
        return sections
    
    def batch_explain(self, predictions_data):
        """
        Generate explanations for multiple predictions.
        
        Args:
            predictions_data: List of dicts with prediction info
            
        Returns:
            list: List of explanation dictionaries
        """
        print(f"\nGenerating LLM explanations for {len(predictions_data)} predictions...")
        
        explanations = []
        for i, pred_data in enumerate(predictions_data):
            explanation = self.explain_prediction(
                attack_type=pred_data['attack_type'],
                confidence=pred_data['confidence'],
                risk_score=pred_data['risk_score'],
                severity=pred_data['severity'],
                top_features=pred_data['top_features']
            )
            explanations.append(explanation)
            
            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1}/{len(predictions_data)} explanations")
        
        print("LLM explanations complete")
        return explanations


def create_ollama_explainer(model_name="llama3.2", temperature=0.3):
    """
    Convenience function to create Ollama explainer.
    
    Args:
        model_name: Name of Ollama model
        temperature: LLM temperature
        
    Returns:
        OllamaExplainer instance
    """
    return OllamaExplainer(model_name=model_name, temperature=temperature)


if __name__ == "__main__":
    # Test Ollama connection
    print("Testing Ollama LLM client...")
    
    try:
        explainer = create_ollama_explainer()
        
        # Test explanation
        test_features = [
            {'feature_name': 'duration', 'shap_value': 2.5},
            {'feature_name': 'src_bytes', 'shap_value': -1.8},
            {'feature_name': 'dst_bytes', 'shap_value': 1.2}
        ]
        
        result = explainer.explain_prediction(
            attack_type="DoS",
            confidence=0.92,
            risk_score=0.85,
            severity="High",
            top_features=test_features
        )
        
        print("\n=== Test Explanation ===")
        print(f"Attack Type: {result['attack_type']}")
        print(f"Severity: {result['severity']}")
        print(f"\nRaw Explanation:\n{result['raw_explanation']}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("\nMake sure Ollama is running:")
        print("  ollama serve")
        print("And the model is pulled:")
        print("  ollama pull llama3.2")
