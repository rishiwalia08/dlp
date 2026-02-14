"""
Prompt templates for LLM reasoning.

Defines structured prompts for:
- Attack explanation
- Risk assessment
- False positive analysis
- Response suggestions
"""

from langchain_core.prompts import PromptTemplate


# Main explanation prompt template
EXPLANATION_PROMPT_TEMPLATE = """You are a cybersecurity expert analyzing network intrusion detection results.

The deep learning model has detected the following:

Attack Type: {attack_type}
Prediction Confidence: {confidence:.2%}
Risk Score: {risk_score:.4f}
Risk Severity: {severity}

Top Contributing Features (from SHAP analysis):
{top_features}

Based on this information, provide a structured analysis:

1. EXPLANATION: Why did the model predict this attack type? Consider the top contributing features and their typical association with this attack pattern.

2. RISK ASSESSMENT: Evaluate the risk severity ({severity}) considering both the model's confidence and the attack type's potential impact.

3. FALSE POSITIVE ANALYSIS: Under what conditions could this be a false positive? Consider edge cases and benign scenarios that might trigger similar patterns.

4. RECOMMENDED RESPONSE: What actions should be taken? Consider the severity level and provide specific, actionable recommendations.

Provide a clear, concise analysis suitable for security operations personnel."""


# Simplified prompt for faster inference
SIMPLE_EXPLANATION_PROMPT = """Cybersecurity Analysis:

Attack: {attack_type} (Confidence: {confidence:.1%})
Risk: {severity} (Score: {risk_score:.2f})
Key Features: {top_features_simple}

Analyze:
1. Why this attack was predicted
2. Risk level justification
3. Possible false positive scenarios
4. Recommended actions

Be concise and actionable."""


def create_explanation_prompt(detailed=True):
    """
    Create a prompt template for attack explanation.
    
    Args:
        detailed: If True, use detailed prompt; else use simple prompt
        
    Returns:
        PromptTemplate: LangChain prompt template
    """
    if detailed:
        template = EXPLANATION_PROMPT_TEMPLATE
        input_vars = ["attack_type", "confidence", "risk_score", "severity", "top_features"]
    else:
        template = SIMPLE_EXPLANATION_PROMPT
        input_vars = ["attack_type", "confidence", "risk_score", "severity", "top_features_simple"]
    
    return PromptTemplate(
        template=template,
        input_variables=input_vars
    )


def format_top_features(top_features, max_features=5):
    """
    Format top features for prompt inclusion.
    
    Args:
        top_features: List of feature dictionaries from SHAP
        max_features: Maximum number of features to include
        
    Returns:
        str: Formatted feature list
    """
    features_list = []
    for i, feature in enumerate(top_features[:max_features], 1):
        feature_name = feature['feature_name']
        shap_value = feature['shap_value']
        impact = "positive" if shap_value > 0 else "negative"
        
        features_list.append(
            f"{i}. {feature_name} (SHAP: {shap_value:.4f}, {impact} contribution)"
        )
    
    return "\n".join(features_list)


def format_top_features_simple(top_features, max_features=3):
    """
    Format top features in a simple, concise format.
    
    Args:
        top_features: List of feature dictionaries from SHAP
        max_features: Maximum number of features to include
        
    Returns:
        str: Comma-separated feature names
    """
    feature_names = [f['feature_name'] for f in top_features[:max_features]]
    return ", ".join(feature_names)


if __name__ == "__main__":
    # Test prompt creation
    print("Testing prompt templates...")
    
    prompt = create_explanation_prompt(detailed=True)
    print("\n=== Detailed Prompt Template ===")
    print(prompt.template)
    
    simple_prompt = create_explanation_prompt(detailed=False)
    print("\n=== Simple Prompt Template ===")
    print(simple_prompt.template)
    
    # Test feature formatting
    test_features = [
        {'feature_name': 'duration', 'shap_value': 2.5},
        {'feature_name': 'src_bytes', 'shap_value': -1.8},
        {'feature_name': 'dst_bytes', 'shap_value': 1.2}
    ]
    
    print("\n=== Formatted Features ===")
    print(format_top_features(test_features))
    print("\n=== Simple Features ===")
    print(format_top_features_simple(test_features))
