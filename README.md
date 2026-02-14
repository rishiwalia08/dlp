# Autonomous Explainable Intrusion Detection System

A research-grade intrusion detection system combining Deep Learning, SHAP explainability, LLM reasoning, and autonomous decision-making.

## 🎯 Project Overview

This project implements an end-to-end IDS that maintains clear separation of concerns:

- **Deep Learning (1D CNN)**: Sole prediction engine for attack classification
- **SHAP**: Feature-level explainability for model predictions
- **LLM (Ollama)**: Natural language reasoning and interpretation
- **Decision Agent**: Autonomous response action execution

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    IDS PIPELINE FLOW                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Data Loading (kagglehub)                                    │
│     └─> Preprocessing → Encoding → Scaling → Reshaping         │
│                                                                  │
│  2. Deep Learning Prediction (1D CNN)                           │
│     └─> Attack Classification + Confidence Scores               │
│                                                                  │
│  3. SHAP Explainability                                         │
│     └─> Top Contributing Features + Importance Values           │
│                                                                  │
│  4. Risk Scoring                                                │
│     └─> Weighted Score (DL + SHAP + Severity)                   │
│                                                                  │
│  5. LLM Reasoning (Ollama)                                      │
│     └─> Natural Language Explanation + Risk Assessment          │
│                                                                  │
│  6. Decision Agent                                              │
│     └─> Automated Response Actions                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📋 Requirements

### System Requirements
- Python 3.8+
- Ollama (for LLM reasoning)
- 8GB+ RAM recommended
- GPU optional (speeds up training)

### Python Dependencies
```bash
pip install -r requirements.txt
```

### Ollama Setup
```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama service
ollama serve

# Pull the model (in a new terminal)
ollama pull llama3.2
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd ids-explainable-agent
pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
# Process 5 samples with Ollama LLM
python pipeline.py --samples 5

# Process 10 samples without LLM (faster)
python pipeline.py --samples 10 --no-ollama

# Force model retraining
python pipeline.py --samples 5 --retrain
```

### 3. View Results
Results are saved to `ids_results_TIMESTAMP.json` with structured output for each sample.

## 📁 Project Structure

```
ids-explainable-agent/
├── data/
│   ├── __init__.py
│   └── loader.py              # Dataset loading and preprocessing
├── models/
│   ├── __init__.py
│   ├── cnn_model.py           # 1D CNN architecture
│   └── trainer.py             # Training and evaluation
├── explainability/
│   ├── __init__.py
│   ├── shap_explainer.py      # SHAP integration
│   └── risk_scorer.py         # Risk score computation
├── llm/
│   ├── __init__.py
│   ├── ollama_client.py       # Ollama LLM interface
│   └── prompts.py             # Prompt templates
├── agent/
│   ├── __init__.py
│   └── decision_agent.py      # Autonomous decision agent
├── pipeline.py                # End-to-end orchestration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔬 Technical Details

### 1. Data Pipeline
- **Dataset**: Kaggle IDS intrusion dataset (solarmainframe/ids-intrusion-csv)
- **Preprocessing**:
  - Numeric conversion with `pd.to_numeric(errors='coerce')`
  - NaN handling (drop fully-NaN columns and rows)
  - Label encoding with `LabelEncoder`
  - Feature scaling with `StandardScaler`
  - Reshaping for 1D CNN: `(samples, features, 1)`

### 2. Deep Learning Model
- **Architecture**: 1D CNN with residual connections
- **Components**:
  - Modular CNN blocks (Conv1D → BatchNorm → ReLU → Dropout)
  - Residual connections for gradient flow
  - 5 stacked blocks with increasing filters (64 → 128 → 256 → 256 → 512)
  - Global Average Pooling
  - Dense layers with BatchNorm
  - Softmax output for multi-class classification
- **Training**:
  - Adam optimizer
  - Early stopping (patience=10)
  - Learning rate reduction (factor=0.5, patience=5)
  - Model checkpointing

### 3. SHAP Explainability
- **Method**: GradientExplainer (optimized for deep learning)
- **Output**:
  - Top-k contributing features per prediction
  - SHAP values (positive/negative contributions)
  - Total absolute SHAP score for anomaly detection

### 4. Risk Scoring
- **Formula**: `risk_score = (confidence × 0.4) + (shap_score × 0.3) + (severity × 0.3)`
- **Components**:
  - DL confidence: Softmax probability
  - SHAP score: Normalized sum of absolute SHAP values
  - Severity weight: Domain-specific attack severity (e.g., DoS=0.9, Probe=0.6)
- **Categories**: Low (<0.3), Medium (0.3-0.6), High (0.6-0.85), Critical (≥0.85)

### 5. LLM Reasoning
- **Model**: Ollama (default: llama3.2)
- **Role**: Explain and reason about DL+SHAP outputs (does NOT predict)
- **Output**:
  - Why the model predicted this attack
  - Risk severity justification
  - Possible false positive conditions
  - Recommended response actions

### 6. Decision Agent
- **Type**: Rule-based autonomous agent
- **Logic**:
  - **Low risk** (<0.3): Log only
  - **Medium risk** (0.3-0.6): Enhanced monitoring + alert SOC
  - **High risk** (0.6-0.85): Rate limiting + alert incident response
  - **Critical risk** (≥0.85): Quarantine IP + escalate
- **Actions**: Simulated (print/log statements)

## 📊 Output Format

Each processed sample returns a structured dictionary:

```json
{
  "sample_index": 0,
  "true_label": "DoS",
  "attack_type": "DoS",
  "confidence": 0.94,
  "top_features": [
    {"name": "duration", "shap_value": 2.5},
    {"name": "src_bytes", "shap_value": -1.8}
  ],
  "llm_explanation": "The model predicted DoS attack because...",
  "risk_score": 0.87,
  "severity": "Critical",
  "agent_decision": "Simulated block: quarantine IP + escalate",
  "action_taken": [
    "[BLOCK] Critical DoS attack blocked",
    "[BLOCK] Source IP quarantined in firewall",
    "[BLOCK] Senior security analyst alerted"
  ]
}
```

## 🎓 Academic Justification

### Why This Design?

1. **DL as Sole Predictor**: Deep learning excels at pattern recognition in high-dimensional network flow data. The 1D CNN architecture is specifically designed for sequential/tabular data.

2. **SHAP for Explainability**: SHAP provides theoretically sound feature attributions based on Shapley values, making the model's decisions interpretable.

3. **LLM for Reasoning**: LLMs bridge the gap between technical SHAP values and human-understandable explanations, making the system accessible to security analysts.

4. **Sequential Pipeline**: Clear separation ensures each component has a single responsibility, making the system maintainable and academically defensible.

5. **Agent for Automation**: The decision agent automates routine responses while maintaining human oversight for critical decisions.

### Viva Defense Points

- **DL vs Traditional ML**: 1D CNNs capture local patterns and temporal dependencies better than traditional methods like Random Forests
- **SHAP vs LIME**: SHAP provides consistent, theoretically grounded explanations with additive feature attribution
- **Local LLM**: Ollama ensures data privacy and eliminates API dependencies
- **Rule-based Agent**: Deterministic, explainable, and auditable decision-making

## 🧪 Testing

### Test Individual Components
```bash
# Test data loader
python data/loader.py

# Test CNN model
python models/cnn_model.py

# Test SHAP explainer
python explainability/shap_explainer.py

# Test risk scorer
python explainability/risk_scorer.py

# Test Ollama client
python llm/ollama_client.py

# Test decision agent
python agent/decision_agent.py
```

### Test Full Pipeline
```bash
# Quick test (5 samples, no LLM)
python pipeline.py --samples 5 --no-ollama

# Full test (10 samples with LLM)
python pipeline.py --samples 10
```

## 📈 Performance Metrics

The system tracks:
- **Model Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class performance
- **SHAP Computation Time**: Explainability overhead
- **LLM Response Latency**: Reasoning time
- **End-to-End Throughput**: Samples processed per second

## 🔧 Customization

### Change Ollama Model
```bash
python pipeline.py --ollama-model mistral
```

### Adjust Risk Weights
Edit `explainability/risk_scorer.py`:
```python
RiskScorer(
    confidence_weight=0.5,  # Increase DL confidence weight
    shap_weight=0.3,
    severity_weight=0.2
)
```

### Modify Decision Rules
Edit `agent/decision_agent.py`:
```python
if risk_score < 0.4:  # Adjust threshold
    action = "Custom action"
```

## 🐛 Troubleshooting

### Ollama Connection Error
```bash
# Ensure Ollama is running
ollama serve

# Check if model is available
ollama list

# Pull model if missing
ollama pull llama3.2
```

### Memory Issues
```bash
# Reduce background samples for SHAP
python pipeline.py --samples 5  # Process fewer samples

# Or disable LLM
python pipeline.py --no-ollama
```

### Dataset Download Issues
```bash
# Ensure kagglehub is installed
pip install kagglehub

# Check internet connection
# Dataset will be cached after first download
```

## 📚 References

- **SHAP**: Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions"
- **1D CNN for IDS**: Kim et al. (2020). "CNN-based Network Intrusion Detection"
- **Ollama**: Local LLM inference framework
- **LangChain**: LLM application framework

## 📝 License

This is a research project for academic purposes.

## 👤 Author

Research-grade implementation for cybersecurity and AI coursework.

---

**Note**: This system is designed for research and educational purposes. For production deployment, additional hardening, testing, and validation are required.
