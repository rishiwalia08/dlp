# Quick Start Guide

## Installation

```bash
cd ids-explainable-agent
chmod +x setup.sh
./setup.sh
```

The setup script will:
1. Install Python dependencies
2. Check Ollama installation
3. Run system tests

## Running the System

### Option 1: Quick Test (No LLM)
```bash
python3 pipeline.py --samples 5 --no-ollama
```
- Fastest option
- Tests DL, SHAP, risk scoring, and agent
- Skips LLM reasoning

### Option 2: Full System (With LLM)
```bash
# Make sure Ollama is running
ollama serve  # In separate terminal

# Run pipeline
python3 pipeline.py --samples 5
```
- Complete system test
- Includes LLM explanations
- Requires Ollama

### Option 3: Force Retrain
```bash
python3 pipeline.py --samples 5 --retrain
```
- Trains new model from scratch
- Takes longer (depends on dataset size)

## Expected Output

The pipeline will:
1. Download IDS dataset (first run only)
2. Preprocess data
3. Train/load CNN model
4. Process test samples through complete pipeline
5. Save results to `ids_results_TIMESTAMP.json`

## Troubleshooting

### Missing Dependencies
```bash
pip3 install -r requirements.txt
```

### Ollama Not Running
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Pull model
ollama pull llama3.2

# Terminal 3: Run pipeline
python3 pipeline.py --samples 5
```

### Memory Issues
```bash
# Process fewer samples
python3 pipeline.py --samples 3 --no-ollama
```

## Project Structure

```
ids-explainable-agent/
├── setup.sh              # Automated setup
├── test_system.py        # System tests
├── pipeline.py           # Main entry point
├── requirements.txt      # Dependencies
├── README.md            # Full documentation
├── QUICKSTART.md        # This file
├── data/                # Data pipeline
├── models/              # CNN model
├── explainability/      # SHAP + risk scoring
├── llm/                 # Ollama integration
└── agent/               # Decision agent
```

## Next Steps

1. **Test the system**: `python3 pipeline.py --samples 5 --no-ollama`
2. **Review results**: Check `ids_results_*.json`
3. **Read full docs**: See `README.md`
4. **Customize**: Modify risk weights, decision rules, etc.

## Academic Use

This system is designed for:
- Research projects
- Academic papers
- Viva defense
- Coursework demonstrations

See `README.md` for academic justifications and references.
