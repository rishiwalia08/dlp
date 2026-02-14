#!/bin/bash

# Setup script for Autonomous Explainable IDS
# This script installs all dependencies and verifies the system

echo "=========================================="
echo "Autonomous Explainable IDS - Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo ""
echo "✓ Python dependencies installed"
echo ""

# Check if Ollama is installed
echo "Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"
    
    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✓ Ollama is running"
    else
        echo "⚠ Ollama is not running"
        echo "  Start it with: ollama serve"
    fi
    
    # Check if llama3.2 is pulled
    if ollama list | grep -q "llama3.2"; then
        echo "✓ llama3.2 model is available"
    else
        echo "⚠ llama3.2 model not found"
        echo "  Pull it with: ollama pull llama3.2"
    fi
else
    echo "⚠ Ollama is not installed"
    echo "  Install it with: brew install ollama"
    echo "  Or visit: https://ollama.ai"
fi

echo ""
echo "=========================================="
echo "Running system tests..."
echo "=========================================="
echo ""

# Run test script
python3 test_system.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Setup complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. If Ollama is not running: ollama serve"
    echo "  2. Run quick test: python3 pipeline.py --samples 5 --no-ollama"
    echo "  3. Run with LLM: python3 pipeline.py --samples 5"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "⚠ Setup incomplete"
    echo "=========================================="
    echo ""
    echo "Please fix the issues above and run setup.sh again"
    echo ""
fi
