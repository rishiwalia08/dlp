#!/bin/bash

# Setup Python 3.11 virtual environment for IDS project
# This resolves TensorFlow compatibility issues with Python 3.14

echo "=========================================="
echo "Setting up Python 3.11 Virtual Environment"
echo "=========================================="
echo ""

# Navigate to project directory
cd /Users/rishiwalia/Documents/rishi/project/ids-explainable-agent

# Create virtual environment with Python 3.11
echo "Creating virtual environment with Python 3.11..."
python3.11 -m venv venv

if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment"
    echo "Make sure Python 3.11 is installed: brew install python@3.11"
    exit 1
fi

echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

echo ""

# Install dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Setup Complete!"
    echo "=========================================="
    echo ""
    echo "Virtual environment is ready at: venv/"
    echo ""
    echo "To activate in VS Code terminal:"
    echo "  source venv/bin/activate"
    echo ""
    echo "To run the project:"
    echo "  python pipeline.py --samples 5"
    echo ""
else
    echo ""
    echo "⚠ Some dependencies failed to install"
    echo "Check the errors above"
fi
