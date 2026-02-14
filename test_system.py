"""
Quick test script to verify all components are working.

This script tests each module independently before running the full pipeline.
"""

import sys
import os

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from data.loader import IDSDataLoader
        print("✓ Data loader imported")
        
        from models.cnn_model import create_ids_model
        print("✓ CNN model imported")
        
        from models.trainer import IDSModelTrainer
        print("✓ Model trainer imported")
        
        from explainability.shap_explainer import create_shap_explainer
        print("✓ SHAP explainer imported")
        
        from explainability.risk_scorer import create_risk_scorer
        print("✓ Risk scorer imported")
        
        from llm.ollama_client import create_ollama_explainer
        print("✓ LLM client imported")
        
        from agent.decision_agent import create_decision_agent
        print("✓ Decision agent imported")
        
        print("\n✓ All imports successful!\n")
        return True
    
    except Exception as e:
        print(f"\n✗ Import failed: {e}\n")
        return False


def test_dependencies():
    """Test that all dependencies are installed."""
    print("Testing dependencies...")
    
    dependencies = [
        'tensorflow',
        'numpy',
        'pandas',
        'sklearn',
        'shap',
        'langchain',
        'langchain_community',
        'kagglehub',
        'matplotlib',
        'seaborn'
    ]
    
    missing = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep} - MISSING")
            missing.append(dep)
    
    if missing:
        print(f"\n✗ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt\n")
        return False
    else:
        print("\n✓ All dependencies installed!\n")
        return True


def test_ollama():
    """Test Ollama connection."""
    print("Testing Ollama connection...")
    
    try:
        from langchain_community.llms import Ollama
        
        llm = Ollama(model="llama3.2", temperature=0.3)
        response = llm.invoke("Say 'OK' if you can read this.")
        
        print(f"✓ Ollama connected")
        print(f"  Response: {response[:50]}...")
        print()
        return True
    
    except Exception as e:
        print(f"✗ Ollama connection failed: {e}")
        print("  Make sure Ollama is running: ollama serve")
        print("  And model is pulled: ollama pull llama3.2\n")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("IDS SYSTEM COMPONENT TEST")
    print("="*60)
    print()
    
    results = []
    
    # Test dependencies
    results.append(("Dependencies", test_dependencies()))
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test Ollama (optional)
    results.append(("Ollama (optional)", test_ollama()))
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:.<40} {status}")
    
    print()
    
    # Overall result
    critical_tests = results[:2]  # Dependencies and Imports are critical
    if all(passed for _, passed in critical_tests):
        print("✓ System ready to run!")
        print("\nNext steps:")
        print("  1. Run pipeline: python pipeline.py --samples 5 --no-ollama")
        print("  2. With Ollama: python pipeline.py --samples 5")
    else:
        print("✗ System not ready. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
