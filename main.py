#!/usr/bin/env python3
"""
Medical Assistant AI - Main Entry Point

This script serves as the main entry point for the Medical Assistant AI application.
It launches the Streamlit interface defined in src/app.py which uses Ollama Llama 3.2
for medical assistance and patient data analysis.
"""

import os
import sys

# Make sure the src directory is in the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    # Check for Ollama
    print("Medical Assistant AI - Powered by Ollama Llama 3.2")
    print("=" * 50)
    print("Before running, please ensure:")
    print("1. Ollama is installed and running")
    print("2. Llama 3.2 model is installed: ollama pull llama3.2")
    print("3. Ollama server is accessible at http://localhost:11434")
    print()
    
    # Run the Streamlit app using subprocess instead of direct import
    # This avoids potential API changes in streamlit.web.cli
    import subprocess
    app_path = os.path.join(os.path.dirname(__file__), 'src/app.py')
    subprocess.run(["streamlit", "run", app_path], check=True)