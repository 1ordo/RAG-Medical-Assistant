#!/usr/bin/env python3
"""
Medical Assistant AI - Main Entry Point

This script serves as the main entry point for the Medical Assistant AI application.
It launches the Streamlit interface defined in src/app.py which supports both
Ollama Llama 3.2 and Google Gemini for medical assistance and patient data analysis.
"""

import os
import sys

# Make sure the src directory is in the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    print("Medical Assistant AI - Powered by Ollama & Google Gemini")
    print("=" * 55)
    print("🎛️  Interactive Model Switching Available!")
    print("AI Provider Options:")
    print("1. Ollama (Local) - Ensure Ollama is running with llama3.2 model")
    print("2. Google Gemini (Cloud) - Set GEMINI_API_KEY environment variable")
    print("3. Auto-detect - Automatically choose available provider")
    print()
    print("🔄 Switch between providers anytime using the sidebar!")
    print()
    print("Setup Instructions:")
    print("📋 For Ollama:")
    print("   - Install Ollama and run: ollama pull llama3.2")
    print("   - Start Ollama server (default: http://localhost:11434)")
    print()
    print("🔑 For Google Gemini:")
    print("   - Get API key from Google AI Studio (https://ai.google.dev/)")
    print("   - Set environment variable: GEMINI_API_KEY=your_api_key")
    print()
    print("💡 Use the new configure.py script for easy setup:")
    print("   python configure.py")
    print()
    print("🚀 Starting the application...")
    print()
    
    # Run the Streamlit app using subprocess instead of direct import
    # This avoids potential API changes in streamlit.web.cli
    import subprocess
    app_path = os.path.join(os.path.dirname(__file__), 'src/app.py')
    subprocess.run(["streamlit", "run", app_path], check=True)