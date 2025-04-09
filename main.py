#!/usr/bin/env python3
"""
Medical Assistant AI - Main Entry Point

This script serves as the main entry point for the Medical Assistant AI application.
It simply launches the Streamlit interface defined in src/app.py
"""

import os
import sys

# Make sure the src directory is in the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    # Check for API key
    if not os.environ.get('GOOGLE_API_KEY'):
        print("Warning: GOOGLE_API_KEY environment variable is not set.")
        print("Please set it before running the application:")
        print("  export GOOGLE_API_KEY='your-api-key-here'")
    
    # Run the Streamlit app using subprocess instead of direct import
    # This avoids potential API changes in streamlit.web.cli
    import subprocess
    app_path = os.path.join(os.path.dirname(__file__), 'src/app.py')
    subprocess.run(["streamlit", "run", app_path], check=True)