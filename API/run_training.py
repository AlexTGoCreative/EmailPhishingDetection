#!/usr/bin/env python3
"""
Simple script to run the URGENCY model training
"""

import subprocess
import sys
import os

def main():
    print("Starting URGENCY Entity Model Training")
    print("=" * 50)
    
    # Check if required packages are installed
    try:
        import spacy
        print("spaCy is installed")
    except ImportError:
        print("spaCy not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
    
    # Check if base model exists
    try:
        nlp = spacy.load("en_core_web_trf")
        print("en_core_web_trf model found")
    except OSError:
        print("en_core_web_trf not found. Downloading...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_trf"])
    
    # Run the training
    print("\nStarting training process...")
    result = subprocess.run([sys.executable, "train_urgency_model.py"], 
                          capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    if result.returncode == 0:
        print("\nTraining completed successfully!")
        print("\nYou can now restart your Flask app to use the new model.")
        print("   The app will automatically load ./custom_urgency_model")
    else:
        print(f"\nTraining failed with return code {result.returncode}")

if __name__ == "__main__":
    main()
