#!/usr/bin/env python3
"""
Test script to verify regex patterns are working correctly
"""

import re
import sys
import os

# Add the current directory to the path so we can import from app.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import REGEX_PATTERNS

def test_regex_patterns():
    """Test all regex patterns with sample text"""
    
    test_text = """
    URGENT: Your account has been suspended!
    
    Hello Sarah Johnson,
    
    This is URGENT! Please verify your account immediately at https://suspicious-bank.tk
    Your credit card 1234-5678-9012-3456 will expire soon and needs immediate attention.
    
    Contact us at support@fake-bank.com or call +1-555-123-4567
    Download this important file: account_verification.exe
    
    Bitcoin payment: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
    SSN: 123-45-6789
    
    Meeting with Microsoft on September 20th, 2025 at 3:00 PM in New York.
    Contract value: $50,000
    
    Best regards,
    Mike Anderson
    """
    
    print("Testing regex patterns...")
    print("=" * 50)
    
    found_entities = []
    
    for pattern in REGEX_PATTERNS:
        flags = pattern.get("flags", 0)
        regex = re.compile(pattern["pattern"], flags)
        
        matches = list(regex.finditer(test_text))
        if matches:
            print(f"✓ {pattern['label']}: {len(matches)} matches")
            for match in matches:
                entity = {
                    'text': match.group(),
                    'label': pattern['label'],
                    'start': match.start(),
                    'end': match.end()
                }
                found_entities.append(entity)
                print(f"  - '{entity['text']}' at position {entity['start']}-{entity['end']}")
        else:
            print(f"✗ {pattern['label']}: No matches")
    
    print("\n" + "=" * 50)
    print(f"Total entities found: {len(found_entities)}")
    
    # Group by label
    by_label = {}
    for entity in found_entities:
        label = entity['label']
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(entity['text'])
    
    print("\nEntities by type:")
    for label, texts in by_label.items():
        print(f"  {label}: {texts}")

if __name__ == "__main__":
    test_regex_patterns()
