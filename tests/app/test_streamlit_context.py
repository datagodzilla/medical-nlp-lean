#!/usr/bin/env python3
"""Test if extract_entities returns context sentences."""

import sys
sys.path.insert(0, '/Users/Wolverine/Clindocs_NLP')

from enhanced_medical_ner_predictor import EnhancedMedicalNERPredictor

# Initialize predictor
predictor = EnhancedMedicalNERPredictor()

# Test text
test_text = "Patient denies chest pain or shortness of breath. However, patient reports persistent cough and mild fever for 3 days."

print("Testing extract_entities method (used by Streamlit)...\n")
print(f"Input text:\n{test_text}\n")
print("="*80)

# Process with extract_entities (Streamlit method)
result = predictor.extract_entities(test_text)

print("\n📋 Keys in result:")
for key in sorted(result.keys()):
    print(f"  - {key}")

print("\n📋 Context Sentence Keys:")
context_keys = [k for k in result.keys() if 'context' in k.lower()]
for key in context_keys:
    val = result.get(key, '')
    print(f"\n  {key}:")
    if val and str(val).strip():
        print(f"    {str(val)[:200]}...")
    else:
        print(f"    ❌ EMPTY")

print("\n" + "="*80)
