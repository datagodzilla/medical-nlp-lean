#!/usr/bin/env python3
"""Test Streamlit display logic for context sentences."""

import sys
sys.path.insert(0, '/Users/Wolverine/Clindocs_NLP')

from enhanced_medical_ner_predictor import EnhancedMedicalNERPredictor

# Initialize predictor
predictor = EnhancedMedicalNERPredictor()

# Test text
test_text = "Patient denies chest pain or shortness of breath. However, patient reports persistent cough and mild fever for 3 days."

print("Testing Streamlit display logic...\n")

# Process with extract_entities (Streamlit method)
result = predictor.extract_entities(test_text)

# Simulate what Streamlit does
result_dict = result  # In Streamlit: results.get('result', {})

print("Checking display conditions:\n")
print("="*80)

# Confirmed
if result_dict.get('confirmed_entities_count', 0) > 0:
    print(f"\n✅ Confirmed Entities ({result_dict.get('confirmed_entities_count', 0)})")
    print(f"   Entities: {result_dict.get('confirmed_entities', '')}")

    # Check the conditional
    if result_dict.get('confirmed_entities_predictors'):
        print(f"   ✓ Has predictors: {result_dict.get('confirmed_entities_predictors', '')}")
    else:
        print(f"   ✗ No predictors")

    if result_dict.get('confirmed_context_sentences'):
        print(f"   ✓ Has context: {result_dict.get('confirmed_context_sentences', '')[:100]}...")
    else:
        print(f"   ✗ No context sentences - WOULD NOT DISPLAY")

# Negated
if result_dict.get('negated_entities_count', 0) > 0:
    print(f"\n❌ Negated Entities ({result_dict.get('negated_entities_count', 0)})")
    print(f"   Entities: {result_dict.get('negated_entities', '')}")

    # Check the conditional
    if result_dict.get('negated_entities_predictors'):
        print(f"   ✓ Has predictors: {result_dict.get('negated_entities_predictors', '')}")
    else:
        print(f"   ✗ No predictors")

    if result_dict.get('negated_context_sentences'):
        print(f"   ✓ Has context: {result_dict.get('negated_context_sentences', '')[:100]}...")
    else:
        print(f"   ✗ No context sentences - WOULD NOT DISPLAY")

print("\n" + "="*80)
print("\nTesting boolean evaluation of context sentences:")
print(f"confirmed_context_sentences: {bool(result_dict.get('confirmed_context_sentences'))}")
print(f"negated_context_sentences: {bool(result_dict.get('negated_context_sentences'))}")
