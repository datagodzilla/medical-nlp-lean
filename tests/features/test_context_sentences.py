#!/usr/bin/env python3
"""Test context sentences after variable shadowing fix."""

import sys
import pandas as pd
sys.path.insert(0, '/Users/Wolverine/Clindocs_NLP')

from enhanced_medical_ner_predictor import EnhancedMedicalNERPredictor

# Initialize predictor
predictor = EnhancedMedicalNERPredictor()

# Test text with clear negation and confirmation
test_data = pd.DataFrame({
    'Index': [1],
    'Text': [
        "Patient denies chest pain or shortness of breath. However, patient reports persistent cough and mild fever for 3 days."
    ]
})

print("Testing context sentences after variable shadowing fix...\n")
print(f"Input text:\n{test_data['Text'].iloc[0]}\n")
print("="*80)

# Process text
result_df = predictor.predict_dataframe(test_data, text_column='Text')

# Check context sentences
print("\n📋 Context Classification Details:")
print("-" * 80)

# Get the first row results
row = result_df.iloc[0]

# Negated
negated_entities = row.get('negated_entities', '')
negated_context = row.get('negated_context_sentences', '')
if negated_entities:
    print(f"\n❌ Negated entities: {negated_entities}")
    if negated_context:
        print(f"   Context: {negated_context}")
    else:
        print("   ⚠️  WARNING: Context sentences are EMPTY!")

# Confirmed
confirmed_entities = row.get('confirmed_entities', '')
confirmed_context = row.get('confirmed_context_sentences', '')
if confirmed_entities:
    print(f"\n✓ Confirmed entities: {confirmed_entities}")
    if confirmed_context:
        print(f"   Context: {confirmed_context}")
    else:
        print("   ⚠️  WARNING: Context sentences are EMPTY!")

# Historical
historical_entities = row.get('historical_entities', '')
historical_context = row.get('historical_context_sentences', '')
if historical_entities:
    print(f"\n🕐 Historical entities: {historical_entities}")
    if historical_context:
        print(f"   Context: {historical_context}")
    else:
        print("   ⚠️  WARNING: Context sentences are EMPTY!")

# Uncertain
uncertain_entities = row.get('uncertain_entities', '')
uncertain_context = row.get('uncertain_context_sentences', '')
if uncertain_entities:
    print(f"\n❓ Uncertain entities: {uncertain_entities}")
    if uncertain_context:
        print(f"   Context: {uncertain_context}")
    else:
        print("   ⚠️  WARNING: Context sentences are EMPTY!")

# Family
family_entities = row.get('family_entities', '')
family_context = row.get('family_context_sentences', '')
if family_entities:
    print(f"\n👨‍👩‍👧‍👦 Family entities: {family_entities}")
    if family_context:
        print(f"   Context: {family_context}")
    else:
        print("   ⚠️  WARNING: Context sentences are EMPTY!")

print("\n" + "="*80)
if negated_context or confirmed_context or historical_context or uncertain_context or family_context:
    print("\n✅ FIX SUCCESSFUL - Context sentences are populated!")
else:
    print("\n❌ FIX FAILED - Context sentences are still empty!")
