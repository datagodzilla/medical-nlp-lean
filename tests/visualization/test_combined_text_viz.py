#!/usr/bin/env python3
"""Test visualization of combined text (with newlines)"""

from enhanced_medical_ner_predictor import EnhancedMedicalNERPredictor

# Initialize predictor
print("Initializing predictor...")
predictor = EnhancedMedicalNERPredictor(model_name='en_core_web_sm', use_gpu=False)

# Test 1: Single text
single_text = "Patient has diabetes and hypertension. No chest pain."
print("\n" + "="*80)
print("TEST 1: Single text")
print("="*80)
print(f"Text: {single_text}")

result1 = predictor.extract_entities(single_text)
viz1 = result1.get('Text Visualization', '')
print(f"\nVisualization length: {len(viz1)}")
print(f"Has <span> tags: {'<span' in viz1}")
print(f"First 200 chars: {viz1[:200]}")

# Test 2: Combined text (like "All rows combined")
combined_text = "Patient has diabetes and hypertension. No chest pain.\n\nPatient diagnosed with KIF5A mutation and Parkinson's disease.\n\nNo fever but patient has persistent cough and headache."
print("\n" + "="*80)
print("TEST 2: Combined text (with \\n\\n separators)")
print("="*80)
print(f"Text: {combined_text[:100]}...")
print(f"Length: {len(combined_text)} chars")
print(f"Number of rows: {combined_text.count(chr(10) + chr(10)) + 1}")

result2 = predictor.extract_entities(combined_text)
viz2 = result2.get('Text Visualization', '')
print(f"\nVisualization length: {len(viz2)}")
print(f"Has <span> tags: {'<span' in viz2}")

# Check entities detected
import json
entities = json.loads(result2.get('all_entities_json', '[]'))
print(f"\nTotal entities detected: {len(entities)}")
for i, e in enumerate(entities[:10], 1):
    print(f"  {i}. {e.get('text')} [{e.get('label')}] at {e.get('start')}-{e.get('end')}")

# Check if visualization highlights are present
confirmed = result2.get('confirmed_entities', '')
negated = result2.get('negated_entities', '')
print(f"\n✅ Confirmed: {confirmed}")
print(f"❌ Negated: {negated}")

# Sample the visualization
print(f"\nVisualization sample (first 500 chars):")
print(viz2[:500])
print("...")
print(f"\nVisualization sample (last 500 chars):")
print(viz2[-500:])
