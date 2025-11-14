#!/usr/bin/env python3
"""Test what Streamlit sees with combined text"""

import pandas as pd
from enhanced_medical_ner_predictor import EnhancedMedicalNERPredictor

# Create test DataFrame (simulating file upload)
test_data = {
    'Index': [0, 1, 2],
    'Text': [
        'Patient has diabetes and hypertension. No chest pain.',
        'Patient diagnosed with KIF5A mutation and Parkinson\'s disease.',
        'No fever but patient has persistent cough and headache.'
    ]
}
df = pd.DataFrame(test_data)

# Initialize predictor
print("Initializing predictor...")
predictor = EnhancedMedicalNERPredictor(model_name='en_core_web_sm', use_gpu=False)

# Simulate "All rows (combined)" - what Streamlit does
combined_text = "\n\n".join(df['Text'].astype(str).tolist())

print("="*80)
print("COMBINED TEXT:")
print("="*80)
print(combined_text)
print(f"\nLength: {len(combined_text)} chars")
print(f"Rows: {combined_text.count(chr(10) + chr(10)) + 1}")

# Process with predictor (what Streamlit process_text does)
print("\n" + "="*80)
print("PROCESSING WITH PREDICTOR:")
print("="*80)
result = predictor.extract_entities(combined_text)

# Check what's returned (what Streamlit displays)
viz = result.get('Text Visualization', '')
print(f"\nVisualization length: {len(viz)} chars")
print(f"Has entity markers: {'▶[' in viz}")

# Count entities
import re
markers = re.findall(r'▶\[([^\]]+)\]◀', viz)
print(f"Entities highlighted: {len(markers)}")
print(f"Entity list: {markers}")

# Check context entities
confirmed = result.get('confirmed_entities', '')
negated = result.get('negated_entities', '')
family = result.get('family_entities', '')

print(f"\n✅ Confirmed: {confirmed}")
print(f"❌ Negated: {negated}")
print(f"👨‍👩‍👧 Family: {family}")

# Show visualization
print("\n" + "="*80)
print("FULL VISUALIZATION:")
print("="*80)
print(viz)

# Check if this matches what we tested before
print("\n" + "="*80)
print("ANALYSIS:")
print("="*80)

expected_entities = ['diabetes', 'hypertension', 'chest pain', 'pain', 'KIF5A mutation',
                     'parkinson', 'Parkinson', 'fever', 'cough', 'headache']
missing = [e for e in expected_entities if e not in markers]

if missing:
    print(f"❌ Missing entities: {missing}")
else:
    print(f"✅ All expected entities found!")
