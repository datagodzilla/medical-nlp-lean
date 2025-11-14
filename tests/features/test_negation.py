#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from enhanced_medical_ner_predictor import EnhancedMedicalNERPredictor

# Initialize predictor
predictor = EnhancedMedicalNERPredictor()

# Test text
text = "Patient denies chest pain but reports dyspnea."

# Extract entities
result = predictor.extract_entities(text)

# Show results
print("Text:", text)
print("\nAll entities detected:")
for entity in result.get('all_entities_json', []):
    import json
    entities = json.loads(result['all_entities_json'])
    for ent in entities:
        print(f"  - {ent['text']} ({ent['label']})")
    break

print("\nNegated entities:", result.get('negated_entities', ''))
print("Negated count:", result.get('negated_entities_count', 0))
print("Negated predictors:", result.get('negated_entities_predictors', ''))

print("\nConfirmed entities:", result.get('confirmed_entities', ''))
print("Confirmed count:", result.get('confirmed_entities_count', 0))
print("Confirmed predictors:", result.get('confirmed_entities_predictors', ''))

# Check context windows
chest_pain_idx = text.find("chest pain")
dyspnea_idx = text.find("dyspnea")

print(f"\nContext around 'chest pain' (50 chars):")
start = max(0, chest_pain_idx - 50)
end = min(len(text), chest_pain_idx + len("chest pain") + 50)
print(f"'{text[start:end]}'")

print(f"\nContext around 'dyspnea' (50 chars):")
start = max(0, dyspnea_idx - 50)
end = min(len(text), dyspnea_idx + len("dyspnea") + 50)
print(f"'{text[start:end]}'")
