#!/usr/bin/env python3
from enhanced_medical_ner_predictor import EnhancedMedicalNERPredictor

predictor = EnhancedMedicalNERPredictor(model_name='en_core_web_sm', use_gpu=False)
text = "Patient has diabetes and hypertension. No chest pain.\n\nPatient diagnosed with KIF5A mutation and Parkinson's disease.\n\nNo fever but patient has persistent cough and headache."
result = predictor.extract_entities(text)

print('Context Classifications:')
print(f'Confirmed: {result.get("confirmed_entities", "")}')
print(f'Negated: {result.get("negated_entities", "")}')
print(f'Historical: {result.get("historical_entities", "")}')
print(f'Uncertain: {result.get("uncertain_entities", "")}')
print(f'Family: {result.get("family_entities", "")}')
