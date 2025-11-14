#!/usr/bin/env python3
from enhanced_medical_ner_predictor import EnhancedMedicalNERPredictor

predictor = EnhancedMedicalNERPredictor(model_name='en_core_web_sm', use_gpu=False)
combined_text = "Patient has diabetes and hypertension. No chest pain.\n\nPatient diagnosed with KIF5A mutation and Parkinson's disease.\n\nNo fever but patient has persistent cough and headache."
result = predictor.extract_entities(combined_text)
viz = result.get('Text Visualization', '')
print(viz)
