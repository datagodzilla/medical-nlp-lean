#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

# Temporarily patch the function to add debug output
from enhanced_medical_ner_predictor import EnhancedMedicalNERPredictor
import types

def debug_negation_detection(self, text: str, medical_entities: list) -> list:
    """Debug version with print statements"""
    negated_entities = []
    
    if self.negated_rules_enabled and self.negated_patterns:
        keywords = self.negated_patterns
    else:
        keywords = [
            'no', 'not', 'without', 'absent', 'negative', 'denies', 'rules out',
            'free of', 'clear of', 'unremarkable', 'normal', 'within normal limits'
        ]
    
    for entity in medical_entities:
        print(f"\nChecking entity: {entity['text']}")
        start_idx = max(0, entity['start'] - 50)
        end_idx = min(len(text), entity['end'] + 50)
        context = text[start_idx:end_idx].lower()
        
        # Find if 'denies' is in context
        if 'denies' in context:
            entity_text_lower = entity.get('text', '').lower()
            entity_text_pos = context.find(entity_text_lower)
            negation_pos = context.find('denies')
            
            print(f"  Context: '{context}'")
            print(f"  Negation pos: {negation_pos}, Entity pos: {entity_text_pos}")
            
            # Check for 'but' between negation and entity
            if negation_pos >= 0 and entity_text_pos >= 0 and negation_pos < entity_text_pos:
                between_text = context[negation_pos:entity_text_pos].lower()
                print(f"  Between text: '{between_text}'")
                print(f"  Contains ' but '? {' but ' in between_text}")
                
                # Check for confirmation phrases
                for phrase in ['but reports', 'but shows', 'but has']:
                    if phrase in context:
                        phrase_pos = context.find(phrase)
                        print(f"  Found '{phrase}' at pos {phrase_pos}")
                        if phrase_pos >= 0 and entity_text_pos >= 0 and phrase_pos < entity_text_pos:
                            print(f"  -> Scope reversed! Entity should NOT be negated")
    
    return negated_entities

# Initialize predictor
predictor = EnhancedMedicalNERPredictor()

# Patch the method temporarily
predictor._detect_negated_medical_entities = types.MethodType(debug_negation_detection, predictor)

# Test text
text = "Patient denies chest pain but reports dyspnea."

# Extract entities
result = predictor.extract_entities(text)
print(f"\n\nFinal negated entities: {result.get('negated_entities', '')}")
