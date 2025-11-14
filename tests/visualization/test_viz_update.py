#!/usr/bin/env python3
"""
Test script to verify the enhanced Text Visualization with entity contexts
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from enhanced_medical_ner_predictor import EnhancedMedicalNERPredictor


def test_enhanced_visualization():
    """Test the enhanced visualization with different entity contexts"""

    # Initialize predictor
    print("Initializing Enhanced Medical NER Predictor...")
    predictor = EnhancedMedicalNERPredictor()

    # Test cases covering all entity context types
    test_cases = [
        {
            "name": "Confirmed Entity",
            "text": "Patient has diabetes and shows signs of hypertension."
        },
        {
            "name": "Negated Entity",
            "text": "Patient denies chest pain. No history of cardiac disease."
        },
        {
            "name": "Uncertain Entity",
            "text": "Possible pneumonia. May have respiratory infection."
        },
        {
            "name": "Historical Entity",
            "text": "History of breast cancer. Previous diagnosis of hypothyroidism 5 years ago."
        },
        {
            "name": "Family History Entity",
            "text": "Mother has diabetes. Family history of heart disease."
        },
        {
            "name": "Mixed Contexts",
            "text": "Patient has hypertension and diabetes. No cardiac disease. History of asthma. Mother had breast cancer. Possible thyroid disorder."
        },
        {
            "name": "Section Category - Summary",
            "text": "Summary: Patient presents with acute chest pain and shortness of breath."
        },
        {
            "name": "Section Category - Imaging",
            "text": "Imaging Results: CT scan reveals pulmonary embolism in right lung."
        }
    ]

    print("\n" + "=" * 80)
    print("TESTING ENHANCED TEXT VISUALIZATION")
    print("=" * 80)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST CASE {i}: {test_case['name']}")
        print(f"{'=' * 80}")
        print(f"\nInput Text:\n{test_case['text']}")
        print(f"\n{'-' * 80}")

        # Extract entities with context
        result = predictor.extract_entities(test_case['text'])

        print(f"\nEnhanced Text Visualization:")
        print(f"{'-' * 80}")
        print(result.get('Text Visualization', 'No visualization generated'))

        # Show detected entities and their contexts
        print(f"\n{'-' * 80}")
        print("ENTITY CONTEXT SUMMARY:")
        print(f"{'-' * 80}")

        if result.get('confirmed_entities'):
            print(f"✅ CONFIRMED: {result['confirmed_entities']}")
            print(f"   Predictors: {result.get('confirmed_entities_predictors', 'N/A')}")

        if result.get('negated_entities'):
            print(f"❌ NEGATED: {result['negated_entities']}")
            print(f"   Predictors: {result.get('negated_entities_predictors', 'N/A')}")

        if result.get('uncertain_entities'):
            print(f"❓ UNCERTAIN: {result['uncertain_entities']}")
            print(f"   Predictors: {result.get('uncertain_entities_predictors', 'N/A')}")

        if result.get('historical_entities'):
            print(f"📅 HISTORICAL: {result['historical_entities']}")
            print(f"   Predictors: {result.get('historical_entities_predictors', 'N/A')}")

        if result.get('family_entities'):
            print(f"👨‍👩‍👧‍👦 FAMILY: {result['family_entities']}")
            print(f"   Predictors: {result.get('family_entities_predictors', 'N/A')}")

        print(f"\n📋 SECTION: {result.get('section_categories', 'N/A')}")

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    test_enhanced_visualization()
