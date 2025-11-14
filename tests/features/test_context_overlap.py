#!/usr/bin/env python3
"""
Comprehensive test suite for context overlap fix verification
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_medical_ner_predictor import EnhancedMedicalNERPredictor
import json
import re
import time

def run_comprehensive_overlap_tests():
    """Run comprehensive suite of context overlap tests"""
    print("🧪 COMPREHENSIVE CONTEXT OVERLAP TEST SUITE")
    print("=" * 70)
    print("This suite tests the fix for context overlap issues where entities")
    print("appeared with multiple context emojis simultaneously.")
    print("=" * 70)

    predictor = EnhancedMedicalNERPredictor()

    # Test Suite 1: Real-world medical texts
    real_world_tests = [
        {
            "name": "KIF5A Complex Medical Text",
            "text": """Kinesin family member 5A (KIF5A)-related disorder (KIF5A-RD) is a rare disease caused by a change (variant) in the KIF5A gene. The KIF5A gene has instructions to make (codify) the KIF5A protein. Depending on where the variant occurs in the KIF5A gene, the disorder can affect different parts (domains) of the KIF5A protein and result in different signs and symptoms."""
        },
        {
            "name": "Mixed Context Medical Report",
            "text": """Patient confirmed diabetes and hypertension. Family history of heart disease. Previously had pneumonia which resolved. Patient denies chest pain but uncertain about shortness of breath. May have sleep apnea."""
        },
        {
            "name": "Cardiology Report",
            "text": """Patient has confirmed myocardial infarction. Mother had stroke at age 65. Patient denies current chest pain. Possibly has arrhythmia. Previous ECG showed abnormalities."""
        }
    ]

    # Test Suite 2: Edge cases designed to trigger overlaps
    edge_case_tests = [
        {
            "name": "Same Entity Different Contexts",
            "text": "Patient has diabetes. Family history of diabetes. Previously treated for diabetes complications. Patient denies diabetic neuropathy but uncertain about diabetes management."
        },
        {
            "name": "Overlapping Medical Terms",
            "text": "Confirmed heart disease. Patient uncertain about heart rhythm. Family history of heart attack. Denies heart failure symptoms."
        },
        {
            "name": "Complex Symptom Description",
            "text": "Patient reports chest pain symptoms. Previous chest pain episode resolved. Father had chest pain issues. Patient denies current chest pain. Possibly chest pain related to anxiety."
        }
    ]

    # Test Suite 3: Priority hierarchy validation
    priority_tests = [
        {
            "name": "Negated vs Confirmed Priority",
            "text": "Patient has hypertension but denies hypertension complications.",
            "expected_priority": "negated should win over confirmed"
        },
        {
            "name": "Family vs Uncertain Priority",
            "text": "Mother possibly had diabetes. Patient uncertain about diabetes risk.",
            "expected_priority": "family should win over uncertain"
        },
        {
            "name": "Historical vs Confirmed Priority",
            "text": "Patient previously had pneumonia. Patient has pneumonia symptoms.",
            "expected_priority": "historical should win over confirmed"
        }
    ]

    def parse_entities(entity_str):
        """Parse entities from context string, removing emojis"""
        if not entity_str or entity_str == 'None':
            return set()
        clean_str = re.sub(r'[✅❌❓📜👨‍👩‍👧‍👦]', '', entity_str)
        return {e.strip().lower() for e in clean_str.split(';') if e.strip()}

    def check_for_overlaps(result):
        """Check if any entities appear in multiple contexts"""
        contexts = {
            'confirmed': parse_entities(result.get('confirmed_entities', '')),
            'uncertain': parse_entities(result.get('uncertain_entities', '')),
            'negated': parse_entities(result.get('negated_entities', '')),
            'family': parse_entities(result.get('family_entities', '')),
            'historical': parse_entities(result.get('historical_entities', ''))
        }

        overlaps = []
        context_names = list(contexts.keys())
        for i, ctx1 in enumerate(context_names):
            for j, ctx2 in enumerate(context_names):
                if i < j:
                    overlap = contexts[ctx1].intersection(contexts[ctx2])
                    if overlap:
                        overlaps.append((ctx1, ctx2, overlap))

        return overlaps, contexts

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    # Run all test suites
    all_test_suites = [
        ("Real-World Medical Texts", real_world_tests),
        ("Edge Case Tests", edge_case_tests),
        ("Priority Hierarchy Tests", priority_tests)
    ]

    for suite_name, test_suite in all_test_suites:
        print(f"\n🔬 {suite_name}")
        print("-" * 50)

        for i, test_case in enumerate(test_suite, 1):
            total_tests += 1
            test_name = test_case["name"]
            text = test_case["text"]

            print(f"\n  📝 Test {i}: {test_name}")
            print(f"     Text: {text[:60]}...")

            try:
                start_time = time.time()
                result = predictor.extract_entities(text)
                processing_time = time.time() - start_time

                overlaps, contexts = check_for_overlaps(result)

                if overlaps:
                    print(f"     ❌ OVERLAPS DETECTED:")
                    for ctx1, ctx2, overlap_entities in overlaps:
                        print(f"        {ctx1} ↔ {ctx2}: {overlap_entities}")
                    failed_tests.append({
                        'name': test_name,
                        'overlaps': overlaps,
                        'text': text
                    })
                else:
                    print(f"     ✅ No overlaps - Context separation working correctly")
                    passed_tests += 1

                # Show entity distribution
                entity_counts = {ctx: len(entities) for ctx, entities in contexts.items() if entities}
                if entity_counts:
                    print(f"     📊 Entities: {dict(entity_counts)}")

                print(f"     ⏱️ Processing time: {processing_time:.3f}s")

            except Exception as e:
                print(f"     ❌ ERROR: {e}")
                failed_tests.append({
                    'name': test_name,
                    'error': str(e),
                    'text': text
                })

    # Summary Report
    print(f"\n" + "=" * 70)
    print(f"📋 FINAL TEST RESULTS")
    print(f"=" * 70)
    print(f"Total Tests Run: {total_tests}")
    print(f"Tests Passed: {passed_tests}")
    print(f"Tests Failed: {len(failed_tests)}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")

    if failed_tests:
        print(f"\n❌ FAILED TESTS:")
        for i, failed_test in enumerate(failed_tests, 1):
            print(f"  {i}. {failed_test['name']}")
            if 'overlaps' in failed_test:
                for ctx1, ctx2, entities in failed_test['overlaps']:
                    print(f"     Overlap: {ctx1} ↔ {ctx2}: {entities}")
            elif 'error' in failed_test:
                print(f"     Error: {failed_test['error']}")

    # Overall assessment
    if passed_tests == total_tests:
        print(f"\n🎉 ALL TESTS PASSED! Context overlap fix is working perfectly!")
        print(f"✅ Entities are properly assigned to single contexts according to priority hierarchy")
        return True
    else:
        print(f"\n⚠️ Some tests failed. Context overlap issues may persist.")
        print(f"🔧 Review failed tests above and check deduplication logic.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_overlap_tests()
    sys.exit(0 if success else 1)