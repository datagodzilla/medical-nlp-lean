#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Scope Reversal Detection
============================================================

This script tests the enhanced scope reversal system with the updated templates
and validates that the new patterns work correctly with the existing medical NER predictor.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from enhanced_medical_ner_predictor import EnhancedMedicalNERPredictor
import json

class ComprehensiveScopeTest:
    """Test suite for comprehensive scope reversal detection"""

    def __init__(self):
        self.predictor = EnhancedMedicalNERPredictor()
        self.test_cases = self._create_comprehensive_test_cases()

    def _create_comprehensive_test_cases(self):
        """Create comprehensive test cases covering all pattern types"""

        return [
            # CATEGORY 1: Basic Negation → Confirmation
            {
                'text': "Patient denies chest pain but reports dyspnea",
                'expected_negated': ['chest pain'],
                'expected_confirmed': ['dyspnea'],
                'pattern_type': 'basic_negation_to_confirmation',
                'description': 'Basic adversative scope reversal'
            },
            {
                'text': "No fever however shows signs of infection",
                'expected_negated': ['fever'],
                'expected_confirmed': ['infection'],
                'pattern_type': 'basic_negation_to_confirmation',
                'description': 'However-based scope reversal'
            },
            {
                'text': "Denies nausea yet reports vomiting",
                'expected_negated': ['nausea'],
                'expected_confirmed': ['vomiting'],
                'pattern_type': 'basic_negation_to_confirmation',
                'description': 'Yet-based scope reversal'
            },

            # CATEGORY 2: Confirmation → Negation
            {
                'text': "Patient has diabetes but denies complications",
                'expected_negated': ['complications'],
                'expected_confirmed': ['diabetes'],
                'pattern_type': 'confirmation_to_negation',
                'description': 'Confirmation followed by negation'
            },
            {
                'text': "Shows hypertension yet no cardiac symptoms",
                'expected_negated': ['cardiac symptoms'],
                'expected_confirmed': ['hypertension'],
                'pattern_type': 'confirmation_to_negation',
                'description': 'Shows pattern with yet negation'
            },
            {
                'text': "Reports headache but denies visual changes",
                'expected_negated': ['visual changes'],
                'expected_confirmed': ['headache'],
                'pattern_type': 'confirmation_to_negation',
                'description': 'Reports pattern with but negation'
            },

            # CATEGORY 3: Temporal Scope Reversal
            {
                'text': "Denies pain but now reports discomfort",
                'expected_negated': ['pain'],
                'expected_confirmed': ['discomfort'],
                'pattern_type': 'temporal_scope_reversal',
                'description': 'Temporal scope change with "now"'
            },
            {
                'text': "No symptoms however currently shows fatigue",
                'expected_negated': ['symptoms'],
                'expected_confirmed': ['fatigue'],
                'pattern_type': 'temporal_scope_reversal',
                'description': 'Temporal scope with "currently"'
            },

            # CATEGORY 4: Exception Patterns
            {
                'text': "Denies all symptoms except for mild headache",
                'expected_negated': ['symptoms'],
                'expected_confirmed': ['headache'],
                'pattern_type': 'exception_pattern',
                'description': 'Exception pattern with "except for"'
            },
            {
                'text': "No complaints except occasional dizziness",
                'expected_negated': ['complaints'],
                'expected_confirmed': ['dizziness'],
                'pattern_type': 'exception_pattern',
                'description': 'Exception pattern with "except"'
            },

            # CATEGORY 5: Complex Multi-Scope
            {
                'text': "Denies chest pain but reports dyspnea, however no wheezing",
                'expected_negated': ['chest pain', 'wheezing'],
                'expected_confirmed': ['dyspnea'],
                'pattern_type': 'multi_scope',
                'description': 'Multiple scope changes in one sentence'
            },
            {
                'text': "Has diabetes but denies complications, yet shows poor control",
                'expected_negated': ['complications'],
                'expected_confirmed': ['diabetes', 'poor control'],
                'pattern_type': 'multi_scope',
                'description': 'Complex confirmation-negation-confirmation pattern'
            },

            # CATEGORY 6: Extended Conjunction Patterns
            {
                'text': "Patient denies fever but demonstrates signs of inflammation",
                'expected_negated': ['fever'],
                'expected_confirmed': ['inflammation'],
                'pattern_type': 'extended_conjunction',
                'description': 'Demonstrates pattern'
            },
            {
                'text': "No pain but exhibits signs of distress",
                'expected_negated': ['pain'],
                'expected_confirmed': ['distress'],
                'pattern_type': 'extended_conjunction',
                'description': 'Exhibits pattern'
            },
            {
                'text': "Denies shortness of breath but complains of fatigue",
                'expected_negated': ['shortness of breath'],
                'expected_confirmed': ['fatigue'],
                'pattern_type': 'extended_conjunction',
                'description': 'Complains of pattern'
            },

            # CATEGORY 7: Nevertheless/Nonetheless patterns
            {
                'text': "Denies pain nevertheless reports discomfort",
                'expected_negated': ['pain'],
                'expected_confirmed': ['discomfort'],
                'pattern_type': 'nevertheless_pattern',
                'description': 'Nevertheless scope reversal'
            },
            {
                'text': "No fever nonetheless shows infection signs",
                'expected_negated': ['fever'],
                'expected_confirmed': ['infection'],
                'pattern_type': 'nonetheless_pattern',
                'description': 'Nonetheless scope reversal'
            },

            # CATEGORY 8: Edge Cases
            {
                'text': "Patient denies any chest pain",  # No scope reversal
                'expected_negated': ['chest pain'],
                'expected_confirmed': [],
                'pattern_type': 'simple_negation',
                'description': 'Simple negation without scope reversal'
            },
            {
                'text': "Patient reports dyspnea",  # No scope reversal
                'expected_negated': [],
                'expected_confirmed': ['dyspnea'],
                'pattern_type': 'simple_confirmation',
                'description': 'Simple confirmation without scope reversal'
            },

            # CATEGORY 9: Challenging cases
            {
                'text': "Although denies chest pain, reports dyspnea",
                'expected_negated': ['chest pain'],
                'expected_confirmed': ['dyspnea'],
                'pattern_type': 'concessive_pattern',
                'description': 'Concessive conjunction pattern'
            },
            {
                'text': "Despite no fever, shows elevated white count",
                'expected_negated': ['fever'],
                'expected_confirmed': ['elevated white count'],
                'pattern_type': 'despite_pattern',
                'description': 'Despite-based scope reversal'
            }
        ]

    def run_comprehensive_test(self):
        """Run all test cases and generate detailed report"""

        print("="*80)
        print("COMPREHENSIVE SCOPE REVERSAL TEST SUITE")
        print("="*80)
        print(f"Total test cases: {len(self.test_cases)}")
        print()

        results = []
        passed = 0
        failed = 0

        for i, test_case in enumerate(self.test_cases, 1):
            print(f"Test {i:2d}: {test_case['description']}")
            print(f"Text: '{test_case['text']}'")
            print(f"Pattern: {test_case['pattern_type']}")

            # Run prediction
            result = self.predictor.extract_entities(test_case['text'])

            # Extract actual results
            actual_negated = self._extract_entity_names(result.get('negated_entities', ''))
            actual_confirmed = self._extract_entity_names(result.get('confirmed_entities', ''))

            # Check results
            negated_correct = self._check_entities_match(test_case['expected_negated'], actual_negated)
            confirmed_correct = self._check_entities_match(test_case['expected_confirmed'], actual_confirmed)

            test_passed = negated_correct and confirmed_correct

            if test_passed:
                passed += 1
                status = "✅ PASS"
            else:
                failed += 1
                status = "❌ FAIL"

            print(f"Expected Negated: {test_case['expected_negated']}")
            print(f"Actual Negated:   {actual_negated}")
            print(f"Expected Confirmed: {test_case['expected_confirmed']}")
            print(f"Actual Confirmed:   {actual_confirmed}")
            print(f"Result: {status}")

            if not test_passed:
                print("Issues:")
                if not negated_correct:
                    print(f"  - Negated entities mismatch")
                if not confirmed_correct:
                    print(f"  - Confirmed entities mismatch")

            print("-" * 60)

            results.append({
                'test_number': i,
                'description': test_case['description'],
                'pattern_type': test_case['pattern_type'],
                'text': test_case['text'],
                'expected_negated': test_case['expected_negated'],
                'actual_negated': actual_negated,
                'expected_confirmed': test_case['expected_confirmed'],
                'actual_confirmed': actual_confirmed,
                'passed': test_passed,
                'negated_correct': negated_correct,
                'confirmed_correct': confirmed_correct
            })

        # Generate summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {len(self.test_cases)}")
        print(f"Passed: {passed} ({passed/len(self.test_cases)*100:.1f}%)")
        print(f"Failed: {failed} ({failed/len(self.test_cases)*100:.1f}%)")

        # Analyze failures by pattern type
        if failed > 0:
            print(f"\nFAILURE ANALYSIS:")
            failure_by_pattern = {}
            for result in results:
                if not result['passed']:
                    pattern = result['pattern_type']
                    if pattern not in failure_by_pattern:
                        failure_by_pattern[pattern] = 0
                    failure_by_pattern[pattern] += 1

            for pattern, count in failure_by_pattern.items():
                print(f"  - {pattern}: {count} failures")

        # Success analysis
        print(f"\nSUCCESS ANALYSIS:")
        success_by_pattern = {}
        for result in results:
            if result['passed']:
                pattern = result['pattern_type']
                if pattern not in success_by_pattern:
                    success_by_pattern[pattern] = 0
                success_by_pattern[pattern] += 1

        for pattern, count in success_by_pattern.items():
            print(f"  - {pattern}: {count} successes")

        return results

    def _extract_entity_names(self, entity_string):
        """Extract entity names from semicolon-separated string"""
        if not entity_string or entity_string.strip() == '':
            return []

        entities = [e.strip() for e in entity_string.split(';') if e.strip()]
        return entities

    def _check_entities_match(self, expected, actual):
        """Check if expected and actual entity lists match (order-independent)"""
        expected_set = set([e.lower() for e in expected])
        actual_set = set([e.lower() for e in actual])
        return expected_set == actual_set

    def run_detailed_analysis(self, text_samples):
        """Run detailed analysis on specific text samples"""

        print("\n" + "="*80)
        print("DETAILED ANALYSIS")
        print("="*80)

        for i, text in enumerate(text_samples, 1):
            print(f"\nSample {i}: '{text}'")
            print("-" * 40)

            result = self.predictor.extract_entities(text)

            # Parse entities
            all_entities = json.loads(result.get('all_entities_json', '[]'))

            print(f"All entities detected: {[e['text'] for e in all_entities]}")
            print(f"Negated entities: {result.get('negated_entities', '')}")
            print(f"Confirmed entities: {result.get('confirmed_entities', '')}")
            print(f"Uncertain entities: {result.get('uncertain_entities', '')}")

            # Show predictors used
            print(f"Negated predictors: {result.get('negated_entities_predictors', '')}")
            print(f"Confirmed predictors: {result.get('confirmed_entities_predictors', '')}")

if __name__ == "__main__":
    tester = ComprehensiveScopeTest()
    results = tester.run_comprehensive_test()

    # Additional detailed analysis on key patterns
    key_samples = [
        "Patient denies chest pain but reports dyspnea",
        "Has diabetes but denies complications",
        "No fever however shows signs of infection",
        "Denies all symptoms except for mild headache"
    ]

    tester.run_detailed_analysis(key_samples)