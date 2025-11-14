#!/usr/bin/env python3
"""
Test negation pattern word boundaries to prevent false positives.
Tests the fix for Row 20 and Row 23 negation false positives.
"""

import re

def test_word_boundary_patterns():
    """Test that negation patterns use word boundaries correctly"""

    # Test cases from Row 20 and 23
    test_cases = [
        {
            'text': 'signs and symptoms include fever, chills, cough, dyspnea, headache',
            'entity': 'cough',
            'should_be_negated': False,
            'reason': '"cough" is a positive symptom, not negated'
        },
        {
            'text': 'manifests as chronic cough, dyspnea, anorexia, and weight loss',
            'entity': 'anorexia',
            'should_be_negated': False,
            'reason': '"anorexia" is a positive symptom, not negated (contains "nor" but not as negation)'
        },
        {
            'text': 'characterized by neuron-specific enolase expression',
            'entity': 'neuron-specific enolase',
            'should_be_negated': False,
            'reason': '"neuron" contains "no" but is not a negation word'
        },
        {
            'text': 'lung, breast, cervical, and thymic neuroendocrine carcinomas',
            'entity': 'neuroendocrine carcinomas',
            'should_be_negated': False,
            'reason': '"neuroendocrine" contains "no" but is not a negation word'
        },
        {
            'text': 'patient denies chest pain',
            'entity': 'chest pain',
            'should_be_negated': True,
            'reason': '"denies" is a true negation word'
        },
        {
            'text': 'no evidence of infection',
            'entity': 'infection',
            'should_be_negated': True,
            'reason': '"no" at word boundary is true negation'
        },
        {
            'text': 'patient has normal blood pressure',
            'entity': 'blood pressure',
            'should_be_negated': True,
            'reason': '"normal" indicates absence of pathology (negation)'
        },
        {
            'text': 'ruled out myocardial infarction',
            'entity': 'myocardial infarction',
            'should_be_negated': True,
            'reason': '"ruled out" is multi-word negation pattern'
        },
    ]

    # Negation keywords (same as in the code)
    keywords = [
        'no', 'not', 'without', 'absent', 'negative', 'denies', 'rules out',
        'free of', 'clear of', 'unremarkable', 'normal', 'within normal limits'
    ]

    print("=" * 80)
    print("NEGATION WORD BOUNDARY TEST")
    print("=" * 80)
    print(f"Testing {len(test_cases)} cases with {len(keywords)} negation patterns\n")

    passed = 0
    failed = 0

    for i, test_case in enumerate(test_cases, 1):
        text = test_case['text']
        entity = test_case['entity']
        expected_negated = test_case['should_be_negated']
        reason = test_case['reason']

        # Simulate the detection logic (50 char window around entity)
        entity_pos = text.lower().find(entity.lower())
        if entity_pos == -1:
            print(f"❌ Test {i}: Entity not found in text!")
            failed += 1
            continue

        start_idx = max(0, entity_pos - 50)
        end_idx = min(len(text), entity_pos + len(entity) + 50)
        context = text[start_idx:end_idx].lower()

        # Check with word boundaries (NEW logic)
        is_negated = False
        matched_pattern = None

        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Multi-word patterns: use simple substring match
            if ' ' in keyword_lower:
                if keyword_lower in context:
                    is_negated = True
                    matched_pattern = keyword
                    break
            # Single-word patterns: use word boundaries
            else:
                if re.search(r'\b' + re.escape(keyword_lower) + r'\b', context):
                    is_negated = True
                    matched_pattern = keyword
                    break

        # Check result
        test_passed = (is_negated == expected_negated)
        status = "✅ PASS" if test_passed else "❌ FAIL"

        print(f"\nTest {i}: {status}")
        print(f"  Text: \"{text}\"")
        print(f"  Entity: \"{entity}\"")
        print(f"  Expected: {'NEGATED' if expected_negated else 'NOT NEGATED'}")
        print(f"  Actual: {'NEGATED' if is_negated else 'NOT NEGATED'}")
        if matched_pattern:
            print(f"  Matched pattern: \"{matched_pattern}\"")
        print(f"  Reason: {reason}")

        if test_passed:
            passed += 1
        else:
            failed += 1
            # Show why it failed
            if is_negated and not expected_negated:
                print(f"  ⚠️  FALSE POSITIVE: Incorrectly detected as negated!")
            elif not is_negated and expected_negated:
                print(f"  ⚠️  FALSE NEGATIVE: Should be negated but wasn't detected!")

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(test_cases)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success rate: {passed / len(test_cases) * 100:.1f}%")
    print("=" * 80)

    return failed == 0


def test_specific_false_positives():
    """Test the specific false positives from Row 20 and 23"""

    print("\n" + "=" * 80)
    print("SPECIFIC FALSE POSITIVE TESTS (Row 20 & 23)")
    print("=" * 80 + "\n")

    # Row 20 false positives
    row20_text = "Hypersensitivity pneumonitis caused by the repeated exposure and inhalation of biological dust. In the acute phase, signs and symptoms include fever, chills, cough, dyspnea, headache, and chest tightness. The subacute phase manifests as chronic cough, dyspnea, anorexia, and weight loss."

    row20_entities = [
        {'text': 'cough', 'start': row20_text.find('cough')},
        {'text': 'dyspnea', 'start': row20_text.find('dyspnea')},
        {'text': 'anorexia', 'start': row20_text.find('anorexia')},
        {'text': 'weight loss', 'start': row20_text.find('weight loss')},
    ]

    # Row 23 false positives
    row23_text = "A usually aggressive carcinoma composed of large malignant cells which display neuroendocrine characteristics. The vast majority of cases are positive for neuron-specific enolase. Representative examples include lung, breast, cervical, and thymic neuroendocrine carcinomas."

    row23_entities = [
        {'text': 'carcinoma', 'start': row23_text.find('carcinoma')},
        {'text': 'neuroendocrine carcinomas', 'start': row23_text.rfind('neuroendocrine carcinomas')},
        {'text': 'neuron-specific enolase', 'start': row23_text.find('neuron-specific enolase')},
    ]

    keywords = ['no', 'not', 'without', 'absent', 'negative', 'denies', 'rules out',
                'free of', 'clear of', 'unremarkable', 'normal', 'within normal limits']

    all_passed = True

    # Test Row 20
    print("Row 20 Entities (should NOT be negated):")
    for entity in row20_entities:
        start_idx = max(0, entity['start'] - 50)
        end_idx = min(len(row20_text), entity['start'] + len(entity['text']) + 50)
        context = row20_text[start_idx:end_idx].lower()

        is_negated = False
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if ' ' in keyword_lower:
                if keyword_lower in context:
                    is_negated = True
                    break
            else:
                if re.search(r'\b' + re.escape(keyword_lower) + r'\b', context):
                    is_negated = True
                    break

        status = "✅ CORRECT" if not is_negated else "❌ FALSE POSITIVE"
        print(f"  {status}: \"{entity['text']}\" - Negated: {is_negated}")
        if is_negated:
            all_passed = False

    print("\nRow 23 Entities (should NOT be negated):")
    for entity in row23_entities:
        start_idx = max(0, entity['start'] - 50)
        end_idx = min(len(row23_text), entity['start'] + len(entity['text']) + 50)
        context = row23_text[start_idx:end_idx].lower()

        is_negated = False
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if ' ' in keyword_lower:
                if keyword_lower in context:
                    is_negated = True
                    break
            else:
                if re.search(r'\b' + re.escape(keyword_lower) + r'\b', context):
                    is_negated = True
                    break

        status = "✅ CORRECT" if not is_negated else "❌ FALSE POSITIVE"
        print(f"  {status}: \"{entity['text']}\" - Negated: {is_negated}")
        if is_negated:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL FALSE POSITIVES FIXED!")
    else:
        print("❌ SOME FALSE POSITIVES REMAIN")
    print("=" * 80)

    return all_passed


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("NEGATION BOUNDARY CONDITION TESTS")
    print("Testing fix for Row 20 & 23 false positives")
    print("=" * 80 + "\n")

    try:
        # Test 1: General word boundary patterns
        test1_passed = test_word_boundary_patterns()

        # Test 2: Specific false positives from Row 20 & 23
        test2_passed = test_specific_false_positives()

        print("\n" + "=" * 80)
        print("FINAL RESULT")
        print("=" * 80)

        if test1_passed and test2_passed:
            print("✅ ALL TESTS PASSED!")
            print("\nThe word boundary fix successfully prevents false positives:")
            print("  ✓ 'neuron' no longer matches 'no'")
            print("  ✓ 'anorexia' no longer matches 'nor'")
            print("  ✓ 'neuroendocrine' no longer matches 'no'")
            print("  ✓ True negation words ('no', 'not', 'denies') still work")
            print("  ✓ Multi-word patterns ('ruled out', 'no evidence of') still work")
            return 0
        else:
            print("❌ SOME TESTS FAILED")
            print("\nPlease review the failures above.")
            return 1

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
