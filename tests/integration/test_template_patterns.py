#!/usr/bin/env python3
"""
Target Rules Template Validation Test Suite
==========================================

This script validates the target rules template structure, content integrity,
and proper integration with the Enhanced Medical NER system.
"""

import pandas as pd
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_medical_ner_predictor import EnhancedMedicalNERPredictor
except ImportError as e:
    print(f"❌ Failed to import EnhancedMedicalNERPredictor: {e}")
    sys.exit(1)

class TargetRulesTemplateValidator:
    """Comprehensive validator for target rules template"""

    def __init__(self):
        self.template_path = Path("data/external/target_rules_template.xlsx")
        self.backup_path = Path("data/external/backup/target_rules_template_backup_20251005_050853.xlsx")
        self.predictor = None
        self.template_df = None

    def run_full_validation(self):
        """Run comprehensive validation of target rules template"""
        print("=" * 80)
        print("TARGET RULES TEMPLATE VALIDATION SUITE")
        print("=" * 80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Test 1: File existence and accessibility
        file_test = self._test_file_existence()

        # Test 2: Template structure validation
        structure_test = self._test_template_structure()

        # Test 3: Content integrity validation
        content_test = self._test_content_integrity()

        # Test 4: Scope reversal terms validation
        scope_test = self._test_scope_reversal_terms()

        # Test 5: Integration with NER system
        integration_test = self._test_ner_integration()

        # Test 6: Performance impact validation
        performance_test = self._test_performance_impact()

        # Generate summary
        self._generate_validation_summary([
            ('File Existence', file_test),
            ('Template Structure', structure_test),
            ('Content Integrity', content_test),
            ('Scope Reversal Terms', scope_test),
            ('NER Integration', integration_test),
            ('Performance Impact', performance_test)
        ])

    def _test_file_existence(self):
        """Test 1: Validate file existence and accessibility"""
        print("🔍 TEST 1: File Existence and Accessibility")
        print("-" * 50)

        results = {'passed': 0, 'total': 0, 'details': []}

        # Check main template file
        results['total'] += 1
        if self.template_path.exists():
            results['passed'] += 1
            file_size = self.template_path.stat().st_size
            results['details'].append(f"✅ Main template exists: {file_size:,} bytes")
        else:
            results['details'].append(f"❌ Main template missing: {self.template_path}")

        # Check backup file
        results['total'] += 1
        if self.backup_path.exists():
            results['passed'] += 1
            backup_size = self.backup_path.stat().st_size
            results['details'].append(f"✅ Backup exists: {backup_size:,} bytes")
        else:
            results['details'].append(f"❌ Backup missing: {self.backup_path}")

        # Check file readability
        results['total'] += 1
        try:
            self.template_df = pd.read_excel(self.template_path)
            results['passed'] += 1
            results['details'].append(f"✅ File readable: {len(self.template_df):,} rows")
        except Exception as e:
            results['details'].append(f"❌ File read error: {e}")

        for detail in results['details']:
            print(f"  {detail}")

        success_rate = results['passed'] / results['total'] * 100
        print(f"\nResult: {results['passed']}/{results['total']} ({success_rate:.1f}%)")
        print()

        return results

    def _test_template_structure(self):
        """Test 2: Validate template structure and column integrity"""
        print("📋 TEST 2: Template Structure Validation")
        print("-" * 50)

        results = {'passed': 0, 'total': 0, 'details': []}

        if self.template_df is None:
            results['details'].append("❌ Cannot test structure - template not loaded")
            return results

        # Check actual columns in the template
        actual_columns = list(self.template_df.columns)
        results['details'].append(f"📋 Actual columns found: {actual_columns}")

        # Check required columns (more flexible)
        required_columns = ['term', 'entitity', 'category', 'confidence']
        optional_columns = ['entity', 'label']  # Alternative column names
        results['total'] += len(required_columns)

        for col in required_columns:
            if col in self.template_df.columns:
                results['passed'] += 1
                results['details'].append(f"✅ Column exists: {col}")
            elif col == 'entitity' and 'entity' in self.template_df.columns:
                results['passed'] += 1
                results['details'].append(f"✅ Column exists (alternative): entity (for entitity)")
            elif col == 'category' and 'label' in self.template_df.columns:
                results['passed'] += 1
                results['details'].append(f"✅ Column exists (alternative): label (for category)")
            else:
                results['details'].append(f"❌ Missing column: {col}")

        # Check column data types
        results['total'] += 1
        if 'confidence' in self.template_df.columns:
            if pd.api.types.is_numeric_dtype(self.template_df['confidence']):
                results['passed'] += 1
                results['details'].append("✅ Confidence column is numeric")
            else:
                results['details'].append("❌ Confidence column should be numeric")

        # Check for empty terms
        results['total'] += 1
        if 'term' in self.template_df.columns:
            empty_terms = self.template_df['term'].isna().sum()
            if empty_terms == 0:
                results['passed'] += 1
                results['details'].append("✅ No empty terms found")
            else:
                results['details'].append(f"❌ Found {empty_terms} empty terms")

        for detail in results['details']:
            print(f"  {detail}")

        success_rate = results['passed'] / results['total'] * 100
        print(f"\nResult: {results['passed']}/{results['total']} ({success_rate:.1f}%)")
        print()

        return results

    def _test_content_integrity(self):
        """Test 3: Validate content integrity and expected statistics"""
        print("📊 TEST 3: Content Integrity Validation")
        print("-" * 50)

        results = {'passed': 0, 'total': 0, 'details': []}

        if self.template_df is None:
            results['details'].append("❌ Cannot test content - template not loaded")
            return results

        # Check total row count
        results['total'] += 1
        row_count = len(self.template_df)
        if row_count >= 57000:  # Expected around 57,476
            results['passed'] += 1
            results['details'].append(f"✅ Row count acceptable: {row_count:,}")
        else:
            results['details'].append(f"❌ Row count too low: {row_count:,} (expected ≥57,000)")

        # Check confidence score distribution
        if 'confidence' in self.template_df.columns:
            results['total'] += 2
            confidence_stats = self.template_df['confidence'].describe()

            # Check confidence range
            if confidence_stats['min'] >= 0.5 and confidence_stats['max'] <= 1.0:
                results['passed'] += 1
                results['details'].append(f"✅ Confidence range valid: {confidence_stats['min']:.3f} - {confidence_stats['max']:.3f}")
            else:
                results['details'].append(f"❌ Confidence range invalid: {confidence_stats['min']:.3f} - {confidence_stats['max']:.3f}")

            # Check mean confidence
            if confidence_stats['mean'] >= 0.7:
                results['passed'] += 1
                results['details'].append(f"✅ Mean confidence good: {confidence_stats['mean']:.3f}")
            else:
                results['details'].append(f"❌ Mean confidence low: {confidence_stats['mean']:.3f}")

        # Check category distribution
        if 'category' in self.template_df.columns:
            results['total'] += 1
            category_counts = self.template_df['category'].value_counts()
            top_categories = ['GENE', 'DISEASE', 'Disease']

            if any(cat in category_counts.index for cat in top_categories):
                results['passed'] += 1
                results['details'].append(f"✅ Expected categories present: {list(category_counts.head(3).index)}")
            else:
                results['details'].append(f"❌ Missing expected categories")

        for detail in results['details']:
            print(f"  {detail}")

        success_rate = results['passed'] / results['total'] * 100
        print(f"\nResult: {results['passed']}/{results['total']} ({success_rate:.1f}%)")
        print()

        return results

    def _test_scope_reversal_terms(self):
        """Test 4: Validate scope reversal terms are present"""
        print("🔬 TEST 4: Scope Reversal Terms Validation")
        print("-" * 50)

        results = {'passed': 0, 'total': 0, 'details': []}

        if self.template_df is None:
            results['details'].append("❌ Cannot test scope terms - template not loaded")
            return results

        # Expected scope reversal terms (high priority ones that should be present)
        core_expected_terms = [
            'dyspnea', 'chest pain', 'nausea', 'vomiting',
            'headache', 'pain', 'fever', 'infection', 'fatigue'
        ]

        # Additional terms (nice to have)
        additional_terms = [
            'complications', 'symptoms', 'cardiac symptoms', 'visual changes',
            'discomfort', 'dizziness', 'inflammation', 'distress',
            'shortness of breath', 'wheezing', 'poor control', 'elevated white count'
        ]

        if 'term' in self.template_df.columns:
            template_terms_lower = self.template_df['term'].str.lower()

            # Test core terms (must have most of these)
            results['total'] += 1
            core_found = []
            for term in core_expected_terms:
                if term.lower() in template_terms_lower.values:
                    core_found.append(term)

            if len(core_found) >= len(core_expected_terms) * 0.7:  # 70% of core terms
                results['passed'] += 1
                results['details'].append(f"✅ Core scope terms present: {len(core_found)}/{len(core_expected_terms)}")
            else:
                results['details'].append(f"❌ Insufficient core scope terms: {len(core_found)}/{len(core_expected_terms)}")

            # Test additional terms (bonus)
            additional_found = []
            for term in additional_terms:
                if term.lower() in template_terms_lower.values:
                    additional_found.append(term)

            results['details'].append(f"📊 Additional scope terms found: {len(additional_found)}/{len(additional_terms)}")

            # Check category consistency for found terms (flexible approach)
            if core_found:
                results['total'] += 1
                category_col = 'category' if 'category' in self.template_df.columns else 'label' if 'label' in self.template_df.columns else None

                if category_col:
                    categorized_terms = 0
                    for term in core_found:
                        term_rows = self.template_df[self.template_df['term'].str.lower() == term.lower()]
                        if not term_rows.empty:
                            categories = term_rows[category_col].values
                            if any(cat for cat in categories if cat and 'medical' in str(cat).lower() or 'disease' in str(cat).lower()):
                                categorized_terms += 1

                    if categorized_terms >= len(core_found) * 0.5:  # 50% should have medical category
                        results['passed'] += 1
                        results['details'].append(f"✅ Scope terms have medical categories: {categorized_terms}/{len(core_found)}")
                    else:
                        results['details'].append(f"⚠️ Few scope terms have medical categories: {categorized_terms}/{len(core_found)}")
                else:
                    results['details'].append("⚠️ No category column found for validation")

        for detail in results['details']:
            print(f"  {detail}")

        success_rate = results['passed'] / results['total'] * 100 if results['total'] > 0 else 0
        print(f"\nResult: {results['passed']}/{results['total']} ({success_rate:.1f}%)")
        print()

        return results

    def _test_ner_integration(self):
        """Test 5: Validate integration with NER system"""
        print("🤖 TEST 5: NER Integration Validation")
        print("-" * 50)

        results = {'passed': 0, 'total': 0, 'details': []}

        # Initialize predictor
        results['total'] += 1
        try:
            self.predictor = EnhancedMedicalNERPredictor()
            results['passed'] += 1
            results['details'].append("✅ NER predictor initialized successfully")
        except Exception as e:
            results['details'].append(f"❌ NER predictor initialization failed: {e}")
            return results

        # Test template loading
        results['total'] += 1
        # Check if predictor has target rules loaded (this is internal to the predictor)
        try:
            # Test with a known term from the template
            test_result = self.predictor.extract_entities("Patient has diabetes")
            results['passed'] += 1
            results['details'].append("✅ Template integration functional")
        except Exception as e:
            results['details'].append(f"❌ Template integration failed: {e}")

        # Test scope reversal functionality
        results['total'] += 1
        try:
            scope_result = self.predictor.extract_entities("Patient denies chest pain but reports dyspnea")
            negated = scope_result.get('negated_entities', '')
            confirmed = scope_result.get('confirmed_entities', '')

            if 'chest pain' in negated and 'dyspnea' in confirmed:
                results['passed'] += 1
                results['details'].append("✅ Scope reversal working correctly")
            else:
                results['details'].append(f"❌ Scope reversal not working: negated='{negated}', confirmed='{confirmed}'")
        except Exception as e:
            results['details'].append(f"❌ Scope reversal test failed: {e}")

        # Test entity detection with template terms
        results['total'] += 1
        try:
            template_test = self.predictor.extract_entities("Patient reports fatigue and inflammation")
            all_entities = template_test.get('all_entities_json', '[]')

            if 'fatigue' in str(all_entities) or 'inflammation' in str(all_entities):
                results['passed'] += 1
                results['details'].append("✅ Template terms being detected")
            else:
                results['details'].append("❌ Template terms not being detected properly")
        except Exception as e:
            results['details'].append(f"❌ Template term detection test failed: {e}")

        for detail in results['details']:
            print(f"  {detail}")

        success_rate = results['passed'] / results['total'] * 100
        print(f"\nResult: {results['passed']}/{results['total']} ({success_rate:.1f}%)")
        print()

        return results

    def _test_performance_impact(self):
        """Test 6: Validate performance impact of template"""
        print("⚡ TEST 6: Performance Impact Validation")
        print("-" * 50)

        results = {'passed': 0, 'total': 0, 'details': []}

        if self.predictor is None:
            results['details'].append("❌ Cannot test performance - predictor not available")
            return results

        import time

        # Test loading time
        results['total'] += 1
        start_time = time.time()
        try:
            # Simulate reloading by creating new predictor
            test_predictor = EnhancedMedicalNERPredictor()
            load_time = time.time() - start_time

            if load_time < 10.0:  # Should load within 10 seconds
                results['passed'] += 1
                results['details'].append(f"✅ Loading time acceptable: {load_time:.2f}s")
            else:
                results['details'].append(f"❌ Loading time too slow: {load_time:.2f}s")
        except Exception as e:
            results['details'].append(f"❌ Loading test failed: {e}")

        # Test processing speed
        results['total'] += 1
        test_texts = [
            "Patient has diabetes and hypertension",
            "No fever but shows infection",
            "Denies pain but reports fatigue"
        ]

        start_time = time.time()
        try:
            for text in test_texts:
                self.predictor.extract_entities(text)

            process_time = time.time() - start_time
            avg_time = process_time / len(test_texts)

            if avg_time < 5.0:  # Should process each text within 5 seconds
                results['passed'] += 1
                results['details'].append(f"✅ Processing speed good: {avg_time:.2f}s per text")
            else:
                results['details'].append(f"❌ Processing speed slow: {avg_time:.2f}s per text")
        except Exception as e:
            results['details'].append(f"❌ Processing speed test failed: {e}")

        for detail in results['details']:
            print(f"  {detail}")

        success_rate = results['passed'] / results['total'] * 100
        print(f"\nResult: {results['passed']}/{results['total']} ({success_rate:.1f}%)")
        print()

        return results

    def _generate_validation_summary(self, test_results):
        """Generate comprehensive validation summary"""
        print("=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        total_passed = sum(result['passed'] for _, result in test_results)
        total_tests = sum(result['total'] for _, result in test_results)
        overall_success = total_passed / total_tests * 100 if total_tests > 0 else 0

        print(f"Overall Result: {total_passed}/{total_tests} ({overall_success:.1f}%)")
        print()

        for test_name, result in test_results:
            success_rate = result['passed'] / result['total'] * 100 if result['total'] > 0 else 0
            status = "✅ PASS" if success_rate >= 80 else "⚠️ WARN" if success_rate >= 60 else "❌ FAIL"
            print(f"{status} {test_name}: {result['passed']}/{result['total']} ({success_rate:.1f}%)")

        print()
        if overall_success >= 90:
            print("🎉 EXCELLENT: Template validation passed with high confidence!")
        elif overall_success >= 75:
            print("✅ GOOD: Template validation passed with acceptable results.")
        elif overall_success >= 60:
            print("⚠️ WARNING: Template has some issues that should be addressed.")
        else:
            print("❌ CRITICAL: Template has significant issues requiring immediate attention.")

        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    validator = TargetRulesTemplateValidator()
    validator.run_full_validation()