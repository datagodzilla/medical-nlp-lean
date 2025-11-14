#!/usr/bin/env python3
"""
Master Test Script for Enhanced Medical NER Pipeline
===================================================

This script runs all test suites for the enhanced medical NER pipeline,
including scope reversal analysis, template validation, and consistency checks.
"""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

class MasterTestRunner:
    """Runs comprehensive test suite for the enhanced medical NER pipeline"""

    def __init__(self):
        # Get the directory where this script is located
        self.script_dir = Path(__file__).parent

        self.test_scripts = [
            {
                'name': 'Scope Reversal Test v2',
                'script': 'features/test_scope_reversal_v2.py',
                'description': 'Enhanced scope reversal detection tests',
                'category': 'scope_reversal'
            },
            {
                'name': 'Scope Reversal Test (Original)',
                'script': 'features/test_scope_reversal.py',
                'description': 'Original comprehensive scope reversal test suite',
                'category': 'scope_reversal'
            },
            {
                'name': 'Context Classification Tests',
                'script': 'features/test_context_classifications.py',
                'description': 'Context type classification validation',
                'category': 'context'
            },
            {
                'name': 'Negation Detection Tests',
                'script': 'features/test_negation.py',
                'description': 'Negation pattern detection and scope',
                'category': 'negation'
            },
            {
                'name': 'Template Pattern Validation',
                'script': 'integration/test_template_patterns.py',
                'description': 'Comprehensive template structure and content validation',
                'category': 'template'
            },
            {
                'name': 'Consistency Test',
                'script': 'integration/test_consistency.py',
                'description': 'Cross-platform consistency validation',
                'category': 'consistency'
            },
            {
                'name': 'Context Overlap Resolution',
                'script': 'features/test_context_overlap.py',
                'description': 'Context priority and overlap handling',
                'category': 'context'
            },
            {
                'name': 'Confidence & Word Boundaries',
                'script': 'features/test_confidence_boundaries.py',
                'description': 'Confidence scoring and word boundary validation',
                'category': 'confidence'
            },
            {
                'name': 'Streamlit Display Tests',
                'script': 'app/test_streamlit_display.py',
                'description': 'Streamlit UI component validation',
                'category': 'ui'
            },
            {
                'name': 'Excel Formatting Tests',
                'script': 'excel_output/test_excel_formatting.py',
                'description': 'Excel output format and text marker validation',
                'category': 'output'
            },
            {
                'name': 'Visualization Tests',
                'script': 'visualization/test_full_viz.py',
                'description': 'DisplaCy visualization rendering',
                'category': 'visualization'
            },
            {
                'name': 'Pipeline Validation',
                'script': 'validation/test_pipeline_validation.py',
                'description': 'End-to-end pipeline validation',
                'category': 'pipeline'
            },
            {
                'name': 'Shell Script Integration',
                'script': 'integration/test_shell_script.py',
                'description': 'Shell script path validation and CLI testing',
                'category': 'shell'
            }
        ]

    def run_all_tests(self, categories=None):
        """Run all tests or specific categories"""
        print("=" * 80)
        print("ENHANCED MEDICAL NER PIPELINE - MASTER TEST SUITE")
        print("=" * 80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        if categories:
            tests_to_run = [t for t in self.test_scripts if t['category'] in categories]
            print(f"Running tests for categories: {', '.join(categories)}")
        else:
            tests_to_run = self.test_scripts
            print("Running all available tests")

        print(f"Total test scripts: {len(tests_to_run)}")
        print()

        results = []
        for i, test in enumerate(tests_to_run, 1):
            print(f"📋 Test {i}/{len(tests_to_run)}: {test['name']}")
            print(f"   Script: {test['script']}")
            print(f"   Description: {test['description']}")
            print(f"   Category: {test['category']}")

            # Check if script exists (relative to tests directory)
            script_path = self.script_dir / test['script']
            if not script_path.exists():
                result = {
                    'name': test['name'],
                    'script': test['script'],
                    'category': test['category'],
                    'status': 'MISSING',
                    'returncode': -1,
                    'output': f'Script file not found at: {script_path}',
                    'error': ''
                }
                results.append(result)
                print(f"   ❌ MISSING: Script file not found at {script_path}")
                print()
                continue

            # Run the test
            print(f"   🚀 Running...")
            try:
                # Use conda environment for consistent execution
                cmd = ['conda', 'run', '-n', 'py311_bionlp', 'python', str(script_path)]

                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )

                status = 'PASS' if process.returncode == 0 else 'FAIL'
                status_icon = '✅' if status == 'PASS' else '❌'

                result = {
                    'name': test['name'],
                    'script': test['script'],
                    'category': test['category'],
                    'status': status,
                    'returncode': process.returncode,
                    'output': process.stdout,
                    'error': process.stderr
                }
                results.append(result)

                print(f"   {status_icon} {status} (exit code: {process.returncode})")

                # Show brief output for failed tests
                if status == 'FAIL':
                    print(f"   Error preview: {process.stderr[:200]}..." if len(process.stderr) > 200 else f"   Error: {process.stderr}")

            except subprocess.TimeoutExpired:
                result = {
                    'name': test['name'],
                    'script': test['script'],
                    'category': test['category'],
                    'status': 'TIMEOUT',
                    'returncode': -2,
                    'output': '',
                    'error': 'Test timed out after 5 minutes'
                }
                results.append(result)
                print(f"   ⏱️ TIMEOUT: Test exceeded 5-minute limit")

            except Exception as e:
                result = {
                    'name': test['name'],
                    'script': test['script'],
                    'category': test['category'],
                    'status': 'ERROR',
                    'returncode': -3,
                    'output': '',
                    'error': str(e)
                }
                results.append(result)
                print(f"   💥 ERROR: {e}")

            print()

        # Generate summary
        self._generate_summary(results)

        # Save detailed results
        self._save_detailed_results(results)

        return results

    def _generate_summary(self, results):
        """Generate test execution summary"""
        print("=" * 80)
        print("TEST EXECUTION SUMMARY")
        print("=" * 80)

        # Overall statistics
        total_tests = len(results)
        passed_tests = len([r for r in results if r['status'] == 'PASS'])
        failed_tests = len([r for r in results if r['status'] == 'FAIL'])
        missing_tests = len([r for r in results if r['status'] == 'MISSING'])
        timeout_tests = len([r for r in results if r['status'] == 'TIMEOUT'])
        error_tests = len([r for r in results if r['status'] == 'ERROR'])

        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"❌ Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        print(f"📁 Missing: {missing_tests} ({missing_tests/total_tests*100:.1f}%)")
        print(f"⏱️ Timeout: {timeout_tests} ({timeout_tests/total_tests*100:.1f}%)")
        print(f"💥 Error: {error_tests} ({error_tests/total_tests*100:.1f}%)")
        print()

        # Category breakdown
        categories = {}
        for result in results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = {'total': 0, 'passed': 0}
            categories[cat]['total'] += 1
            if result['status'] == 'PASS':
                categories[cat]['passed'] += 1

        print("Results by Category:")
        for category, stats in sorted(categories.items()):
            success_rate = stats['passed'] / stats['total'] * 100 if stats['total'] > 0 else 0
            status_icon = '✅' if success_rate >= 80 else '⚠️' if success_rate >= 50 else '❌'
            print(f"  {status_icon} {category}: {stats['passed']}/{stats['total']} ({success_rate:.1f}%)")

        print()

        # Individual test results
        print("Individual Test Results:")
        for result in results:
            status_icons = {
                'PASS': '✅',
                'FAIL': '❌',
                'MISSING': '📁',
                'TIMEOUT': '⏱️',
                'ERROR': '💥'
            }
            icon = status_icons.get(result['status'], '❓')
            print(f"  {icon} {result['name']} ({result['status']})")

        print()

        # Overall assessment with recommendations
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        if passed_tests == total_tests:
            print("🎉 EXCELLENT: All tests passed!")
            print("   ➡️ System is ready for production use")
        elif success_rate >= 0.8:
            print("✅ GOOD: Most tests passed successfully.")
            print("   ➡️ System is stable, review failed tests for improvements")
        elif success_rate >= 0.6:
            print("⚠️ WARNING: Some tests failed. Review needed.")
            print("   ➡️ Check failed tests, particularly high-priority ones")
        else:
            print("❌ CRITICAL: Many tests failed. Immediate attention required.")
            print("   ➡️ Focus on core functionality before proceeding")

        # Specific guidance for scope reversal
        scope_tests = [r for r in results if 'scope' in r['category']]
        if scope_tests:
            scope_passed = len([r for r in scope_tests if r['status'] == 'PASS'])
            scope_total = len(scope_tests)
            scope_rate = scope_passed / scope_total if scope_total > 0 else 0

            print(f"\n🔬 SCOPE REVERSAL ANALYSIS:")
            if scope_rate >= 0.8:
                print(f"   ✅ Scope reversal working well: {scope_passed}/{scope_total} ({scope_rate*100:.1f}%)")
            elif scope_rate >= 0.6:
                print(f"   ⚠️ Scope reversal needs tuning: {scope_passed}/{scope_total} ({scope_rate*100:.1f}%)")
            else:
                print(f"   ❌ Scope reversal needs major fixes: {scope_passed}/{scope_total} ({scope_rate*100:.1f}%)")

        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def _save_detailed_results(self, results):
        """Save detailed test results to file"""
        output_dir = Path("output/test_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"master_test_results_{timestamp}.txt"

        with open(results_file, 'w') as f:
            f.write("Enhanced Medical NER Pipeline - Detailed Test Results\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for result in results:
                f.write(f"Test: {result['name']}\n")
                f.write(f"Script: {result['script']}\n")
                f.write(f"Category: {result['category']}\n")
                f.write(f"Status: {result['status']}\n")
                f.write(f"Return Code: {result['returncode']}\n")
                f.write(f"Output:\n{result['output']}\n")
                f.write(f"Error:\n{result['error']}\n")
                f.write("-" * 60 + "\n\n")

        print(f"📁 Detailed results saved to: {results_file}")

    def run_quick_tests(self):
        """Run only quick/essential tests"""
        quick_categories = ['scope_reversal', 'template', 'consistency']
        return self.run_all_tests(categories=quick_categories)

    def run_scope_reversal_tests(self):
        """Run only scope reversal related tests"""
        return self.run_all_tests(categories=['scope_reversal'])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Enhanced Medical NER Pipeline tests')
    parser.add_argument('--quick', action='store_true', help='Run only quick/essential tests')
    parser.add_argument('--scope', action='store_true', help='Run only scope reversal tests')
    parser.add_argument('--category',
                       choices=['scope_reversal', 'template', 'consistency', 'ui', 'context',
                               'negation', 'confidence', 'output', 'visualization', 'pipeline', 'shell'],
                       help='Run tests for specific category')

    args = parser.parse_args()

    runner = MasterTestRunner()

    if args.quick:
        results = runner.run_quick_tests()
    elif args.scope:
        results = runner.run_scope_reversal_tests()
    elif args.category:
        results = runner.run_all_tests(categories=[args.category])
    else:
        results = runner.run_all_tests()

    # Exit with appropriate code
    failed_count = len([r for r in results if r['status'] != 'PASS'])
    sys.exit(1 if failed_count > 0 else 0)