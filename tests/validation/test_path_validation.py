#!/usr/bin/env python3
"""
Path Validation Test Script
===========================

Comprehensive test to validate that all absolute paths have been replaced
with relative paths and dynamic path detection in the project.
"""

import os
import re
import sys
from pathlib import Path
import subprocess

def test_absolute_paths_in_files():
    """Test for absolute paths in project files"""
    print("🔍 Testing for absolute paths in project files...")

    # File patterns to check
    file_patterns = ['*.py', '*.sh', '*.yml', '*.yaml', '*.json']

    # Absolute path patterns to look for
    absolute_patterns = [
        r'project_root  # /Users/username paths
        r'/home/[^/\s]+',   # /home/username paths
        r'/opt/[^/\s]+',    # /opt paths
        r'/usr/local/[^/\s]+', # /usr/local paths
        r'/var/[^/\s]+',    # /var paths
    ]

    issues_found = []
    project_root = Path.cwd()

    for pattern in file_patterns:
        for file_path in project_root.glob(f'**/{pattern}'):
            # Skip certain directories and files
            if any(skip_dir in str(file_path) for skip_dir in ['.git', '__pycache__', 'test_relative_paths.py', 'test_path_validation.py']):
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                for line_num, line in enumerate(content.split('\n'), 1):
                    # Skip lines that are legitimate fallback paths or comments
                    if any(skip_pattern in line for skip_pattern in [
                        '/opt/anaconda3',  # Standard conda fallback location
                        '/opt/miniconda3', # Standard miniconda fallback location
                        '# /Users/',       # Comment examples
                        '# /home/',        # Comment examples
                        '"#'               # Documentation strings
                    ]):
                        continue

                    for abs_pattern in absolute_patterns:
                        matches = re.findall(abs_pattern, line)
                        if matches:
                            issues_found.append({
                                'file': str(file_path.relative_to(project_root)),
                                'line': line_num,
                                'content': line.strip(),
                                'matches': matches
                            })
            except Exception as e:
                print(f"⚠️ Could not read {file_path}: {e}")

    if issues_found:
        print("❌ Found absolute paths:")
        for issue in issues_found:
            print(f"  📁 {issue['file']}:{issue['line']}")
            print(f"     {issue['content']}")
            print(f"     Matches: {issue['matches']}")
        return False
    else:
        print("✅ No absolute paths found in project files")
        return True

def test_dynamic_conda_detection():
    """Test dynamic conda detection in scripts"""
    print("\n🔍 Testing dynamic conda detection...")

    # Test launch script
    try:
        result = subprocess.run([sys.executable, 'launch_medical_nlp_project.py', '--status'],
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Launch script works with dynamic conda detection")
        else:
            print(f"❌ Launch script failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Launch script test failed: {e}")
        return False

    # Test shell script help
    try:
        result = subprocess.run(['bash', 'run_ner_pipeline.sh', '--help'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Shell script works with dynamic conda detection")
        else:
            print(f"❌ Shell script failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Shell script test failed: {e}")
        return False

    return True

def test_relative_path_usage():
    """Test that relative paths are used correctly"""
    print("\n🔍 Testing relative path usage...")

    # Check that data directory references are relative
    expected_relative_paths = [
        'data/raw/',
        'data/external/',
        'output/',
        'models/',
        'configs/'
    ]

    found_relative_paths = []
    project_root = Path.cwd()

    for py_file in project_root.glob('**/*.py'):
        if any(skip_dir in str(py_file) for skip_dir in ['.git', '__pycache__']):
            continue

        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

            for rel_path in expected_relative_paths:
                if rel_path in content:
                    found_relative_paths.append(rel_path)
        except Exception:
            continue

    if found_relative_paths:
        print(f"✅ Found relative paths in use: {set(found_relative_paths)}")
        return True
    else:
        print("❌ No relative paths found")
        return False

def test_file_accessibility():
    """Test that key files are accessible with relative paths"""
    print("\n🔍 Testing file accessibility...")

    key_files = [
        'enhanced_medical_ner_predictor.py',
        'enhanced_performance_analyzer.py',
        'data/external/target_rules_template.xlsx',
        'data/external/historical_rules_template.xlsx',
        'data/external/negated_rules_template.xlsx',
        'data/external/uncertainty_rules_template.xlsx'
    ]

    all_accessible = True
    for file_path in key_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} not found")
            all_accessible = False

    return all_accessible

def main():
    """Run comprehensive path validation tests"""
    print("="*80)
    print("🧪 COMPREHENSIVE PATH VALIDATION TEST")
    print("="*80)

    tests = [
        ("Absolute Paths Test", test_absolute_paths_in_files),
        ("Dynamic Conda Detection Test", test_dynamic_conda_detection),
        ("Relative Path Usage Test", test_relative_path_usage),
        ("File Accessibility Test", test_file_accessibility)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    print("\n" + "="*80)
    print("📊 TEST RESULTS SUMMARY")
    print("="*80)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Project uses relative paths correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please review absolute path usage.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)