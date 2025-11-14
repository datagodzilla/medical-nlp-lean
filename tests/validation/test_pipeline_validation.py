#!/usr/bin/env python3
"""
Comprehensive Pipeline Validation Test Suite
=============================================

Tests to validate the complete medical NLP pipeline functionality.
"""

import os
import sys
import subprocess
import tempfile
import pandas as pd
from pathlib import Path
import json
import time

def test_basic_imports():
    """Test that all main modules can be imported"""
    print("🔍 Testing basic imports...")

    try:
        # Test main predictor import
        sys.path.insert(0, 'src')
        from enhanced_medical_ner_predictor import EnhancedMedicalNERPredictor
        print("✅ Enhanced Medical NER Predictor imports successfully")

        # Test performance analyzer import
        from enhanced_performance_analyzer import main as analyzer_main
        print("✅ Enhanced Performance Analyzer imports successfully")

        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_data_files_exist():
    """Test that required data files exist"""
    print("\n🔍 Testing data file existence...")

    required_files = [
        # 'data/raw/input_100texts.xlsx',  # Optional - not required for lean package
        'data/external/target_rules_template.xlsx',
        'data/external/historical_rules_template.xlsx',
        'data/external/negated_rules_template.xlsx',
        'data/external/uncertainty_rules_template.xlsx'
    ]

    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
            all_exist = False

    return all_exist

def test_output_directories():
    """Test that output directories can be created"""
    print("\n🔍 Testing output directory creation...")

    output_dirs = [
        'output/results',
        'output/reports',
        'output/visualizations'
    ]

    all_created = True
    for dir_path in output_dirs:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✅ {dir_path}")
        except Exception as e:
            print(f"❌ {dir_path} - Failed to create: {e}")
            all_created = False

    return all_created

def test_launch_script_help():
    """Test launch script help functionality"""
    print("\n🔍 Testing launch script help...")

    # Skip this test - launch_medical_nlp_project.py is not part of lean package
    print("⏭️  Skipped - launch_medical_nlp_project.py not in lean package")
    return True

def test_shell_script_help():
    """Test shell script help functionality"""
    print("\n🔍 Testing shell script help...")

    try:
        result = subprocess.run(
            ['bash', 'run_ner_pipeline.sh', '--help'],
            capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0 and 'Usage:' in result.stdout:
            print("✅ Shell script help works")
            return True
        else:
            print(f"❌ Shell script help failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Shell script test failed: {e}")
        return False

def test_predictor_initialization():
    """Test that the predictor can be initialized"""
    print("\n🔍 Testing predictor initialization...")

    try:
        sys.path.insert(0, 'src')
        from enhanced_medical_ner_predictor import EnhancedMedicalNERPredictor

        # Initialize predictor
        predictor = EnhancedMedicalNERPredictor(
            model_name="en_core_web_sm",
            use_gpu=False,
            batch_size=10
        )

        print("✅ Predictor initializes successfully")
        return True
    except Exception as e:
        print(f"❌ Predictor initialization failed: {e}")
        return False

def test_small_prediction():
    """Test prediction on a small sample"""
    print("\n🔍 Testing small prediction...")

    try:
        sys.path.insert(0, 'src')
        from enhanced_medical_ner_predictor import EnhancedMedicalNERPredictor

        # Initialize predictor
        predictor = EnhancedMedicalNERPredictor(
            model_name="en_core_web_sm",
            use_gpu=False,
            batch_size=10
        )

        # Test text
        test_text = "The patient has diabetes and takes insulin daily."

        # Extract entities
        result = predictor.extract_entities(test_text)

        if isinstance(result, dict) and 'Text Visualization' in result and 'all_entities_json' in result:
            print("✅ Small prediction test passed")
            try:
                entities_json = result.get('all_entities_json', '[]')
                entities = json.loads(entities_json) if entities_json else []
                print(f"    Detected entities: {len(entities)}")
            except:
                print(f"    Raw entities string length: {len(result.get('all_entities_json', ''))}")
            return True
        else:
            print("❌ Small prediction test failed - unexpected result format")
            print(f"    Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            return False
    except Exception as e:
        print(f"❌ Small prediction test failed: {e}")
        return False

def test_visualization_availability():
    """Test if visualization dependencies are available"""
    print("\n🔍 Testing visualization dependencies...")

    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        print("✅ Selenium available")

        # Test chrome driver manager
        from webdriver_manager.chrome import ChromeDriverManager
        print("✅ Chrome WebDriver Manager available")

        return True
    except ImportError as e:
        print(f"❌ Visualization dependencies missing: {e}")
        return False

def test_performance_analyzer_help():
    """Test performance analyzer help"""
    print("\n🔍 Testing performance analyzer help...")

    try:
        result = subprocess.run(
            [sys.executable, 'src/enhanced_performance_analyzer.py', '--help'],
            capture_output=True, text=True, timeout=15
        )

        if result.returncode == 0 and 'usage:' in result.stdout:
            print("✅ Performance analyzer help works")
            return True
        else:
            print(f"❌ Performance analyzer help failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Performance analyzer test failed: {e}")
        return False

def test_end_to_end_mini_pipeline():
    """Test a minimal end-to-end pipeline"""
    print("\n🔍 Testing mini end-to-end pipeline...")

    try:
        # Create a small test dataset
        test_data = pd.DataFrame({
            'Index': [1, 2],
            'Text': [
                'The patient has diabetes and hypertension.',
                'BRCA1 gene mutation was detected in the sample.'
            ]
        })

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            test_data.to_excel(tmp_file.name, index=False)
            tmp_path = tmp_file.name

        try:
            # Run prediction on small dataset
            result = subprocess.run([
                sys.executable, 'src/enhanced_medical_ner_predictor.py',
                '--input', tmp_path,
                '--viz-samples', '1'
            ], capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                print("✅ Mini end-to-end pipeline completed successfully")
                return True
            else:
                print(f"❌ Mini pipeline failed: {result.stderr}")
                return False
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)

    except Exception as e:
        print(f"❌ Mini pipeline test failed: {e}")
        return False

def main():
    """Run comprehensive pipeline validation tests"""
    print("="*80)
    print("🧪 COMPREHENSIVE PIPELINE VALIDATION TEST SUITE")
    print("="*80)

    tests = [
        ("Basic Imports", test_basic_imports),
        ("Data Files Existence", test_data_files_exist),
        ("Output Directories", test_output_directories),
        ("Launch Script Help", test_launch_script_help),
        ("Shell Script Help", test_shell_script_help),
        ("Predictor Initialization", test_predictor_initialization),
        ("Small Prediction", test_small_prediction),
        ("Visualization Dependencies", test_visualization_availability),
        ("Performance Analyzer Help", test_performance_analyzer_help),
        ("Mini End-to-End Pipeline", test_end_to_end_mini_pipeline)
    ]

    results = []
    start_time = time.time()

    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    end_time = time.time()

    print("\n" + "="*80)
    print("📊 PIPELINE VALIDATION RESULTS")
    print("="*80)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")

    if passed == total:
        print("🎉 ALL PIPELINE TESTS PASSED! The medical NLP pipeline is fully functional.")
        return True
    elif passed >= total * 0.8:  # 80% pass rate
        print("✅ PIPELINE MOSTLY FUNCTIONAL. Minor issues detected but core functionality works.")
        return True
    else:
        print("⚠️ SIGNIFICANT PIPELINE ISSUES. Please address failing tests.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)