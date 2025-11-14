#!/usr/bin/env python3
"""
Shell Script Integration Tests
===============================

Tests that validate the run_ner_pipeline.sh shell script uses correct paths
and executes properly. These tests ensure the user-facing CLI works correctly.

This test category was added to catch issues that unit/integration tests might miss
by testing the actual interface users interact with.
"""

import subprocess
import sys
from pathlib import Path


def test_shell_script_paths():
    """Validate shell script references correct Python script paths"""
    print("\n🔍 Testing shell script path configuration...")

    # Read shell script
    script_path = Path('run_ner_pipeline.sh')
    if not script_path.exists():
        print("❌ run_ner_pipeline.sh not found")
        return False

    with open(script_path, 'r') as f:
        content = f.read()

    # Check for correct paths with src/ prefix
    issues = []

    # Should have src/ prefix for predictor
    if 'src/enhanced_medical_ner_predictor.py' not in content:
        issues.append("❌ Missing: src/enhanced_medical_ner_predictor.py")

    # Should have src/ prefix for analyzer
    if 'src/enhanced_performance_analyzer.py' not in content:
        issues.append("❌ Missing: src/enhanced_performance_analyzer.py")

    # Should have src/ prefix for basic predictor (if used)
    if 'medical_ner_predictor.py' in content and 'src/medical_ner_predictor.py' not in content:
        issues.append("❌ Missing: src/medical_ner_predictor.py")

    if issues:
        print("❌ Shell script path issues found:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("✅ Shell script uses correct paths with src/ prefix")
        return True


def test_shell_script_help():
    """Test shell script help command executes without errors"""
    print("\n🔍 Testing shell script help command...")

    try:
        result = subprocess.run(
            ['bash', './run_ner_pipeline.sh', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0 and 'Usage:' in result.stdout:
            print("✅ Shell script help command works")
            return True
        else:
            print(f"❌ Shell script help failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Shell script help test failed: {e}")
        return False


def test_shell_script_dry_run():
    """Test shell script dry-run mode shows correct paths"""
    print("\n🔍 Testing shell script dry-run mode...")

    try:
        result = subprocess.run(
            ['bash', './run_ner_pipeline.sh', '--dry-run', '--run'],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Check that dry-run output shows src/ prefix
        if 'src/enhanced_medical_ner_predictor.py' in result.stdout:
            print("✅ Shell script dry-run shows correct paths")
            return True
        else:
            print(f"❌ Shell script dry-run missing src/ prefix in output")
            print(f"   Output: {result.stdout[:200]}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Shell script dry-run timed out")
        return False
    except Exception as e:
        print(f"❌ Shell script dry-run test failed: {e}")
        return False


def test_shell_script_status():
    """Test shell script status command"""
    print("\n🔍 Testing shell script status command...")

    try:
        result = subprocess.run(
            ['bash', './run_ner_pipeline.sh', '--status'],
            capture_output=True,
            text=True,
            timeout=15
        )

        if result.returncode == 0:
            print("✅ Shell script status command works")
            return True
        else:
            print(f"❌ Shell script status failed with exit code {result.returncode}")
            return False

    except Exception as e:
        print(f"❌ Shell script status test failed: {e}")
        return False


def test_shell_script_validate():
    """Test shell script validate command"""
    print("\n🔍 Testing shell script validate command...")

    try:
        result = subprocess.run(
            ['bash', './run_ner_pipeline.sh', '--validate'],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Validate command should check environment and dependencies
        if 'Environment check' in result.stdout or 'Validating' in result.stdout:
            print("✅ Shell script validate command works")
            return True
        else:
            print(f"❌ Shell script validate output unexpected")
            print(f"   Output: {result.stdout[:200]}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Shell script validate timed out")
        return False
    except Exception as e:
        print(f"❌ Shell script validate test failed: {e}")
        return False


def main():
    """Run all shell script integration tests"""
    print("=" * 80)
    print("🐚 SHELL SCRIPT INTEGRATION TESTS")
    print("=" * 80)

    tests = [
        ("Shell Script Paths", test_shell_script_paths),
        ("Shell Script Help", test_shell_script_help),
        ("Shell Script Dry Run", test_shell_script_dry_run),
        ("Shell Script Status", test_shell_script_status),
        ("Shell Script Validate", test_shell_script_validate),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("📊 SHELL SCRIPT TEST RESULTS")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 EXCELLENT: All shell script tests passed!")
        print("   ➡️ Shell script is correctly configured")
        sys.exit(0)
    elif passed >= total * 0.8:
        print("✅ GOOD: Most shell script tests passed")
        print("   ➡️ Review failed tests")
        sys.exit(1)
    else:
        print("❌ CRITICAL: Many shell script tests failed")
        print("   ➡️ Shell script needs attention")
        sys.exit(1)


if __name__ == "__main__":
    main()
