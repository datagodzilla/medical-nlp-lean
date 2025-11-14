#!/bin/bash
################################################################################
# Medical NLP Lean Package - Test Runner
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="py311_bionlp"
TEST_SCRIPT="$SCRIPT_DIR/tests/master_test_script.py"

# Check if test script exists
if [[ ! -f "$TEST_SCRIPT" ]]; then
    echo "Error: Master test script not found: $TEST_SCRIPT"
    exit 1
fi

# Run tests
echo "Running Medical NLP Test Suite..."
echo "Logs will be saved to: tests/test_logs/"
echo ""

conda run -n "$CONDA_ENV" python "$TEST_SCRIPT" "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ All tests passed"
else
    echo ""
    echo "✗ Some tests failed - check logs in tests/test_logs/"
fi

exit $EXIT_CODE
