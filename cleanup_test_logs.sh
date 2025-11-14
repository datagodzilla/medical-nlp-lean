#!/bin/bash
################################################################################
# Medical NLP Lean Package - Test Log Cleanup
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/tests/test_logs"
AGE_DAYS="${1:-30}"

echo "Cleaning up test logs older than $AGE_DAYS days..."

if [[ ! -d "$LOG_DIR" ]]; then
    echo "Log directory not found: $LOG_DIR"
    exit 0
fi

# Remove old log files
REMOVED=0
while IFS= read -r -d '' file; do
    echo "Removing: $(basename "$file")"
    rm -f "$file"
    REMOVED=$((REMOVED + 1))
done < <(find "$LOG_DIR" -name "*.log" -o -name "test_summary_*.txt" -type f -mtime +"$AGE_DAYS" -print0)

if [[ $REMOVED -eq 0 ]]; then
    echo "No old log files found"
else
    echo "Removed $REMOVED log files"
fi
