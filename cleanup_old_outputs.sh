#!/bin/bash
#
# cleanup_old_outputs.sh - Archive old output files and directories
#
# This script moves files older than a specified number of days from output directories
# to an archive directory with timestamps. It helps maintain a clean workspace while
# preserving historical data.
#
# Usage:
#   ./cleanup_old_outputs.sh [AGE_IN_DAYS]
#
# Parameters:
#   AGE_IN_DAYS  - Archive files older than this many days (default: 3)
#
# Examples:
#   ./cleanup_old_outputs.sh           # Archive files older than 3 days
#   ./cleanup_old_outputs.sh 7         # Archive files older than 7 days
#   ./cleanup_old_outputs.sh 1         # Archive files older than 1 day

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BRIGHT_GREEN='\033[1;32m'
BRIGHT_CYAN='\033[1;36m'
NC='\033[0m'

# Default configuration
AGE_DAYS=${1:-3}  # Default to 3 days if not specified
ARCHIVE_ROOT="output/archives"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Directories to clean
CLEANUP_DIRS=(
    "output/results"
    "output/logs"
    "output/visualizations"
    "output/reports"
)

# Print colored output
print_header() {
    echo -e "${BRIGHT_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BRIGHT_CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_status() {
    echo -e "${BRIGHT_GREEN}✓${NC} ${GREEN}$1${NC}"
}

print_info() {
    echo -e "${CYAN}➜${NC} $1"
}

# Print banner
echo ""
print_header "🧹 Output Cleanup & Archival Script"
echo ""
echo -e "${CYAN}Configuration:${NC}"
echo -e "  Archive Age:     ${YELLOW}${AGE_DAYS} days${NC}"
echo -e "  Archive Root:    ${YELLOW}${ARCHIVE_ROOT}${NC}"
echo -e "  Timestamp:       ${YELLOW}${TIMESTAMP}${NC}"
echo ""

# Create archive root directory
mkdir -p "$ARCHIVE_ROOT"
print_status "Archive directory ready: $ARCHIVE_ROOT"
echo ""

# Track statistics
total_files_archived=0
total_size_archived=0

# Process each directory
for dir in "${CLEANUP_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        print_info "Skipping $dir (directory does not exist)"
        continue
    fi

    dir_name=$(basename "$dir")
    archive_subdir="${ARCHIVE_ROOT}/${dir_name}_${TIMESTAMP}"

    echo -e "${CYAN}Processing: ${NC}$dir"

    # Find files older than AGE_DAYS
    old_files=$(find "$dir" -type f -mtime +$AGE_DAYS 2>/dev/null)

    if [ -z "$old_files" ]; then
        echo -e "  ${GREEN}✓${NC} No files older than ${AGE_DAYS} days found"
        echo ""
        continue
    fi

    # Count and calculate size
    file_count=$(echo "$old_files" | wc -l | tr -d ' ')

    # Create archive subdirectory
    mkdir -p "$archive_subdir"

    # Move files
    moved_count=0
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            # Get file size before moving
            if [[ "$OSTYPE" == "darwin"* ]]; then
                # macOS
                file_size=$(stat -f%z "$file" 2>/dev/null || echo 0)
            else
                # Linux
                file_size=$(stat -c%s "$file" 2>/dev/null || echo 0)
            fi

            # Preserve directory structure within archive
            rel_path="${file#$dir/}"
            target_dir="${archive_subdir}/$(dirname "$rel_path")"
            mkdir -p "$target_dir"

            # Move file
            if mv "$file" "$target_dir/" 2>/dev/null; then
                ((moved_count++))
                ((total_size_archived+=file_size))
            fi
        fi
    done <<< "$old_files"

    if [ $moved_count -gt 0 ]; then
        print_status "Archived $moved_count file(s) to $archive_subdir"
        ((total_files_archived+=moved_count))
    fi
    echo ""
done

# Convert bytes to human-readable format
format_size() {
    local size=$1
    if [ $size -lt 1024 ]; then
        echo "${size}B"
    elif [ $size -lt 1048576 ]; then
        echo "$(( size / 1024 ))KB"
    elif [ $size -lt 1073741824 ]; then
        echo "$(( size / 1048576 ))MB"
    else
        echo "$(( size / 1073741824 ))GB"
    fi
}

# Print summary
print_header "📊 Cleanup Summary"
echo ""
echo -e "${CYAN}Total Files Archived:${NC}  ${YELLOW}${total_files_archived}${NC}"
echo -e "${CYAN}Total Size Archived:${NC}   ${YELLOW}$(format_size $total_size_archived)${NC}"
echo -e "${CYAN}Archive Location:${NC}      ${YELLOW}${ARCHIVE_ROOT}${NC}"
echo ""

if [ $total_files_archived -gt 0 ]; then
    print_status "Cleanup completed successfully!"
else
    print_info "No files were archived (nothing older than ${AGE_DAYS} days)"
fi

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
