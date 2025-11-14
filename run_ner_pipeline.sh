#!/bin/bash
#
# run_ner_pipeline.sh - Enhanced Medical NLP Pipeline Execution Script
#
# This script provides comprehensive CLI options for running the Medical NLP Pipeline
# with flexible input handling, custom configurations, and advanced processing options.
#
# FLAG OPTIONS DEFINITIONS:
#
# EXECUTION OPTIONS:
#   --run                    Execute the medical NER prediction pipeline using the specified
#                           input file and generate predictions with entity detection,
#                           negation analysis, and context classification
#
#   --setup                  Run the complete project setup including conda environment
#                           creation, dependency installation, and model downloads
#
#   --analyze                Execute performance analysis on existing prediction results,
#                           generating detailed reports with metrics and recommendations
#
#   --all                    Execute the complete end-to-end workflow: setup (if needed),
#                           prediction pipeline, and performance analysis
#
#   --validate               Validate the installation by checking conda environment,
#                           dependencies, models, and script availability
#
#   --status                 Display current project status including environment health,
#                           file availability, and directory structure
#
# INPUT/OUTPUT OPTIONS:
#   --input FILE             Specify custom input Excel file path. File must contain
#                           'Index' and 'Text' columns. All input columns are preserved
#                           in output with 15 prediction columns appended
#
#   --output FILE            Specify custom output file path for predictions. Default
#                           generates timestamped filename in output/predictions/
#
#   --text-column COL        Specify the name of the column containing clinical text
#                           to analyze (default: 'Text')
#
# PROCESSING OPTIONS:
#   --enhanced               Use enhanced_medical_ner_predictor.py with advanced features
#                           including flexible input handling, Excel-friendly visualizations,
#                           and comprehensive entity detection (default, recommended)
#
#   --basic                  Use basic medical_ner_predictor.py with standard functionality
#                           for simpler processing requirements
#
#   --model MODEL            Specify spaCy model to use for NLP processing. Options include
#                           en_core_web_sm (small, ~50MB), en_core_web_md (medium, ~50MB),
#                           or en_core_web_lg (large, ~750MB)
#
#   --batch-size SIZE        Set processing batch size for handling large datasets.
#                           Larger batches use more memory but may process faster
#
#   --gpu                    Enable GPU acceleration for processing (requires GPU-enabled
#                           spaCy installation and compatible hardware)
#
# ANALYSIS OPTIONS:
#   --performance            Include comprehensive performance analysis with precision,
#                           recall, F1-scores, and accuracy metrics for all entity types
#
#   --visualization          Generate visualizations including entity highlighting,
#                           distribution charts, and performance graphs (enabled by default,
#                           generates 5 sample visualizations)
#
#   --report                 Generate final execution report with pipeline configuration,
#                           performance summary, and recommendations
#
#   --export-json            Export prediction results in JSON format in addition to
#                           Excel output for programmatic access
#
# DEBUGGING OPTIONS:
#   --verbose                Enable detailed output showing processing steps, intermediate
#                           results, and diagnostic information (enabled by default)
#
#   --debug                  Enable debug mode with maximum verbosity, command tracing,
#                           and detailed error information (implies --verbose)
#
#   --dry-run                Show all commands that would be executed without actually
#                           running them. Useful for testing and validation
#
#   --quiet                  Suppress all non-error output for automated execution
#                           or when minimal output is desired (disables verbose mode)
#
# CONFIGURATION OPTIONS:
#   --env ENV_NAME           Specify conda environment name to use (default: py311_bionlp).
#                           Environment must contain required NLP packages
#
#   --python PATH            Specify custom Python executable path instead of using
#                           the default conda environment Python
#
#   --config FILE            Use custom configuration file for advanced pipeline settings
#                           and parameter overrides
#
# Usage Examples:
#   ./run_ner_pipeline.sh --run                           # Quick run with defaults
#   ./run_ner_pipeline.sh --input data/custom.xlsx        # Custom input file
#   ./run_ner_pipeline.sh --setup --run --analyze         # Full workflow
#   ./run_ner_pipeline.sh --enhanced --verbose            # Enhanced predictor with verbose output
#   ./run_ner_pipeline.sh --model en_core_web_md --gpu    # Use medium model with GPU
#

# Default configuration
# DEFAULT_INPUT="data/raw/context_test_results_complete.xlsx"
DEFAULT_INPUT="data/raw/input_100texts.xlsx"
DEFAULT_OUTPUT="output/results/output_results_$(date +%Y%m%d_%H%M%S).xlsx"
DEFAULT_MODEL="en_core_web_sm"
CONDA_ENV="py311_bionlp"
ARCHIVE_AGE_DAYS=3  # Archive files older than this many days

# Suppress Python warnings by default (package compatibility warnings, deprecations, etc.)
export PYTHONWARNINGS="ignore"

# Dynamically determine conda environment Python path
if command -v conda &> /dev/null; then
    CONDA_BASE=$(conda info --base)
    ENV_PYTHON="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
else
    # Fallback: try common conda installation paths
    for CONDA_BASE in "$HOME/anaconda3" "$HOME/miniconda3" "/opt/anaconda3" "/opt/miniconda3"; do
        if [ -d "$CONDA_BASE" ]; then
            ENV_PYTHON="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
            break
        fi
    done
    # Final fallback to system python
    ENV_PYTHON="${ENV_PYTHON:-python}"
fi

# Color codes for output (enhanced with bold and bright variants)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BRIGHT_GREEN='\033[1;32m'
BRIGHT_BLUE='\033[1;34m'
BRIGHT_CYAN='\033[1;36m'
BRIGHT_YELLOW='\033[1;93m'
MAGENTA='\033[0;95m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Function to print colored output with enhanced styling
print_status() {
    echo -e "${BRIGHT_GREEN}✓${NC} ${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${BRIGHT_YELLOW}⚠${NC} ${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}✗ [ERROR]${NC} $1"
}

print_header() {
    echo -e "${BRIGHT_CYAN}${BOLD}▶${NC} ${BRIGHT_BLUE}$1${NC}"
}

print_success() {
    echo -e "${BRIGHT_GREEN}${BOLD}✓✓✓${NC} ${GREEN}$1${NC}"
}

print_step() {
    echo -e "${CYAN}➜${NC} $1"
}

# Function to print banner with fixed alignment
print_banner() {
    echo ""
    echo -e "${MAGENTA}${BOLD}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                       MEDICAL NLP PIPELINE RUNNER                            ║
║                      Enhanced CLI Execution Script                           ║
║                                                                              ║
║  🏥 Medical Entity Recognition     🧬 Gene Detection                         ║
║  🚫 Negation Analysis             ⏰ Historical Context                      ║
║  👨‍👩‍👧 Family History             ❓ Uncertainty Detection                  ║
║  📊 Performance Analysis           📝 Comprehensive Reporting                ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "EXECUTION OPTIONS:"
    echo "  --run                    Run prediction pipeline only"
    echo "  --setup                  Run complete project setup"
    echo "  --analyze                Run performance analysis only"
    echo "  --all                    Run complete workflow (setup + run + analyze)"
    echo "  --validate               Validate installation"
    echo "  --status                 Show project status"
    echo ""
    echo "INPUT/OUTPUT OPTIONS:"
    echo "  --input FILE             Input Excel file (default: $DEFAULT_INPUT)"
    echo "  --output FILE            Output file path (default: auto-generated)"
    echo "  --text-column COL        Text column name (default: 'Text')"
    echo ""
    echo "PROCESSING OPTIONS:"
    echo "  --enhanced               Use enhanced_medical_ner_predictor.py (recommended)"
    echo "  --basic                  Use basic medical_ner_predictor.py"
    echo "  --model MODEL            spaCy model to use (default: $DEFAULT_MODEL)"
    echo "  --batch-size SIZE        Processing batch size (default: 100)"
    echo "  --gpu                    Enable GPU processing (if available)"
    echo ""
    echo "ANALYSIS OPTIONS:"
    echo "  --performance            Include performance analysis"
    echo "  --visualization          Generate visualizations (enabled by default - 5 samples)"
    echo "  --no-visualization       Disable visualization generation"
    echo "  --report                 Generate final report"
    echo "  --export-json            Export results as JSON"
    echo ""
    echo "DEBUGGING OPTIONS:"
    echo "  --verbose                Enable verbose output (enabled by default)"
    echo "  --debug                  Enable debug mode"
    echo "  --dry-run                Show commands without execution"
    echo "  --quiet                  Suppress non-error output (disables verbose)"
    echo ""
    echo "CONFIGURATION OPTIONS:"
    echo "  --env ENV_NAME           Conda environment name (default: $CONDA_ENV)"
    echo "  --python PATH            Python executable path"
    echo "  --config FILE            Custom configuration file"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 --run                                    # Quick run with defaults"
    echo "  $0 --input data/custom.xlsx --enhanced      # Process custom file with enhanced predictor"
    echo "  $0 --setup --run --analyze --verbose        # Full workflow with verbose output"
    echo "  $0 --model en_core_web_md --gpu --batch-size 50  # Use medium model with GPU"
    echo "  $0 --all --report --visualization           # Complete analysis with reporting"
    echo ""
    echo "For more information, see the project documentation."
}

# Initialize variables
SETUP=false
RUN=false
ANALYZE=false
ALL=false
VALIDATE=false
STATUS=false
ENHANCED=true
BASIC=false
PERFORMANCE=true  # Default to true - always generate performance reports
VISUALIZATION=true  # Default to true - always generate 5 visualizations
REPORT=false
VERBOSE=true  # Default to true - show detailed output
DEBUG=false
DRY_RUN=false
QUIET=false
GPU=false
EXPORT_JSON=false

INPUT_FILE=""
OUTPUT_FILE=""
TEXT_COLUMN="Text"
MODEL="$DEFAULT_MODEL"
BATCH_SIZE=100
PYTHON_EXEC="$ENV_PYTHON"
CONFIG_FILE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_usage
            exit 0
            ;;
        --run)
            RUN=true
            shift
            ;;
        --setup)
            SETUP=true
            shift
            ;;
        --analyze)
            ANALYZE=true
            shift
            ;;
        --all)
            ALL=true
            shift
            ;;
        --validate)
            VALIDATE=true
            shift
            ;;
        --status)
            STATUS=true
            shift
            ;;
        --enhanced)
            ENHANCED=true
            BASIC=false
            shift
            ;;
        --basic)
            ENHANCED=false
            BASIC=true
            shift
            ;;
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --text-column)
            TEXT_COLUMN="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gpu)
            GPU=true
            shift
            ;;
        --performance)
            PERFORMANCE=true
            shift
            ;;
        --visualization)
            VISUALIZATION=true
            shift
            ;;
        --no-visualization)
            VISUALIZATION=false
            shift
            ;;
        --report)
            REPORT=true
            shift
            ;;
        --export-json)
            EXPORT_JSON=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --debug)
            DEBUG=true
            VERBOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --quiet)
            QUIET=true
            shift
            ;;
        --env)
            CONDA_ENV="$2"
            # Dynamically determine conda environment Python path
if command -v conda &> /dev/null; then
    CONDA_BASE=$(conda info --base)
    ENV_PYTHON="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
else
    # Fallback: try common conda installation paths
    for CONDA_BASE in "$HOME/anaconda3" "$HOME/miniconda3" "/opt/anaconda3" "/opt/miniconda3"; do
        if [ -d "$CONDA_BASE" ]; then
            ENV_PYTHON="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
            break
        fi
    done
    # Final fallback to system python
    ENV_PYTHON="${ENV_PYTHON:-python}"
fi
            PYTHON_EXEC="$ENV_PYTHON"
            shift 2
            ;;
        --python)
            PYTHON_EXEC="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Function to truncate long strings for display
truncate_string() {
    local string="$1"
    local max_length=${2:-100}

    if [ ${#string} -gt $max_length ]; then
        echo "${string:0:$max_length}... [truncated]"
    else
        echo "$string"
    fi
}

# Function to execute command with enhanced visual feedback and output filtering
execute_command() {
    local cmd="$1"
    local description="$2"

    if [ "$DRY_RUN" = true ]; then
        echo -e "${BRIGHT_YELLOW}[DRY RUN]${NC} Would execute: ${CYAN}$description${NC}"
        local truncated_cmd=$(truncate_string "$cmd" 150)
        echo -e "${BLUE}Command:${NC} $truncated_cmd"
        return 0
    fi

    if [ "$VERBOSE" = true ] && [ "$QUIET" = false ]; then
        echo ""
        print_step "${BOLD}$description${NC}"
        local truncated_cmd=$(truncate_string "$cmd" 150)
        echo -e "${BLUE}Command:${NC} ${CYAN}$truncated_cmd${NC}"
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    elif [ "$QUIET" = false ]; then
        print_step "$description"
    fi

    if [ "$DEBUG" = true ]; then
        set -x
    fi

    # Execute command and filter output for long lines (especially HTML/visualization data)
    eval "$cmd" 2>&1 | while IFS= read -r line; do
        # Check if line contains HTML visualization or is very long
        if [[ "$line" =~ \<div.*displacy\> ]] || [[ "$line" =~ \<svg ]] || [ ${#line} -gt 200 ]; then
            # Truncate visualization/HTML output
            local truncated_line=$(truncate_string "$line" 100)
            echo "$truncated_line"
        else
            # Normal output - display as is
            echo "$line"
        fi
    done
    local exit_code=${PIPESTATUS[0]}

    if [ "$DEBUG" = true ]; then
        set +x
    fi

    if [ $exit_code -eq 0 ]; then
        if [ "$VERBOSE" = true ] && [ "$QUIET" = false ]; then
            print_success "$description completed successfully"
            echo ""
        elif [ "$QUIET" = false ]; then
            echo -e "${BRIGHT_GREEN}✓${NC}"
        fi
    else
        print_error "$description failed with exit code $exit_code"
        return $exit_code
    fi

    return 0
}

# Function to check environment
check_environment() {
    print_status "🔍 Checking conda environment: $CONDA_ENV"

    if ! conda env list | grep -q "^$CONDA_ENV "; then
        print_error "Conda environment '$CONDA_ENV' not found"
        print_status "Available environments:"
        conda env list
        return 1
    fi

    if [ ! -f "$PYTHON_EXEC" ]; then
        print_error "Python executable not found: $PYTHON_EXEC"
        return 1
    fi

    print_status "✅ Environment check passed"
    return 0
}

# Function to run launcher
run_launcher() {
    local launcher_args=""

    if [ "$SETUP" = true ]; then
        launcher_args="$launcher_args --setup"
    fi

    if [ "$RUN" = true ]; then
        launcher_args="$launcher_args --run"
    fi

    if [ "$ANALYZE" = true ]; then
        launcher_args="$launcher_args --analyze"
    fi

    if [ "$ALL" = true ]; then
        launcher_args="$launcher_args --all"
    fi

    if [ "$VALIDATE" = true ]; then
        launcher_args="$launcher_args --validate"
    fi

    if [ "$STATUS" = true ]; then
        launcher_args="$launcher_args --status"
    fi

    if [ -n "$INPUT_FILE" ]; then
        launcher_args="$launcher_args --input \"$INPUT_FILE\""
    fi

    local cmd="$PYTHON_EXEC launch_medical_nlp_project.py $launcher_args"
    execute_command "$cmd" "Running Medical NLP Launcher"
}

# Function to display output file paths
display_output_paths() {
    local output="${OUTPUT_FILE:-$DEFAULT_OUTPUT}"
    local basename=$(basename "$output" .xlsx)
    local dirname=$(dirname "$output")

    # Get relative paths from project root
    local rel_output="${output#./}"
    local rel_log="output/logs/${basename}.log"
    local rel_viz="output/visualizations/"
    local rel_reports="output/reports/"

    echo ""
    print_header "📁 Generated Output Files (Relative Paths):"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}📊 Excel Results:${NC}      $rel_output"
    echo -e "${GREEN}📝 Log File:${NC}           $rel_log"
    echo -e "${GREEN}🎨 Visualizations:${NC}     ${rel_viz}visualization_sample_idx_*.png"
    echo -e "${GREEN}📈 Reports:${NC}            ${rel_reports}entity_prediction_analysis_report_*.txt"
    if [ "$EXPORT_JSON" = true ]; then
        local rel_json="${rel_output%.xlsx}.json"
        echo -e "${GREEN}📄 JSON Export:${NC}        $rel_json"
    fi
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Function to run enhanced predictor directly
run_enhanced_predictor() {
    local predictor_args=""

    # Set input file
    local input="${INPUT_FILE:-$DEFAULT_INPUT}"
    local output="${OUTPUT_FILE:-$DEFAULT_OUTPUT}"

    predictor_args="--input \"$input\" --output \"$output\""

    if [ "$VERBOSE" = true ]; then
        predictor_args="$predictor_args --verbose"
    fi

    if [ "$GPU" = true ]; then
        predictor_args="$predictor_args --gpu"
    fi

    if [ "$EXPORT_JSON" = true ]; then
        predictor_args="$predictor_args --json"
    fi

    # Add visualization flag (default enabled with 5 samples)
    if [ "$VISUALIZATION" = true ]; then
        predictor_args="$predictor_args --visualizations --viz-samples 5"
    fi

    # Add model specification
    predictor_args="$predictor_args --model $MODEL"

    # Add batch size
    predictor_args="$predictor_args --batch-size $BATCH_SIZE"

    # Select predictor script (from src/ directory)
    local script="src/enhanced_medical_ner_predictor.py"
    if [ "$BASIC" = true ]; then
        script="src/medical_ner_predictor.py"
    fi

    local cmd="$PYTHON_EXEC $script $predictor_args"

    # Store the pipeline command and output file for the performance analyzer
    PIPELINE_COMMAND="$cmd"
    ACTUAL_OUTPUT_FILE="$output"
    export PIPELINE_COMMAND
    export ACTUAL_OUTPUT_FILE

    # Execute the predictor
    execute_command "$cmd" "Running Medical NER Predictor ($script)"

    # Display output paths after successful execution
    if [ $? -eq 0 ]; then
        display_output_paths
    fi
}

# Function to run performance analysis
run_performance_analysis() {
    local analysis_args=""

    # Use ACTUAL_OUTPUT_FILE if available (set by run_enhanced_predictor), otherwise fall back to OUTPUT_FILE
    local input_file="${ACTUAL_OUTPUT_FILE:-$OUTPUT_FILE}"

    if [ -n "$input_file" ]; then
        analysis_args="--input \"$input_file\""
    else
        print_error "No output file available for performance analysis"
        return 1
    fi

    if [ "$VERBOSE" = true ]; then
        analysis_args="$analysis_args --verbose"
    fi

    # Pass the pipeline command if available
    if [ -n "$PIPELINE_COMMAND" ]; then
        analysis_args="$analysis_args --pipeline-command \"$PIPELINE_COMMAND\""
    fi

    local cmd="$PYTHON_EXEC src/enhanced_performance_analyzer.py $analysis_args"
    execute_command "$cmd" "Running Performance Analysis"
}

# Main execution logic
main() {
    # Show banner unless quiet
    if [ "$QUIET" = false ]; then
        print_banner
    fi

    # Set default action if none specified
    if [ "$SETUP" = false ] && [ "$RUN" = false ] && [ "$ANALYZE" = false ] && [ "$ALL" = false ] && [ "$VALIDATE" = false ] && [ "$STATUS" = false ]; then
        print_status "No action specified, defaulting to --run"
        RUN=true
    fi

    # Check environment
    if ! check_environment; then
        print_error "Environment check failed. Please run setup first."
        exit 1
    fi

    # Execute based on mode
    if [ "$ALL" = true ] || [ "$SETUP" = true ] || [ "$VALIDATE" = true ] || [ "$STATUS" = true ]; then
        # Use launcher for complex operations
        run_launcher
    else
        # Direct execution mode
        if [ "$RUN" = true ]; then
            if [ "$ENHANCED" = true ]; then
                run_enhanced_predictor
            else
                run_launcher
            fi
        fi

        if [ "$ANALYZE" = true ] || [ "$PERFORMANCE" = true ]; then
            run_performance_analysis
        fi
    fi

    # Generate final report if requested
    if [ "$REPORT" = true ]; then
        print_status "📄 Generating final execution report..."
        local timestamp=$(date +%Y%m%d_%H%M%S)
        echo "# Pipeline Execution Report - $timestamp" > "PIPELINE_EXECUTION_REPORT_$timestamp.md"
        echo "Execution completed at: $(date)" >> "PIPELINE_EXECUTION_REPORT_$timestamp.md"
        print_status "✅ Report saved: PIPELINE_EXECUTION_REPORT_$timestamp.md"
    fi

    # Run cleanup to archive old files
    if [ -f "cleanup_old_outputs.sh" ]; then
        if [ "$QUIET" = false ]; then
            echo ""
            print_status "🧹 Running cleanup to archive files older than ${ARCHIVE_AGE_DAYS} days..."
        fi
        bash cleanup_old_outputs.sh "$ARCHIVE_AGE_DAYS" 2>/dev/null || true
    fi

    if [ "$QUIET" = false ]; then
        echo ""
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${BRIGHT_GREEN}${BOLD}"
        echo "    ✓✓✓ PIPELINE EXECUTION COMPLETED SUCCESSFULLY ✓✓✓"
        echo -e "${NC}"
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
    fi
}

# Handle interrupts gracefully
trap 'echo ""; print_warning "Operation cancelled by user"; exit 130' INT TERM

# Execute main function
main "$@"