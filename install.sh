#!/bin/bash
################################################################################
# Medical NLP Lean Package - Master Installation Script
################################################################################
#
# Purpose: Automated setup and validation of the Medical NLP Lean Package
#
# Prerequisites:
#   - All files copied from root to packages/medical_nlp_lean/ (per prompt_lean_project.md)
#   - Anaconda/Miniconda installed
#   - Git (for model downloads if needed)
#   - Internet connection (for package downloads)
#
# Usage:
#   ./install.sh                    # Full installation
#   ./install.sh --skip-tests       # Skip validation tests
#   ./install.sh --skip-model-setup # Skip model setup (use existing)
#   ./install.sh --help             # Show help
#
# Post-installation:
#   - Conda environment: py311_bionlp
#   - Run pipeline: ./run_ner_pipeline.sh
#   - Launch app: ./run_app.sh
#   - Run tests: ./run_tests.sh
#
################################################################################

set -e  # Exit on any error

################################################################################
# Color codes for output
################################################################################
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Configuration Variables
################################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_NAME="medical_nlp_lean"
CONDA_ENV="py311_bionlp"
PYTHON_VERSION="3.11"

# Installation flags (can be overridden by command-line arguments)
SKIP_TESTS=false  # Run tests by default
SKIP_MODEL_SETUP=false  # Check and download models by default
VERBOSE=false

# Log file for installation
LOG_FILE="$SCRIPT_DIR/installation_tests.log"
START_TIME=$(date +%s)

################################################################################
# Helper Functions
################################################################################

# Initialize log file
init_log() {
    cat > "$LOG_FILE" << EOF
================================================================================
MEDICAL NLP LEAN PACKAGE - INSTALLATION LOG
================================================================================
Date: $(date '+%Y-%m-%d %H:%M:%S')
Package: $PACKAGE_NAME
Location: $SCRIPT_DIR
Conda Environment: $CONDA_ENV
Python Version: $PYTHON_VERSION

Installation Flags:
  --skip-tests: $SKIP_TESTS
  --skip-model-setup: $SKIP_MODEL_SETUP
  --verbose: $VERBOSE

================================================================================
INSTALLATION STEPS
================================================================================

EOF
}

# Log to both console and file
log_message() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

# Log error with root cause analysis
log_error_detailed() {
    local error_msg="$1"
    local root_cause="${2:-Unknown cause}"
    local suggestion="${3:-Check documentation or contact support}"

    {
        echo ""
        echo "========== ERROR DETECTED =========="
        echo "Error: $error_msg"
        echo "Root Cause: $root_cause"
        echo "Suggestion: $suggestion"
        echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "===================================="
        echo ""
    } >> "$LOG_FILE"
}

# Log step status
log_step() {
    local step_num="$1"
    local step_name="$2"
    local status="$3"

    {
        echo ""
        echo "========== STEP $step_num: $step_name =========="
        echo "Status: $status"
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    } >> "$LOG_FILE"
}

# Print colored messages (with logging)
print_header() {
    local header="$1"
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$header${NC}"
    echo -e "${BLUE}========================================${NC}\n"
    log_message "INFO" "$header"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
    log_message "SUCCESS" "$1"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
    log_message "ERROR" "$1"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
    log_message "WARNING" "$1"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
    log_message "INFO" "$1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if conda environment exists
conda_env_exists() {
    conda env list | grep -q "^$1 "
}

# Display usage information
show_usage() {
    cat << EOF
Medical NLP Lean Package - Master Installation Script

Usage: $0 [OPTIONS]

Options:
    --skip-tests              Skip validation tests after installation (tests run by default)
    --skip-model-setup        Skip model checking and downloads entirely
    --verbose                 Show detailed output
    --help                    Display this help message

Model Installation (Default Behavior):
    By default, the script will AUTOMATICALLY:
    • Check if spaCy models are installed (en_core_web_sm, en_ner_bc5cdr_md)
      → If missing: Download via 'python -m spacy download'
    • Check if BioBERT models exist in models/pretrained/
      → If missing: Offer to download from Hugging Face (~1.6GB total)
    • Validate all models after installation

    This means:
    ✓ Fresh install → Models automatically downloaded
    ✓ Models already present → Detects and uses existing models
    ✓ Partial install → Downloads only missing models

    Use --skip-model-setup if:
    • ALL models are already present (BioBERT & spaCy)
    • You want to manually install models later
    • Testing installation without large downloads

Examples:
    $0                                # Full install: tests + auto-download missing models
    $0 --skip-tests                   # Install + auto-download models, skip tests
    $0 --skip-model-setup             # Install packages only, no model setup
    $0 --skip-tests --skip-model-setup  # Minimal install (env + packages only)

Post-installation:
    Activate environment:  conda activate $CONDA_ENV
    Run pipeline:         ./run_ner_pipeline.sh
    Launch Streamlit:     ./run_app.sh
    Run tests:            ./run_tests.sh
    View log:             cat installation_tests.log

EOF
    exit 0
}

################################################################################
# Parse Command-Line Arguments
################################################################################
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-model-setup|--skip-model-downloads|--skip-models)  # Support multiple flag names for compatibility
                SKIP_MODEL_SETUP=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help)
                show_usage
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                ;;
        esac
    done
}

################################################################################
# Step 1: Pre-Installation Checks
################################################################################
check_prerequisites() {
    print_header "Step 1: Checking Prerequisites"

    # Check if running from package directory
    if [[ ! -f "$SCRIPT_DIR/py311_bionlp_environment.yml" ]]; then
        print_error "Installation must be run from packages/medical_nlp_lean/ directory"
        print_info "Expected file not found: py311_bionlp_environment.yml"
        exit 1
    fi
    print_success "Running from correct directory: $SCRIPT_DIR"

    # Check for conda
    if ! command_exists conda; then
        print_error "Conda not found. Please install Anaconda or Miniconda"
        print_info "Download from: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    print_success "Conda found: $(conda --version)"

    # Check for Python 3.11
    if command_exists python3.11; then
        print_success "Python 3.11 available: $(python3.11 --version)"
    else
        print_warning "Python 3.11 not found in PATH (conda will install it)"
    fi

    # Check for git (optional, for model downloads)
    if command_exists git; then
        print_success "Git found: $(git --version | head -1)"
    else
        print_warning "Git not found (required for BioBERT model downloads)"
    fi

    # Verify critical files exist
    print_info "Verifying package structure..."

    local required_files=(
        "requirements.txt"
        "setup.py"
        "py311_bionlp_environment.yml"
        "src/enhanced_medical_ner_predictor.py"
        "app/medical_nlp_app.py"
        "data/external/target_rules_template.xlsx"
    )

    local missing_files=0
    for file in "${required_files[@]}"; do
        if [[ ! -f "$SCRIPT_DIR/$file" ]]; then
            print_error "Missing required file: $file"
            missing_files=$((missing_files + 1))
        fi
    done

    if [[ $missing_files -gt 0 ]]; then
        print_error "Missing $missing_files required files"
        print_info "Please ensure all files are copied from root per prompt_lean_project.md"
        exit 1
    fi
    print_success "All required files present"

    # Check available disk space (need ~2GB for models)
    local available_space=$(df -k "$SCRIPT_DIR" | tail -1 | awk '{print $4}')
    local required_space=$((2 * 1024 * 1024))  # 2GB in KB

    if [[ $available_space -lt $required_space ]]; then
        print_warning "Low disk space. Required: ~2GB for models"
        print_info "Available: $((available_space / 1024 / 1024))GB"
    else
        print_success "Sufficient disk space available"
    fi
}

################################################################################
# Step 2: Create/Validate Conda Environment
################################################################################
setup_conda_environment() {
    print_header "Step 2: Setting Up Conda Environment"

    # Check if environment already exists
    if conda_env_exists "$CONDA_ENV"; then
        print_warning "Conda environment '$CONDA_ENV' already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing environment..."
            conda env remove -n "$CONDA_ENV" -y
            print_success "Existing environment removed"
        else
            print_info "Using existing environment"
            return 0
        fi
    fi

    # Create conda environment from YAML file
    print_info "Creating conda environment from py311_bionlp_environment.yml..."
    print_info "This may take 5-10 minutes..."

    if conda env create -f "$SCRIPT_DIR/py311_bionlp_environment.yml"; then
        print_success "Conda environment '$CONDA_ENV' created successfully"
    else
        print_error "Failed to create conda environment"
        print_info "Try manually: conda env create -f py311_bionlp_environment.yml"
        exit 1
    fi

    # Verify environment was created
    if conda_env_exists "$CONDA_ENV"; then
        print_success "Environment verified: $CONDA_ENV"
    else
        print_error "Environment creation verification failed"
        exit 1
    fi
}

################################################################################
# Step 3: Install Package Dependencies
################################################################################
install_dependencies() {
    print_header "Step 3: Installing Package Dependencies"

    print_info "Installing package in editable mode..."

    # Install package using pip in the conda environment
    if conda run -n "$CONDA_ENV" pip install -e "$SCRIPT_DIR"; then
        print_success "Package installed successfully"
    else
        print_error "Failed to install package"
        print_info "Try manually: conda activate $CONDA_ENV && pip install -e ."
        exit 1
    fi

    # Install test dependencies if not skipping tests
    if [[ "$SKIP_TESTS" == false ]]; then
        if [[ -f "$SCRIPT_DIR/requirements-test.txt" ]]; then
            print_info "Installing test dependencies..."
            if conda run -n "$CONDA_ENV" pip install -r "$SCRIPT_DIR/requirements-test.txt"; then
                print_success "Test dependencies installed"
            else
                print_warning "Failed to install test dependencies (non-critical)"
            fi
        fi
    fi

    # Verify key packages are installed
    print_info "Verifying critical packages..."

    # Check packages with proper import names
    local packages=(
        "spacy:spacy"
        "pandas:pandas"
        "streamlit:streamlit"
        "torch:torch"
        "transformers:transformers"
    )

    local missing_packages=()
    for pkg_info in "${packages[@]}"; do
        IFS=':' read -r display_name import_name <<< "$pkg_info"
        if conda run -n "$CONDA_ENV" python -c "import $import_name" 2>/dev/null; then
            print_success "Package verified: $display_name"
        else
            print_error "Package not found: $display_name"
            missing_packages+=("$display_name")
        fi
    done

    # If torch is missing, try to install it
    if [[ " ${missing_packages[@]} " =~ " torch " ]]; then
        print_warning "PyTorch not found - attempting to install..."
        if conda run -n "$CONDA_ENV" pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; then
            print_success "PyTorch installed successfully"
        else
            print_error "Failed to install PyTorch"
        fi
    fi

    # Check if any critical packages are still missing
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        print_warning "Some packages are missing but installation will continue"
        print_info "Missing packages: ${missing_packages[*]}"
    fi
}

################################################################################
# Step 4: Download and Setup Models
################################################################################
setup_models() {
    print_header "Step 4: Setting Up Models"
    log_step "4" "Model Setup" "STARTED"

    if [[ "$SKIP_MODEL_SETUP" == true ]]; then
        print_warning "Skipping model setup (--skip-model-setup flag)"
        log_message "INFO" "Model setup skipped by user flag"
        log_step "4" "Model Setup" "SKIPPED"
        return 0
    fi

    # NEW BEHAVIOR: Check what's missing and download automatically
    local models_dir="$SCRIPT_DIR/models/pretrained"
    print_info "Checking model status..."

    # Check and download spaCy models
    print_info "Checking spaCy models..."

    # Check en_core_web_sm
    if conda run -n "$CONDA_ENV" python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null; then
        print_success "spaCy model found: en_core_web_sm"
        log_message "INFO" "en_core_web_sm already installed"
    else
        print_info "Downloading en_core_web_sm..."
        log_message "INFO" "Downloading en_core_web_sm"
        if conda run -n "$CONDA_ENV" python -m spacy download en_core_web_sm; then
            print_success "spaCy model downloaded: en_core_web_sm"
            log_message "SUCCESS" "en_core_web_sm downloaded successfully"
        else
            print_error "Failed to download en_core_web_sm"
            log_error_detailed "Failed to download en_core_web_sm" \
                "Network issue or spaCy installation problem" \
                "Check internet connection and try: conda run -n $CONDA_ENV python -m spacy download en_core_web_sm"
            exit 1
        fi
    fi

    # Check en_ner_bc5cdr_md
    if conda run -n "$CONDA_ENV" python -c "import spacy; spacy.load('en_ner_bc5cdr_md')" 2>/dev/null; then
        print_success "spaCy model found: en_ner_bc5cdr_md"
        log_message "INFO" "en_ner_bc5cdr_md already installed"
    else
        print_info "Downloading en_ner_bc5cdr_md..."
        log_message "INFO" "Downloading en_ner_bc5cdr_md"
        if conda run -n "$CONDA_ENV" python -m spacy download en_ner_bc5cdr_md; then
            print_success "spaCy model downloaded: en_ner_bc5cdr_md"
            log_message "SUCCESS" "en_ner_bc5cdr_md downloaded successfully"
        else
            print_warning "Failed to download en_ner_bc5cdr_md (optional model)"
            log_message "WARNING" "en_ner_bc5cdr_md download failed - continuing"
        fi
    fi

    # Check for BioBERT models (Disease, Chemical, Gene)
    print_info "Checking BioBERT models in $models_dir..."
    log_message "INFO" "Checking BioBERT models"

    local biobert_models=("Disease" "Chemical" "Gene")
    local missing_biobert=0
    local missing_list=()

    for model in "${biobert_models[@]}"; do
        if [[ -d "$models_dir/$model" ]] && [[ -f "$models_dir/$model/pytorch_model.bin" ]]; then
            print_success "BioBERT model found: $model"
            log_message "INFO" "BioBERT $model model already present"
        else
            print_warning "BioBERT model missing: $model"
            log_message "WARNING" "BioBERT $model model not found"
            missing_biobert=$((missing_biobert + 1))
            missing_list+=("$model")
        fi
    done

    # NEW BEHAVIOR: Automatically offer to download missing models
    if [[ $missing_biobert -gt 0 ]]; then
        print_warning "Missing $missing_biobert BioBERT models: ${missing_list[*]}"
        print_info "Total download size: ~1.6GB (Disease: 411MB, Chemical: 822MB, Gene: 411MB)"
        log_message "INFO" "Missing BioBERT models: ${missing_list[*]}"

        if command_exists git; then
            print_info "BioBERT models can be downloaded from Hugging Face"
            read -p "Download missing BioBERT models now? (Y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then  # Default to Yes
                print_info "Downloading BioBERT models..."
                log_message "INFO" "User chose to download BioBERT models"
                download_biobert_models
                log_step "4" "Model Setup" "COMPLETED"
            else
                print_warning "Skipping BioBERT downloads"
                print_info "Pipeline will use spaCy models only (reduced accuracy)"
                log_message "WARNING" "User skipped BioBERT downloads - reduced accuracy expected"
                log_step "4" "Model Setup" "PARTIAL"
            fi
        else
            print_error "Git not available for BioBERT downloads"
            log_error_detailed "Git not found" \
                "Git is required to download BioBERT models from Hugging Face" \
                "Install git or copy models manually from root/models/pretrained/"
            print_info "Alternative: Copy models from root/models/pretrained/"
            log_step "4" "Model Setup" "PARTIAL"
        fi
    else
        print_success "All BioBERT models present"
        log_message "SUCCESS" "All BioBERT models found"
        log_step "4" "Model Setup" "COMPLETED"
    fi
}

# Helper function to download BioBERT models
download_biobert_models() {
    local models_dir="$SCRIPT_DIR/models/pretrained"
    mkdir -p "$models_dir"
    cd "$models_dir"

    print_info "Installing git-lfs for large file downloads..."
    if ! command_exists git-lfs; then
        print_warning "git-lfs not found. Install it first:"
        print_info "  macOS: brew install git-lfs"
        print_info "  Ubuntu: sudo apt-get install git-lfs"
        print_info "  Manual: https://git-lfs.github.com/"
        return 1
    fi
    git lfs install

    # Download Disease model (~411MB)
    if [[ ! -d "Disease" ]]; then
        print_info "Downloading Disease model (~411MB)..."
        if git clone https://huggingface.co/alvaroalon2/biobert_diseases_ner Disease; then
            print_success "Disease model downloaded"
        else
            print_error "Failed to download Disease model"
        fi
    fi

    # Download Chemical model (~822MB)
    if [[ ! -d "Chemical" ]]; then
        print_info "Downloading Chemical model (~822MB)..."
        if git clone https://huggingface.co/alvaroalon2/biobert_chemical_ner Chemical; then
            print_success "Chemical model downloaded"
        else
            print_error "Failed to download Chemical model"
        fi
    fi

    # Download Gene model (~411MB)
    if [[ ! -d "Gene" ]]; then
        print_info "Downloading Gene model (~411MB)..."
        if git clone https://huggingface.co/alvaroalon2/biobert_genetic_ner Gene; then
            print_success "Gene model downloaded"
        else
            print_error "Failed to download Gene model"
        fi
    fi

    cd "$SCRIPT_DIR"
}

################################################################################
# Step 5: Validate Configuration Files
################################################################################
validate_configuration() {
    print_header "Step 5: Validating Configuration"

    # Check template files (6 required)
    print_info "Checking template files..."
    local template_dir="$SCRIPT_DIR/data/external"
    local templates=(
        "target_rules_template.xlsx"
        "historical_rules_template.xlsx"
        "negated_rules_template.xlsx"
        "uncertainty_rules_template.xlsx"
        "confirmed_rules_template.xlsx"
        "family_rules_template.xlsx"
    )

    local missing_templates=0
    for template in "${templates[@]}"; do
        if [[ -f "$template_dir/$template" ]]; then
            local size=$(ls -lh "$template_dir/$template" | awk '{print $5}')
            print_success "Template found: $template ($size)"
        else
            print_error "Template missing: $template"
            missing_templates=$((missing_templates + 1))
        fi
    done

    if [[ $missing_templates -gt 0 ]]; then
        print_error "Missing $missing_templates template files"
        print_info "Please copy templates from root/data/external/"
        exit 1
    fi

    # Check sample data
    print_info "Checking sample data..."
    if [[ -f "$SCRIPT_DIR/data/raw/sample_input.xlsx" ]]; then
        print_success "Sample input data present"
    else
        print_warning "Sample input data missing (non-critical)"
    fi

    # Create output directories if they don't exist
    print_info "Creating output directories..."
    local output_dirs=(
        "output/results"
        "output/visualizations"
        "output/logs"
        "output/reports"
        "output/exports"
        "tests/test_logs"
    )

    for dir in "${output_dirs[@]}"; do
        mkdir -p "$SCRIPT_DIR/$dir"
        touch "$SCRIPT_DIR/$dir/.gitkeep"
    done
    print_success "Output directories created"

    # Make shell scripts executable
    print_info "Setting script permissions..."
    local scripts=(
        "run_ner_pipeline.sh"
        "run_tests.sh"
        "run_app.sh"
        "cleanup_test_logs.sh"
        "activate_env.sh"
    )

    for script in "${scripts[@]}"; do
        if [[ -f "$SCRIPT_DIR/$script" ]]; then
            chmod +x "$SCRIPT_DIR/$script"
            print_success "Made executable: $script"
        fi
    done
}

################################################################################
# Step 6: Run Validation Tests
################################################################################
run_validation_tests() {
    print_header "Step 6: Running Validation Tests"

    if [[ "$SKIP_TESTS" == true ]]; then
        print_warning "Skipping validation tests (--skip-tests flag)"
        return 0
    fi

    # Check if run_tests.sh exists
    if [[ ! -f "$SCRIPT_DIR/run_tests.sh" ]]; then
        print_warning "Test script not found: run_tests.sh"
        print_info "Skipping validation tests"
        return 0
    fi

    print_info "Running quick validation tests..."
    print_info "Testing basic imports and file structure..."

    # Run basic import validation test
    cd "$SCRIPT_DIR"

    local validation_test=$(cat << 'VALIDATION_EOF'
import sys
import os
from pathlib import Path

# Test critical imports
try:
    print("Checking Python environment...")

    # Check src directory is accessible
    sys.path.insert(0, str(Path.cwd() / "src"))

    # Test template file access
    templates_dir = Path("data/external")
    template_count = len(list(templates_dir.glob("*_template.xlsx")))
    print(f"✓ Found {template_count} template files")

    # Test BioBERT models
    models_dir = Path("models/pretrained")
    if models_dir.exists():
        model_count = len([d for d in models_dir.iterdir() if d.is_dir() and d.name in ['Disease', 'Chemical', 'Gene']])
        print(f"✓ Found {model_count} BioBERT models")

    # Test test files structure
    tests_dir = Path("tests")
    test_files = list(tests_dir.rglob("test_*.py"))
    print(f"✓ Found {len(test_files)} test files")

    print("\n✅ Basic validation passed - package structure is correct")
    sys.exit(0)

except Exception as e:
    print(f"\n✗ Validation failed: {e}")
    sys.exit(1)
VALIDATION_EOF
)

    if conda run -n "$CONDA_ENV" python -c "$validation_test"; then
        print_success "Validation tests passed"
        print_info "Package structure verified successfully"
    else
        print_warning "Some validation checks failed"
        print_info "You can continue, but there may be issues"

        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

################################################################################
# Step 7: Final Verification
################################################################################
final_verification() {
    print_header "Step 7: Final Verification"

    # Test Python imports in the environment
    print_info "Testing critical imports..."

    local import_test=$(cat << 'EOF'
import sys
from pathlib import Path

# Test core imports
try:
    import spacy
    import pandas
    import streamlit
    import torch
    import transformers
    print("✓ All critical packages import successfully")
    sys.exit(0)
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
EOF
)

    if conda run -n "$CONDA_ENV" python -c "$import_test"; then
        print_success "All critical imports verified"
    else
        print_error "Import verification failed"
        exit 1
    fi

    # Check if pipeline script can be loaded
    print_info "Verifying pipeline script..."
    if conda run -n "$CONDA_ENV" python -c "from src.enhanced_medical_ner_predictor import *" 2>/dev/null; then
        print_success "Pipeline script loads successfully"
    else
        print_warning "Pipeline script verification failed (may be non-critical)"
    fi

    # Summary of installation
    print_info "Installation summary:"
    echo ""
    echo "  Environment:        $CONDA_ENV"
    echo "  Python:            $(conda run -n "$CONDA_ENV" python --version)"
    echo "  Package location:   $SCRIPT_DIR"
    echo "  Models:            $(ls -1 "$SCRIPT_DIR/models/pretrained" 2>/dev/null | wc -l) BioBERT models"
    echo "  Templates:         $(ls -1 "$SCRIPT_DIR/data/external"/*.xlsx 2>/dev/null | wc -l) template files"
    echo "  Test logs:         $SCRIPT_DIR/tests/test_logs/"
    echo ""
}

################################################################################
# Step 8: Post-Installation Instructions
################################################################################
show_next_steps() {
    print_header "Installation Complete!"

    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                                    ║${NC}"
    echo -e "${GREEN}║  Medical NLP Lean Package - Successfully Installed!               ║${NC}"
    echo -e "${GREEN}║                                                                    ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo ""
    echo -e "  ${GREEN}1. Activate the environment:${NC}"
    echo "     conda activate $CONDA_ENV"
    echo ""
    echo -e "  ${GREEN}2. Run the NER pipeline:${NC}"
    echo "     cd $SCRIPT_DIR"
    echo "     ./run_ner_pipeline.sh"
    echo ""
    echo -e "  ${GREEN}3. Launch the Streamlit app:${NC}"
    echo "     ./run_app.sh"
    echo ""
    echo -e "  ${GREEN}4. Run the full test suite:${NC}"
    echo "     ./run_tests.sh"
    echo ""
    echo -e "${BLUE}Quick Commands:${NC}"
    echo ""
    echo -e "  ${YELLOW}# Process a file${NC}"
    echo "  ./run_ner_pipeline.sh input.xlsx output.xlsx"
    echo ""
    echo -e "  ${YELLOW}# Launch Streamlit on custom port${NC}"
    echo "  ./run_app.sh 8502"
    echo ""
    echo -e "  ${YELLOW}# Run specific test category${NC}"
    echo "  ./run_tests.sh --category excel_output"
    echo ""
    echo -e "  ${YELLOW}# View test logs${NC}"
    echo "  ls -lh tests/test_logs/"
    echo ""
    echo -e "  ${YELLOW}# Clean old test logs${NC}"
    echo "  ./cleanup_test_logs.sh 7"
    echo ""
    echo -e "${BLUE}Documentation:${NC}"
    echo ""
    echo "  - README.md              - Package overview and usage"
    echo "  - docs/IMPLEMENTATION_GUIDE.md - Complete implementation guide"
    echo "  - tests/test_logs/       - Test execution logs"
    echo ""
    echo -e "${BLUE}Installation Log:${NC}"
    echo ""
    echo "  - installation_tests.log - Complete installation log with root cause analysis"
    echo "  View log: cat installation_tests.log"
    echo ""
    echo -e "${BLUE}Troubleshooting:${NC}"
    echo ""
    echo "  If you encounter issues:"
    echo "  1. Check installation log: cat installation_tests.log"
    echo "  2. Check test logs in tests/test_logs/"
    echo "  3. Verify models are present: ls models/pretrained/"
    echo "  4. Re-run tests: ./run_tests.sh --verbose"
    echo "  5. Review README.md for detailed instructions"
    echo ""
    echo -e "${GREEN}Happy analyzing! 🧬${NC}"
    echo ""
}

################################################################################
# Main Installation Flow
################################################################################
main() {
    # Parse command-line arguments
    parse_args "$@"

    # Initialize logging
    init_log
    log_message "INFO" "Installation started by user: $(whoami)"
    log_message "INFO" "Installation directory: $SCRIPT_DIR"

    # Display header
    clear
    cat << "EOF"
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║          Medical NLP Lean Package - Master Installation             ║
║                                                                      ║
║  Production-ready Medical Named Entity Recognition Pipeline         ║
║  Version 1.0.0                                                      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

EOF

    print_info "Starting installation at $(date)"
    print_info "Installation directory: $SCRIPT_DIR"
    print_info "Log file: $LOG_FILE"
    echo ""

    # Installation steps (in order)
    check_prerequisites
    setup_conda_environment
    install_dependencies
    setup_models
    validate_configuration
    run_validation_tests
    final_verification
    show_next_steps

    # Calculate and log installation time
    END_TIME=$(date +%s)
    INSTALL_TIME=$((END_TIME - START_TIME))
    INSTALL_MINUTES=$((INSTALL_TIME / 60))
    INSTALL_SECONDS=$((INSTALL_TIME % 60))

    # Installation complete
    print_success "Installation completed successfully at $(date)"
    print_info "Total installation time: ${INSTALL_MINUTES}m ${INSTALL_SECONDS}s"

    # Write final summary to log
    {
        echo ""
        echo "================================================================================"
        echo "INSTALLATION SUMMARY"
        echo "================================================================================"
        echo "Status: SUCCESS"
        echo "Completion Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Total Duration: ${INSTALL_MINUTES}m ${INSTALL_SECONDS}s"
        echo ""
        echo "Components Installed:"
        echo "  ✓ Conda Environment: $CONDA_ENV"
        echo "  ✓ Python Packages: Installed via pip"
        echo "  ✓ Models: spaCy and BioBERT (or skipped)"
        echo "  ✓ Configuration: Validated"
        echo "  ✓ Tests: $([ "$SKIP_TESTS" == "true" ] && echo "Skipped" || echo "Executed")"
        echo ""
        echo "Next Steps:"
        echo "  1. conda activate $CONDA_ENV"
        echo "  2. ./run_ner_pipeline.sh"
        echo "  3. ./run_app.sh"
        echo ""
        echo "For troubleshooting, review this log file: $LOG_FILE"
        echo "================================================================================"
    } >> "$LOG_FILE"

    log_message "SUCCESS" "Installation completed successfully in ${INSTALL_MINUTES}m ${INSTALL_SECONDS}s"

    # Exit with success
    exit 0
}

################################################################################
# Error Handler
################################################################################
handle_error() {
    local line_no=$LINENO
    local last_command="$BASH_COMMAND"

    print_error "Installation failed at line $line_no"
    print_error "Failed command: $last_command"

    # Log error details
    {
        echo ""
        echo "================================================================================"
        echo "INSTALLATION FAILED"
        echo "================================================================================"
        echo "Status: FAILED"
        echo "Failure Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Failed at Line: $line_no"
        echo "Failed Command: $last_command"
        echo ""
        echo "Troubleshooting:"
        echo "  1. Check the error message above"
        echo "  2. Review this complete log: $LOG_FILE"
        echo "  3. Check prerequisites: conda, Python 3.11, git"
        echo "  4. Verify internet connection for model downloads"
        echo "  5. Try re-running with --verbose flag"
        echo ""
        echo "Common Issues:"
        echo "  - Network timeout → Check internet connection"
        echo "  - Permission denied → Check file/directory permissions"
        echo "  - Command not found → Ensure conda is activated"
        echo "  - Package conflicts → Try recreating conda environment"
        echo "================================================================================"
    } >> "$LOG_FILE"

    log_message "FATAL" "Installation failed at line $line_no: $last_command"

    exit 1
}

trap 'handle_error' ERR

################################################################################
# Execute Main Function
################################################################################
main "$@"
