# Medical NLP Lean Package - Complete Implementation Guide

**Status**: Ready for implementation
**Based on**: CLINDOCS_NLP Medical NLP Pipeline v2.2.0
**Target**: Production-ready lean package with Streamlit + CLI
**Version**: 2.0 (Comprehensive Merged Guide)
**Last Updated**: 2025-10-08

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [File Copying Requirements](#file-copying-requirements)
4. [Models Directory Setup](#models-directory-setup)
5. [Comprehensive Testing Suite](#comprehensive-testing-suite)
6. [Script Updates](#script-updates)
7. [Configuration Files](#configuration-files)
8. [Installation & Usage](#installation--usage)
9. [Validation Checklist](#validation-checklist)
10. [Success Criteria](#success-criteria)

---

## 1. Overview

### 1.1 Objective

Create a lean, production-ready version of the CLINDOCS_NLP project as a standalone package named `medical_nlp_lean` in the `packages/` directory. This package contains only essential components needed to run the Medical NER pipeline via CLI and Streamlit interface.

### 1.2 What Gets Created

- **Core Components**: Enhanced predictor, performance analyzer, scope reversal engine
- **Streamlit App**: Interactive web interface for entity recognition
- **Complete Testing**: 7 test categories, 40+ test files covering all features
- **Models Directory**: Pretrained models from `root/models/`
- **Full Documentation**: Setup, usage, and validation guides

### 1.3 Key Features to Include

✅ 57,476 target medical terms (entity detection)
✅ 103 scope reversal patterns with confidence scoring
✅ 99 negation patterns
✅ 82 historical patterns
✅ 79 family history patterns
✅ 48 uncertainty patterns
✅ 138 confirmed patterns
✅ 15-column Excel output with formatting
✅ DisplaCy visualization
✅ CLI and Streamlit 100% output consistency

---

## 2. Project Structure

Create this complete structure in `packages/medical_nlp_lean/`:

```
packages/medical_nlp_lean/
├── README.md                              # Package documentation
├── requirements.txt                       # Python dependencies
├── requirements-test.txt                  # Test dependencies
├── setup.py                              # Package installation script
├── pyproject.toml                         # Modern Python config
├── py311_bionlp_environment.yml          # Conda environment specification
├── activate_env.sh                       # Environment activation script
├── run_ner_pipeline.sh                   # CLI execution script
├── run_tests.sh                          # Master test runner script
├── run_app.sh                            # Streamlit app launcher
├── Makefile                              # Automation commands
├── .gitignore                            # Git ignore rules
│
├── src/                                  # Core Python modules
│   ├── __init__.py
│   ├── enhanced_medical_ner_predictor.py
│   ├── enhanced_performance_analyzer.py
│   ├── enhanced_target_rules_loader.py
│   ├── enhanced_template_processor.py      # If exists
│   └── scope_reversal_engine.py
│
├── app/                                  # Streamlit application
│   ├── __init__.py
│   └── medical_nlp_app.py                 # Renamed from bio_ner_streamlit_app.py
│
├── data/                                 # Data files
│   ├── external/                         # Template files (6 files)
│   │   ├── target_rules_template.xlsx
│   │   ├── historical_rules_template.xlsx
│   │   ├── negated_rules_template.xlsx
│   │   ├── uncertainty_rules_template.xlsx
│   │   ├── confirmed_rules_template.xlsx
│   │   └── family_rules_template.xlsx
│   │
│   └── raw/                              # Sample input data
│       └── sample_input.xlsx              # From root/data/raw/input_100texts.xlsx
│
├── models/                               # Copy from root/models (~1-2GB)
│   ├── pretrained/                       # Downloaded spaCy/BioBERT models
│   │   └── [model files]
│   └── trained/                          # Empty initially
│       └── .gitkeep
│
├── output/                               # Generated outputs
│   ├── results/
│   │   └── .gitkeep
│   ├── visualizations/
│   │   └── .gitkeep
│   ├── logs/
│   │   └── .gitkeep
│   ├── reports/
│   │   └── .gitkeep
│   └── exports/                          # Streamlit exports
│       └── .gitkeep
│
├── tests/                                # Comprehensive test suite (50 test files)
│   ├── __init__.py
│   ├── master_test_script.py             # Copy from run_all_tests.py
│   │
│   ├── unit/                             # Unit tests (4 files)
│   │   ├── __init__.py
│   │   ├── test_predictor.py             # From tests/test_enhanced_medical_ner_predictor.py
│   │   ├── test_performance_analyzer.py  # From tests/test_enhanced_performance_analyzer.py
│   │   ├── test_rules_loader.py          # From tests/test_enhanced_target_rules_loader.py
│   │   └── test_scope_reversal.py        # New unit test for scope reversal engine
│   │
│   ├── integration/                      # Integration tests (3 files)
│   │   ├── __init__.py
│   │   ├── test_pipeline_integration.py  # New - end-to-end pipeline test
│   │   ├── test_template_patterns.py     # From test_template_patterns.py
│   │   └── test_consistency.py           # From test_consistency.py
│   │
│   ├── features/                         # Feature-specific tests (16 files)
│   │   ├── __init__.py
│   │   ├── test_entity_detection.py      # From test_template_patterns.py - Entity rules
│   │   ├── test_scope_reversal.py        # From test_comprehensive_scope_reversal.py - 103 patterns
│   │   ├── test_scope_reversal_v2.py     # From test_comprehensive_scope_reversal_v2.py - Enhanced
│   │   ├── test_negation.py              # From test_negation_scope.py - 99 patterns
│   │   ├── test_negation_boundaries.py   # From test_negation_boundaries.py - Word boundaries
│   │   ├── test_negation_debug.py        # From test_negation_with_debug.py - Debug mode
│   │   ├── test_confirmed_patterns.py    # From test_confirmed_patterns.py - 138 patterns
│   │   ├── test_historical_detection.py  # From tests/test_historical_detection_improvements.py - 82 patterns
│   │   ├── test_historical.py            # From test_historical.py - Historical context
│   │   ├── test_section_detection.py     # From test_section_detection.py - Section headers
│   │   ├── test_context_sentences.py     # From test_context_fix.py - Context formatting
│   │   ├── test_context_classifications.py # From test_context_classifications.py - Classification logic
│   │   ├── test_context_overlap.py       # From test_context_overlap_suite.py - Overlap resolution
│   │   ├── test_entity_context.py        # From tests/test_entity_context_detection.py
│   │   ├── test_confidence_boundaries.py # From test_confidence_and_word_boundaries.py
│   │   └── test_family_detection.py      # New - 79 family history patterns (create based on template)
│   │
│   ├── excel_output/                     # Excel output validation (4 files)
│   │   ├── __init__.py
│   │   ├── test_excel_columns.py         # New - Validate 15 required columns
│   │   ├── test_excel_formatting.py      # New - Fonts, colors, highlighting
│   │   ├── test_export_formatting.py     # From test_export_formatting_v2.py
│   │   └── test_context_formatting.py    # New - Context sentence display in Excel
│   │
│   ├── app/                              # Streamlit app tests (10 files)
│   │   ├── __init__.py
│   │   ├── test_streamlit_buttons.py     # From test_streamlit_buttons.py - Button functions
│   │   ├── test_streamlit_combined.py    # From test_streamlit_combined.py - Combined features
│   │   ├── test_streamlit_context.py     # From test_streamlit_context.py - Context display
│   │   ├── test_streamlit_display.py     # From test_streamlit_display_logic.py - Display logic
│   │   ├── test_export_results.py        # New - Export Results button
│   │   ├── test_file_upload.py           # From test_file_upload_comparison.py
│   │   ├── test_file_upload_report.py    # From test_file_upload_comparison_with_report.py
│   │   ├── test_file_upload_consistency.py # From tests/test_file_upload_consistency.py
│   │   ├── test_entity_colors.py         # New - Entity color highlighting
│   │   └── test_pipeline_validation.py   # From test_pipeline_validation.py
│   │
│   ├── visualization/                    # Visualization tests (8 files)
│   │   ├── __init__.py
│   │   ├── test_color_highlighting.py    # From test_color_highlighting_complete.py
│   │   ├── test_colored_visualization.py # From test_colored_visualization.py
│   │   ├── test_combined_viz.py          # From test_combined_text_visualization.py
│   │   ├── test_full_viz.py              # From test_full_viz.py - Complete viz suite
│   │   ├── test_visualization_fix.py     # From test_visualization_fix.py
│   │   ├── test_visualization_simple.py  # From test_visualization_simple.py
│   │   ├── test_visualization_update.py  # From test_visualization_update.py
│   │   └── test_font_size.py             # New - Font size validation
│   │
│   ├── validation/                       # Validation tests (3 files)
│   │   ├── __init__.py
│   │   ├── test_path_validation.py       # From test_path_validation.py - Relative paths
│   │   ├── test_flexible_input.py        # From test_flexible_input.py - Multiple formats
│   │   └── test_pipeline_validation.py   # Merge with app/test_pipeline_validation.py
│   │
│   ├── debugging/                        # Debug/investigation tests (6 files) - OPTIONAL
│   │   ├── __init__.py
│   │   ├── test_negation_detailed.py     # From test_negation_detailed_debug.py - Step-by-step
│   │   ├── test_model_debug.py           # From test_model_debug.py - Model investigation
│   │   ├── test_kif5a_overlap.py         # From test_kif5a_overlap.py - KIF5A specific
│   │   ├── test_specific_overlap.py      # From test_specific_overlap.py - Overlap cases
│   │   ├── test_export_formatting_v1.py  # From test_export_formatting.py - Legacy
│   │   └── README.md                     # Note: These are optional debug tests
│   │
│   ├── test_data/                        # Test fixtures and sample data
│   │   ├── sample_data.xlsx              # From root/data/raw/input_100texts.xlsx (PRIMARY)
│   │   ├── confidence_comparison_test.xlsx     # Optional - Confidence threshold tests
│   │   ├── context_test_results_complete.xlsx  # Optional - Context formatting tests
│   │   ├── kif5a_test.xlsx               # Optional - KIF5A gene detection
│   │   ├── template_override_test.xlsx   # Optional - Template priority tests
│   │   ├── test_visualization_2samples.xlsx    # Optional - Visualization tests
│   │   ├── test_contexts.json            # New - Context test scenarios
│   │   └── expected_outputs/             # Reference outputs for validation
│   │       ├── expected_excel_output.xlsx
│   │       ├── expected_entities.json
│   │       └── expected_visualization.png
│   │
│   └── test_logs/                        # Test execution logs (auto-generated)
│       ├── .gitkeep
│       ├── {testscript_name}_{timestamp}.log    # Individual test failure logs
│       └── test_summary_{timestamp}.txt         # Master test execution summaries
│
├── configs/                              # Configuration files
│   └── pipeline_config.yaml
│
└── logs/                                 # Log files
    └── .gitkeep
```

---

## 3. File Copying Requirements

### 3.1 Core Python Scripts (Root → src/)

Copy these files from CLINDOCS_NLP root to `packages/medical_nlp_lean/src/`:

```bash
# Main pipeline components
cp enhanced_medical_ner_predictor.py packages/medical_nlp_lean/src/
cp enhanced_performance_analyzer.py packages/medical_nlp_lean/src/
cp enhanced_target_rules_loader.py packages/medical_nlp_lean/src/
cp scope_reversal_engine.py packages/medical_nlp_lean/src/

# If exists:
cp enhanced_template_processor.py packages/medical_nlp_lean/src/
```

### 3.2 Streamlit Application (app/ → app/)

```bash
# Copy and rename
cp app/bio_ner_streamlit_app.py packages/medical_nlp_lean/app/medical_nlp_app.py
```

### 3.3 Template Files (data/external/ → data/external/)

```bash
# All 6 template files
cp data/external/target_rules_template.xlsx packages/medical_nlp_lean/data/external/
cp data/external/historical_rules_template.xlsx packages/medical_nlp_lean/data/external/
cp data/external/negated_rules_template.xlsx packages/medical_nlp_lean/data/external/
cp data/external/uncertainty_rules_template.xlsx packages/medical_nlp_lean/data/external/
cp data/external/confirmed_rules_template.xlsx packages/medical_nlp_lean/data/external/
cp data/external/family_rules_template.xlsx packages/medical_nlp_lean/data/external/
```

**Template Statistics**:
- target_rules_template.xlsx: 57,476 medical terms
- historical_rules_template.xlsx: 82 patterns
- negated_rules_template.xlsx: 99 patterns
- uncertainty_rules_template.xlsx: 48 patterns
- confirmed_rules_template.xlsx: 138 patterns
- family_rules_template.xlsx: 79 patterns

### 3.4 Models Directory (models/ → models/)

**CRITICAL**: Copy entire models directory structure:

```bash
# Copy models directory from root
cp -r models/ packages/medical_nlp_lean/models/

# This includes:
# - models/pretrained/  (~1.6GB - BioBERT NER models)
# - models/trained/     (any trained models)
```

**Model Contents** (`models/pretrained/`):

The pretrained directory contains three BioBERT-based NER models from Hugging Face:

1. **Disease Model** (`models/pretrained/Disease/`) - ~411MB
   - Source: `alvaroalon2/biobert_diseases_ner`
   - Files: `config.json`, `pytorch_model.bin`, `vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json`
   - Purpose: Disease entity recognition

2. **Chemical Model** (`models/pretrained/Chemical/`) - ~822MB
   - Source: `alvaroalon2/biobert_chemical_ner`
   - Files: `config.json`, `pytorch_model.bin`, `tf_model.h5`, `vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json`
   - Purpose: Chemical entity recognition

3. **Gene Model** (`models/pretrained/Gene/`) - ~411MB
   - Source: `alvaroalon2/biobert_genetic_ner`
   - Files: `config.json`, `pytorch_model.bin`, `vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json`
   - Purpose: Gene/Protein entity recognition

4. **Metadata Files**:
   - `README.md` - Model download instructions
   - `manifest.json` - Model registry

**Total Size**: ~1.6GB

**Important Notes**:
- Models are large (~1.6GB total for BioBERT models)
- SpaCy models (`en_core_web_sm`, `en_ner_bc5cdr_md`) are downloaded separately via pip
- For deployment: Include models in package (full bundle)
- For development: Document model download steps separately (see Section 4.2)

### 3.5 Environment Configuration

```bash
# Copy environment files
cp py311_bionlp_environment.yml packages/medical_nlp_lean/
cp activate_py311_bionlp.sh packages/medical_nlp_lean/activate_env.sh
```

### 3.6 Sample Data (data/raw/ → data/raw/ and tests/test_data/)

```bash
# Primary sample input (main testing file)
cp data/raw/input_100texts.xlsx packages/medical_nlp_lean/data/raw/sample_input.xlsx

# Also copy to test_data for testing
cp data/raw/input_100texts.xlsx packages/medical_nlp_lean/tests/test_data/sample_data.xlsx

# Optional: Copy additional test files for specific test scenarios
cp data/raw/confidence_comparison_test.xlsx packages/medical_nlp_lean/tests/test_data/
cp data/raw/context_test_results_complete.xlsx packages/medical_nlp_lean/tests/test_data/
cp data/raw/kif5a_test.xlsx packages/medical_nlp_lean/tests/test_data/
cp data/raw/template_override_test.xlsx packages/medical_nlp_lean/tests/test_data/
cp data/raw/test_visualization_2samples.xlsx packages/medical_nlp_lean/tests/test_data/
```

**Data Files in `data/raw/`**:

1. **input_100texts.xlsx** (37KB) - **PRIMARY FILE**
   - Main production test dataset
   - 100 clinical text samples
   - Used for: `sample_input.xlsx` and `sample_data.xlsx`

2. **confidence_comparison_test.xlsx** (5.9KB) - OPTIONAL
   - Tests confidence threshold differences (0.3 vs 0.5)
   - Validates template-priority scoring

3. **context_test_results_complete.xlsx** (12KB) - OPTIONAL
   - Tests context sentence formatting
   - Validates entity highlighting in context

4. **kif5a_test.xlsx** (5.9KB) - OPTIONAL
   - Tests specific gene entity detection (KIF5A)
   - Validates gene/protein recognition

5. **template_override_test.xlsx** (6.0KB) - OPTIONAL
   - Tests template priority override scenarios
   - Validates curated vs general pattern priority

6. **test_visualization_2samples.xlsx** (5.4KB) - OPTIONAL
   - Tests DisplaCy visualization generation
   - Validates color highlighting and formatting

**Recommendation**:
- **Required**: Copy `input_100texts.xlsx` only (as `sample_input.xlsx` and `sample_data.xlsx`)
- **Optional**: Copy additional test files if implementing specific feature tests
- **Total size**: 37KB (required) + 41KB (optional) = ~78KB total

### 3.7 Master Test Script

**CRITICAL**: Copy master test orchestrator:

```bash
# Copy master test runner
cp run_all_tests.py packages/medical_nlp_lean/tests/master_test_script.py
```

This script coordinates all test categories and provides comprehensive validation.

---

## 4. Models Directory Setup

### 4.1 Required Models

The package needs these models:

**BioBERT NER Models** (in `models/pretrained/`):
```
models/pretrained/
├── Disease/                     # Disease entity recognition (~411MB)
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── vocab.txt
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── Chemical/                    # Chemical entity recognition (~822MB)
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tf_model.h5
│   ├── vocab.txt
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── Gene/                        # Gene/Protein entity recognition (~411MB)
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── vocab.txt
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── README.md                    # Model download instructions
└── manifest.json                # Model registry
```

**SpaCy Models** (downloaded separately):
- `en_core_web_sm` - General English model
- `en_ner_bc5cdr_md` - Biomedical NER model

### 4.2 Model Download Instructions

**Option A: Copy from parent project** (Recommended for immediate use):
```bash
# BioBERT models already included from Step 2
cp -r models/pretrained/ packages/medical_nlp_lean/models/pretrained/

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_ner_bc5cdr_md
```

**Option B: Download BioBERT models from Hugging Face** (If not copying):

Include in README.md:

```bash
# 1. Install git-lfs (for large file storage)
git lfs install

# 2. Download BioBERT models
cd models/pretrained

# Disease NER model (~411MB)
git clone https://huggingface.co/alvaroalon2/biobert_diseases_ner Disease

# Chemical NER model (~822MB)
git clone https://huggingface.co/alvaroalon2/biobert_chemical_ner Chemical

# Gene/Protein NER model (~411MB)
git clone https://huggingface.co/alvaroalon2/biobert_genetic_ner Gene

cd ../..

# 3. Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_ner_bc5cdr_md
```

### 4.3 Model Configuration

Update `configs/pipeline_config.yaml`:

```yaml
models:
  # BioBERT NER models (local paths)
  disease_model: "models/pretrained/Disease"
  chemical_model: "models/pretrained/Chemical"
  gene_model: "models/pretrained/Gene"

  # SpaCy models (installed via pip)
  spacy_model: "en_core_web_sm"
  biomedical_model: "en_ner_bc5cdr_md"

  # Model cache directory
  model_cache_dir: "models/pretrained"
```

**Model Usage in Pipeline**:
- **Disease entities**: Uses `models/pretrained/Disease/` (BioBERT)
- **Chemical entities**: Uses `models/pretrained/Chemical/` (BioBERT)
- **Gene/Protein entities**: Uses `models/pretrained/Gene/` (BioBERT)
- **General NER**: Uses spaCy models as fallback
- **Total model size**: ~1.6GB (BioBERT) + SpaCy models

---

## 5. Comprehensive Testing Suite

### 5.1 Test Categories (10 Categories, 12 Test Scripts)

**Summary**: The lean package includes **12 test scripts** organized across **10 categories** for comprehensive pipeline validation.

**✅ All Tests Passing**: 12/12 (100%) as of October 9, 2025

---

### 5.2 Test Structure Overview

#### Actual Implementation (Current)

| Category | Test Scripts | Location | Purpose |
|----------|-------------|----------|---------|
| **scope_reversal** | test_scope_reversal.py<br>test_scope_reversal_v2.py | tests/features/ | Tests "denies X but has Y" patterns (103 patterns) |
| **template** | test_template_patterns.py | tests/integration/ | Validates 57,476 medical terms and pattern structure |
| **consistency** | test_consistency.py | tests/integration/ | CLI vs Streamlit output matching |
| **ui** | test_streamlit_display.py | tests/app/ | Streamlit UI component validation |
| **context** | test_context_classifications.py<br>test_context_overlap.py | tests/features/ | Context type classification and overlap resolution |
| **negation** | test_negation.py | tests/features/ | Negation pattern detection (99 patterns) |
| **confidence** | test_confidence_boundaries.py | tests/features/ | Confidence scoring and word boundary validation |
| **output** | test_excel_formatting.py | tests/excel_output/ | Excel output format and text marker validation |
| **visualization** | test_full_viz.py | tests/visualization/ | DisplaCy visualization rendering |
| **pipeline** | test_pipeline_validation.py | tests/validation/ | End-to-end pipeline validation |

### 5.3 Running Tests

#### Command Reference

```bash
# Run all tests (12 scripts)
./run_tests.sh
python tests/master_test_script.py

# Run quick/essential tests (scope_reversal, template, consistency)
./run_tests.sh --quick
python tests/master_test_script.py --quick

# Run only scope reversal tests
./run_tests.sh --scope
python tests/master_test_script.py --scope

# Run specific test category
python tests/master_test_script.py --category scope_reversal
python tests/master_test_script.py --category template
python tests/master_test_script.py --category consistency
python tests/master_test_script.py --category ui
python tests/master_test_script.py --category context
python tests/master_test_script.py --category negation
python tests/master_test_script.py --category confidence
python tests/master_test_script.py --category output
python tests/master_test_script.py --category visualization
python tests/master_test_script.py --category pipeline
```

#### Test Output Example

```
================================================================================
TEST EXECUTION SUMMARY
================================================================================
Total Tests: 12
✅ Passed: 12 (100.0%)
❌ Failed: 0 (0.0%)
📁 Missing: 0 (0.0%)

Results by Category:
  ✅ scope_reversal: 2/2 (100.0%)
  ✅ template: 1/1 (100.0%)
  ✅ consistency: 1/1 (100.0%)
  ...

🎉 EXCELLENT: All tests passed!
   ➡️ System is ready for production use

📁 Detailed results saved to: output/test_results/master_test_results_{timestamp}.txt
```

### 5.4 Original Plan Reference (For Historical Context)

The original implementation plan called for 50 test files across 8 categories. The actual lean package implementation was streamlined to 12 focused test scripts across 10 categories, providing comprehensive coverage while being more maintainable.

**Original categories planned** (not all implemented):
- Unit Tests (4 files)
- Integration Tests (3 files)
- Feature Tests (16 files)
- Excel Output Tests (4 files)

Validate 15-column Excel output with proper formatting

```bash
# From root
test_export_formatting_v2.py → tests/excel_output/test_export_formatting.py

# Create new tests
# test_excel_columns.py - Validate all 15 required columns present
# test_excel_formatting.py - Fonts, colors, entity highlighting
# test_context_formatting.py - Context sentence display formatting
```

**Validates**:
- ✅ All 15 columns present and correctly named
- ✅ Column formatting (fonts, colors, highlighting)
- ✅ Entity highlighting in context sentences
- ✅ Valid JSON in all_entities_json column
- ✅ Correct integer counts
- ✅ Proper comma-separated entity lists

---

#### Category 5: App/Streamlit Tests (`tests/app/`) - 10 files

Streamlit application functionality and UI testing

```bash
# From root
test_streamlit_buttons.py → tests/app/test_streamlit_buttons.py
test_streamlit_combined.py → tests/app/test_streamlit_combined.py
test_streamlit_context.py → tests/app/test_streamlit_context.py
test_streamlit_display_logic.py → tests/app/test_streamlit_display.py
test_file_upload_comparison.py → tests/app/test_file_upload.py
test_file_upload_comparison_with_report.py → tests/app/test_file_upload_report.py
test_pipeline_validation.py → tests/app/test_pipeline_validation.py

# From tests/ directory
tests/test_file_upload_consistency.py → tests/app/test_file_upload_consistency.py

# Create new tests
# test_export_results.py - Export Results button functionality
# test_entity_colors.py - Entity color highlighting in UI
```

**Validates**:
- ✅ Export Results button functionality
- ✅ File upload consistency (CLI vs Streamlit 100% match)
- ✅ Entity color highlighting
- ✅ Context display logic
- ✅ Pipeline validation
- ✅ Download buttons
- ✅ Combined features integration

---

#### Category 6: Visualization Tests (`tests/visualization/`) - 8 files

DisplaCy visualization and text display formatting

```bash
# From root
test_color_highlighting_complete.py → tests/visualization/test_color_highlighting.py
test_colored_visualization.py → tests/visualization/test_colored_visualization.py
test_combined_text_visualization.py → tests/visualization/test_combined_viz.py
test_full_viz.py → tests/visualization/test_full_viz.py
test_visualization_fix.py → tests/visualization/test_visualization_fix.py
test_visualization_simple.py → tests/visualization/test_visualization_simple.py
test_visualization_update.py → tests/visualization/test_visualization_update.py

# Create new
# test_font_size.py - Font size validation
```

**Validates**:
- ✅ Entity highlighting colors
- ✅ Font sizes
- ✅ Combined text visualization
- ✅ Full DisplaCy visualization
- ✅ Text markers for Excel
- ✅ Colored output formatting

---

#### Category 7: Validation Tests (`tests/validation/`) - 3 files

Input and path validation

```bash
# From root
test_path_validation.py → tests/validation/test_path_validation.py
test_flexible_input.py → tests/validation/test_flexible_input.py
test_pipeline_validation.py → tests/validation/test_pipeline_validation.py
```

**Validates**:
- ✅ Relative path handling (package portability)
- ✅ Multiple input formats (CSV, Excel, TXT)
- ✅ Pipeline component loading
- ✅ Template file validation

---

#### Category 8: Debugging Tests (`tests/debugging/`) - 6 files - **OPTIONAL**

⚠️ **Note**: These tests are OPTIONAL and used for investigating specific issues during development

```bash
# From root - Optional debugging tests
test_negation_detailed_debug.py → tests/debugging/test_negation_detailed.py
test_model_debug.py → tests/debugging/test_model_debug.py
test_kif5a_overlap.py → tests/debugging/test_kif5a_overlap.py
test_specific_overlap.py → tests/debugging/test_specific_overlap.py
test_export_formatting.py → tests/debugging/test_export_formatting_v1.py  # Legacy version

# Create README
# tests/debugging/README.md - Explains these are optional debug tests
```

**Purpose**:
- Step-by-step negation debugging
- Model behavior investigation
- Specific edge case analysis (e.g., KIF5A gene overlap)
- Legacy test comparison

**Recommendation**: **Skip** these tests for lean package unless debugging is needed

---

### 5.2 Master Test Script Structure

**`tests/master_test_script.py`** (Copy from `run_all_tests.py`):

```python
#!/usr/bin/env python3
"""
Master Test Script for Medical NLP Lean Package
Coordinates all test suites to validate complete pipeline functionality

Categories: 8 total (7 required + 1 optional debugging)
Total Tests: 50 files
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

class MasterTestRunner:
    """Comprehensive test suite runner for all 50 tests across 8 categories"""

    def __init__(self):
        self.package_root = Path(__file__).parent.parent
        self.test_categories = {
            'unit': {
                'description': 'Unit tests for core components (4 files)',
                'tests': [
                    'tests/unit/test_predictor.py',
                    'tests/unit/test_performance_analyzer.py',
                    'tests/unit/test_rules_loader.py',
                    'tests/unit/test_scope_reversal.py',
                ],
                'priority': 'critical',
                'required': True
            },
            'integration': {
                'description': 'Integration tests for pipeline flow (3 files)',
                'tests': [
                    'tests/integration/test_pipeline_integration.py',
                    'tests/integration/test_template_patterns.py',
                    'tests/integration/test_consistency.py',
                ],
                'priority': 'critical',
                'required': True
            },
            'features': {
                'description': 'Feature-specific validation (16 files)',
                'tests': [
                    'tests/features/test_entity_detection.py',
                    'tests/features/test_scope_reversal.py',
                    'tests/features/test_scope_reversal_v2.py',
                    'tests/features/test_negation.py',
                    'tests/features/test_negation_boundaries.py',
                    'tests/features/test_negation_debug.py',
                    'tests/features/test_confirmed_patterns.py',
                    'tests/features/test_historical_detection.py',
                    'tests/features/test_historical.py',
                    'tests/features/test_section_detection.py',
                    'tests/features/test_context_sentences.py',
                    'tests/features/test_context_classifications.py',
                    'tests/features/test_context_overlap.py',
                    'tests/features/test_entity_context.py',
                    'tests/features/test_confidence_boundaries.py',
                    'tests/features/test_family_detection.py',
                ],
                'priority': 'critical',
                'required': True
            },
            'excel_output': {
                'description': 'Excel output validation - 15 columns (4 files)',
                'tests': [
                    'tests/excel_output/test_excel_columns.py',
                    'tests/excel_output/test_excel_formatting.py',
                    'tests/excel_output/test_export_formatting.py',
                    'tests/excel_output/test_context_formatting.py',
                ],
                'priority': 'critical',
                'required': True
            },
            'app': {
                'description': 'Streamlit app functionality (10 files)',
                'tests': [
                    'tests/app/test_streamlit_buttons.py',
                    'tests/app/test_streamlit_combined.py',
                    'tests/app/test_streamlit_context.py',
                    'tests/app/test_streamlit_display.py',
                    'tests/app/test_export_results.py',
                    'tests/app/test_file_upload.py',
                    'tests/app/test_file_upload_report.py',
                    'tests/app/test_file_upload_consistency.py',
                    'tests/app/test_entity_colors.py',
                    'tests/app/test_pipeline_validation.py',
                ],
                'priority': 'high',
                'required': True
            },
            'visualization': {
                'description': 'Visualization and formatting (8 files)',
                'tests': [
                    'tests/visualization/test_color_highlighting.py',
                    'tests/visualization/test_colored_visualization.py',
                    'tests/visualization/test_combined_viz.py',
                    'tests/visualization/test_full_viz.py',
                    'tests/visualization/test_visualization_fix.py',
                    'tests/visualization/test_visualization_simple.py',
                    'tests/visualization/test_visualization_update.py',
                    'tests/visualization/test_font_size.py',
                ],
                'priority': 'high',
                'required': True
            },
            'validation': {
                'description': 'Path and input validation (3 files)',
                'tests': [
                    'tests/validation/test_path_validation.py',
                    'tests/validation/test_flexible_input.py',
                    'tests/validation/test_pipeline_validation.py',
                ],
                'priority': 'high',
                'required': True
            },
            'debugging': {
                'description': 'Debug/investigation tests (6 files) - OPTIONAL',
                'tests': [
                    'tests/debugging/test_negation_detailed.py',
                    'tests/debugging/test_model_debug.py',
                    'tests/debugging/test_kif5a_overlap.py',
                    'tests/debugging/test_specific_overlap.py',
                    'tests/debugging/test_export_formatting_v1.py',
                ],
                'priority': 'low',
                'required': False  # Optional debugging tests
            }
        }

    def run_all_tests(self):
        """Run all test categories and generate logs"""
        results = {
            'passed': [],
            'failed': [],
            'skipped': [],
            'total_time': 0
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = self.package_root / "tests" / "test_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        summary_file = log_dir / f"test_summary_{timestamp}.txt"

        for category, config in self.test_categories.items():
            # Skip optional debugging tests if requested
            if not config.get('required', True) and self.skip_debugging:
                continue

            for test_file in config['tests']:
                result = self._run_single_test(test_file, log_dir, timestamp)

                if result['status'] == 'passed':
                    results['passed'].append(result)
                elif result['status'] == 'failed':
                    results['failed'].append(result)
                    # Create individual failure log
                    self._write_failure_log(result, log_dir, timestamp)
                else:
                    results['skipped'].append(result)

        # Write summary file
        self._write_summary(results, summary_file, timestamp)

        return 0 if not results['failed'] else 1

    def _run_single_test(self, test_file, log_dir, timestamp):
        """Run a single test file and capture results"""
        # Implementation details
        pass

    def _write_failure_log(self, result, log_dir, timestamp):
        """Write individual test failure log"""
        test_name = Path(result['test_file']).stem
        log_file = log_dir / f"{test_name}_{timestamp}.log"

        with open(log_file, 'w') as f:
            f.write(f"Test Failure Log\n")
            f.write(f"{'='*80}\n")
            f.write(f"Test: {result['test_file']}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Duration: {result['duration']:.2f}s\n")
            f.write(f"\nError Output:\n")
            f.write(f"{'-'*80}\n")
            f.write(result['error_output'])
            f.write(f"\n{'='*80}\n")

    def _write_summary(self, results, summary_file, timestamp):
        """Write test execution summary"""
        with open(summary_file, 'w') as f:
            f.write(f"Medical NLP Lean Package - Test Execution Summary\n")
            f.write(f"{'='*80}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total Tests: {len(results['passed']) + len(results['failed']) + len(results['skipped'])}\n")
            f.write(f"Passed: {len(results['passed'])}\n")
            f.write(f"Failed: {len(results['failed'])}\n")
            f.write(f"Skipped: {len(results['skipped'])}\n")
            f.write(f"Total Time: {results['total_time']:.2f}s\n")
            f.write(f"{'='*80}\n\n")

            if results['failed']:
                f.write(f"Failed Tests:\n")
                f.write(f"{'-'*80}\n")
                for result in results['failed']:
                    f.write(f"  - {result['test_file']}\n")
                f.write(f"\n")

            f.write(f"Test logs available in: tests/test_logs/\n")

if __name__ == "__main__":
    runner = MasterTestRunner()
    sys.exit(runner.run_all_tests())
```

**Note**: The master test script includes:
- Individual failure logs: `tests/test_logs/{test_name}_{timestamp}.log`
- Summary logs: `tests/test_logs/test_summary_{timestamp}.txt`
- Automatic log directory creation
- Detailed error output capture
- Execution time tracking

### 5.3 Excel Output Requirements (15 Columns)

The pipeline MUST generate these exact columns:

1. **Text DisplaCy entities visualization** - HTML with entity highlighting
2. **detected_diseases** - Comma-separated disease entities
3. **total_diseases_count** - Integer count
4. **detected_genes** - Comma-separated gene/protein entities
5. **total_gene_count** - Integer count
6. **negated_entities** - Negated conditions with context sentences
7. **negated_entities_count** - Integer count
8. **historical_entities** - Past medical history with context
9. **historical_entities_count** - Integer count
10. **uncertain_entities** - Possible/speculative conditions with context
11. **uncertain_entities_count** - Integer count
12. **family_entities** - Family medical history with context
13. **family_entities_count** - Integer count
14. **section_categories** - Clinical sections (comma-separated)
15. **all_entities_json** - Complete JSON with all entity details

**Formatting Requirements**:
- Context sentences: Entity highlighted with special markers
- Fonts: Appropriate sizes for readability
- Colors: Different colors for different entity types
- Highlighting: Entities within context sentences highlighted
- JSON: Valid, parseable JSON

---

## 6. Script Updates

### 6.1 Create `run_ner_pipeline.sh`

```bash
#!/bin/bash
# Medical NLP Lean - NER Pipeline Runner
# Purpose: Execute NER pipeline with various options

# Get package root directory
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate conda environment
source "$PACKAGE_ROOT/activate_env.sh"

# Default paths
DEFAULT_INPUT="$PACKAGE_ROOT/data/raw/sample_input.xlsx"
DEFAULT_OUTPUT="$PACKAGE_ROOT/output/results/output_results_$(date +%Y%m%d_%H%M%S).xlsx"

# Parse arguments
INPUT_FILE="${1:-$DEFAULT_INPUT}"
OUTPUT_FILE="${2:-$DEFAULT_OUTPUT}"

# Show usage
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: ./run_ner_pipeline.sh [INPUT_FILE] [OUTPUT_FILE]"
    echo ""
    echo "Examples:"
    echo "  ./run_ner_pipeline.sh"
    echo "  ./run_ner_pipeline.sh my_data.xlsx"
    echo "  ./run_ner_pipeline.sh my_data.xlsx my_output.xlsx"
    exit 0
fi

# Run pipeline
echo "Running Medical NER Pipeline..."
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_FILE"

conda run -n py311_bionlp python "$PACKAGE_ROOT/src/enhanced_medical_ner_predictor.py" \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --verbose \
    --visualizations \
    --viz-samples 5

# Check success
if [ $? -eq 0 ]; then
    echo "✓ Pipeline completed successfully"
    echo "Results: $OUTPUT_FILE"
else
    echo "✗ Pipeline failed"
    exit 1
fi
```

### 6.2 Create `run_tests.sh`

```bash
#!/bin/bash
# Run comprehensive test suite with logging

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$PACKAGE_ROOT/activate_env.sh"

echo "Running Medical NLP Lean Test Suite..."
echo "Logs will be saved to: tests/test_logs/"

# Run master test script
conda run -n py311_bionlp python "$PACKAGE_ROOT/tests/master_test_script.py" "$@"

# Check exit code
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ All tests passed"
else
    echo "✗ Some tests failed - check logs in tests/test_logs/"
fi

exit $EXIT_CODE
```

### 6.3 Create `run_app.sh`

```bash
#!/bin/bash
# Launch Streamlit Application

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$PACKAGE_ROOT/activate_env.sh"

echo "Launching Medical NLP Streamlit App..."

# Default port
PORT="${1:-8501}"

conda run -n py311_bionlp streamlit run "$PACKAGE_ROOT/app/medical_nlp_app.py" \
    --server.port "$PORT"
```

### 6.4 Create `cleanup_test_logs.sh`

```bash
#!/bin/bash
# Cleanup old test logs
# Purpose: Remove test logs older than specified days (default: 30)

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PACKAGE_ROOT/tests/test_logs"
AGE_DAYS="${1:-30}"

echo "Cleaning up test logs older than $AGE_DAYS days..."

if [ ! -d "$LOG_DIR" ]; then
    echo "Test logs directory not found: $LOG_DIR"
    exit 1
fi

# Find and remove old log files
DELETED_COUNT=0

# Remove old test failure logs
find "$LOG_DIR" -name "test_*.log" -type f -mtime +$AGE_DAYS -print0 | while IFS= read -r -d '' file; do
    echo "Removing: $file"
    rm -f "$file"
    ((DELETED_COUNT++))
done

# Remove old summary logs
find "$LOG_DIR" -name "test_summary_*.txt" -type f -mtime +$AGE_DAYS -print0 | while IFS= read -r -d '' file; do
    echo "Removing: $file"
    rm -f "$file"
    ((DELETED_COUNT++))
done

echo "Cleanup complete. Removed logs older than $AGE_DAYS days."
echo "Remaining logs:"
ls -lh "$LOG_DIR" | grep -v "^d" | grep -v ".gitkeep" || echo "  (no log files)"
```

**Usage**:
```bash
# Remove logs older than 30 days (default)
./cleanup_test_logs.sh

# Remove logs older than 7 days
./cleanup_test_logs.sh 7

# Remove logs older than 1 day
./cleanup_test_logs.sh 1
```

### 6.5 Update Python Script Paths

In ALL copied Python files, update paths to use relative imports:

**Pattern to find and replace**:
```python
# OLD (hardcoded paths)
template_path = "data/external/target_rules_template.xlsx"

# NEW (relative paths)
from pathlib import Path
PACKAGE_ROOT = Path(__file__).parent.parent
template_path = PACKAGE_ROOT / "data" / "external" / "target_rules_template.xlsx"
```

**In `src/enhanced_medical_ner_predictor.py`**:
```python
# Add at top of file
from pathlib import Path
import sys

# Get package root
PACKAGE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

# Update all paths
TEMPLATE_DIR = PACKAGE_ROOT / "data" / "external"
MODEL_DIR = PACKAGE_ROOT / "models" / "pretrained"
OUTPUT_DIR = PACKAGE_ROOT / "output"
```

**In `app/medical_nlp_app.py`**:
```python
# Update imports
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from enhanced_medical_ner_predictor import EnhancedMedicalNERPredictor
from enhanced_performance_analyzer import EnhancedPerformanceAnalyzer
# ... other imports
```

---

## 7. Configuration Files

### 7.1 `requirements.txt`

```txt
# Core dependencies
spacy==3.7.2
negspacy
pandas>=2.0.0
openpyxl
numpy>=1.24.0

# Streamlit
streamlit>=1.28.0

# Deep Learning
transformers>=4.35.0
torch>=2.1.0
scikit-learn>=1.3.0

# Utilities
python-dotenv
pyyaml
tqdm
click
```

### 7.2 `requirements-test.txt`

```txt
# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.12.0
pytest-asyncio>=0.21.0

# Code quality (optional but recommended)
black>=23.12.0
flake8>=7.0.0
mypy>=1.8.0
```

### 7.3 `setup.py`

```python
from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="medical_nlp_lean",
    version="1.0.0",
    description="Production-ready Medical NLP pipeline with Streamlit interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Medical NLP Team",
    author_email="team@example.com",
    url="https://github.com/your-org/medical-nlp-lean",
    packages=find_packages(),
    package_data={
        "medical_nlp_lean": [
            "data/external/*.xlsx",
            "configs/*.yaml",
        ]
    },
    install_requires=[
        "spacy==3.7.2",
        "negspacy",
        "pandas>=2.0.0",
        "openpyxl",
        "streamlit>=1.28.0",
        "transformers>=4.35.0",
        "torch>=2.1.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ]
    },
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "medical-nlp=src.enhanced_medical_ner_predictor:main",
        ]
    },
)
```

### 7.4 `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=65.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "medical_nlp_lean"
version = "1.0.0"
description = "Medical NLP Pipeline - Production Lean Package"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=src --cov-report=html --cov-report=term"
```

### 7.5 `Makefile`

```makefile
.PHONY: help install test test-fast lint clean run-pipeline run-app

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install:  ## Install package
	conda env create -f py311_bionlp_environment.yml
	conda run -n py311_bionlp pip install -e .
	conda run -n py311_bionlp python -m spacy download en_core_web_sm
	conda run -n py311_bionlp python -m spacy download en_ner_bc5cdr_md

test:  ## Run all tests
	./run_tests.sh

test-fast:  ## Run fast tests only
	./run_tests.sh --fast

test-category:  ## Run specific test category (e.g., make test-category CATEGORY=excel_output)
	./run_tests.sh --category $(CATEGORY)

lint:  ## Run linters (optional)
	conda run -n py311_bionlp flake8 src/ tests/
	conda run -n py311_bionlp black --check src/ tests/

clean:  ## Clean generated files
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	rm -rf output/results/* output/visualizations/* output/logs/*

run-pipeline:  ## Run NER pipeline
	./run_ner_pipeline.sh

run-app:  ## Run Streamlit app
	./run_app.sh
```

### 7.6 `configs/pipeline_config.yaml`

```yaml
pipeline:
  confidence_thresholds:
    curated_templates: 0.3
    general_patterns: 0.5
  proximity_weighting:
    max_boost: 0.3

templates:
  target_rules: "data/external/target_rules_template.xlsx"
  historical_rules: "data/external/historical_rules_template.xlsx"
  negated_rules: "data/external/negated_rules_template.xlsx"
  uncertainty_rules: "data/external/uncertainty_rules_template.xlsx"
  confirmed_rules: "data/external/confirmed_rules_template.xlsx"
  family_rules: "data/external/family_rules_template.xlsx"

models:
  spacy_model: "models/pretrained/en_ner_bc5cdr_md"
  biobert_model: "dmis-lab/biobert-base-cased-v1.1"
  model_cache_dir: "models/pretrained"

output:
  results_dir: "output/results"
  visualizations_dir: "output/visualizations"
  logs_dir: "output/logs"
  reports_dir: "output/reports"
  exports_dir: "output/exports"
```

### 7.7 `.gitignore`

```
# Python
__pycache__/
*.py[cod]
*.so
.Python
*.egg-info/
dist/
build/

# Virtual environments
venv/
.venv/
env/

# Models (large files)
models/pretrained/*
!models/pretrained/.gitkeep
models/trained/*
!models/trained/.gitkeep

# Outputs
output/results/*
!output/results/.gitkeep
output/visualizations/*
!output/visualizations/.gitkeep
output/logs/*
!output/logs/.gitkeep
output/reports/*
!output/reports/.gitkeep
output/exports/*
!output/exports/.gitkeep

# Testing
.pytest_cache/
.coverage
htmlcov/

# Test logs (keep .gitkeep, ignore log files)
tests/test_logs/*.log
tests/test_logs/*.txt
!tests/test_logs/.gitkeep

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
```

---

## 8. Installation & Usage

### 8.1 Installation Steps

```bash
# 1. Navigate to package
cd packages/medical_nlp_lean

# 2. Create conda environment
conda env create -f py311_bionlp_environment.yml

# 3. Activate environment
conda activate py311_bionlp
# OR
source activate_env.sh

# 4. Install package
pip install -e .

# 5. Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download en_ner_bc5cdr_md

# 6. Run tests to verify installation
./run_tests.sh
```

### 8.2 CLI Usage

```bash
# Basic run (uses sample_input.xlsx)
./run_ner_pipeline.sh

# Custom input
./run_ner_pipeline.sh path/to/input.xlsx

# Custom input and output
./run_ner_pipeline.sh input.xlsx output.xlsx

# Direct Python call with all options
conda run -n py311_bionlp python src/enhanced_medical_ner_predictor.py \
    --input data/raw/sample_input.xlsx \
    --output output/results/results.xlsx \
    --verbose \
    --visualizations \
    --viz-samples 10 \
    --json \
    --validate
```

### 8.3 Streamlit Usage

```bash
# Launch Streamlit app (default port 8501)
./run_app.sh

# Custom port
./run_app.sh 8502

# Direct call
conda run -n py311_bionlp streamlit run app/medical_nlp_app.py
```

### 8.4 Testing

```bash
# Run all tests
./run_tests.sh

# Run specific category
./run_tests.sh --category excel_output
./run_tests.sh --category app

# Run fast tests only
./run_tests.sh --fast

# Using Makefile
make test                        # All tests
make test-fast                   # Fast tests
make test-category CATEGORY=app  # App/Streamlit tests
```

### 8.5 Using Makefile

```bash
# Show all available commands
make help

# Install everything
make install

# Run pipeline
make run-pipeline

# Run Streamlit app
make run-app

# Clean outputs
make clean
```

---

## 9. Validation Checklist

### 9.1 Installation Validation

- [ ] Conda environment created successfully
- [ ] All dependencies installed without errors
- [ ] spaCy models downloaded (`en_core_web_sm`, `en_ner_bc5cdr_md`)
- [ ] Package installed in editable mode (`pip install -e .`)
- [ ] Models directory populated (~1-2GB)
- [ ] Template files present (all 6 files)
- [ ] Sample input data available (`sample_input.xlsx`)

### 9.2 CLI Validation

- [ ] `./run_ner_pipeline.sh` executes without errors
- [ ] Sample input processes successfully
- [ ] Excel output generated with 15 columns
- [ ] All column names correct
- [ ] All column data properly formatted
- [ ] Visualizations created (PNG files in `output/visualizations/`)
- [ ] Performance report generated (`output/reports/`)
- [ ] Logs created (`output/logs/`)
- [ ] Output paths relative to package root

### 9.3 Streamlit Validation

- [ ] `./run_app.sh` launches app successfully
- [ ] App loads without errors
- [ ] File upload works (CSV, Excel, TXT)
- [ ] Text input works
- [ ] Export Results button functional
- [ ] Download buttons work
- [ ] Entity colors display correctly
- [ ] Context sentences formatted properly
- [ ] CLI and Streamlit outputs match 100%

### 9.4 Testing Validation

- [ ] `./run_tests.sh` runs master test script
- [ ] All unit tests pass (4 tests)
- [ ] Integration tests pass (3 tests)
- [ ] Feature tests pass (9 tests - entity, scope, negation, family, historical, section, context, entity_context, visualization)
- [ ] Excel output tests pass (4 tests - columns, formatting, export, context)
- [ ] App/Streamlit tests pass (7 tests - buttons, export, upload, upload_consistency, colors, display, validation)
- [ ] Visualization tests pass (4 tests - highlighting, font, combined, full)
- [ ] Validation tests pass (3 tests - paths, flexible input, pipeline)
- [ ] Test coverage > 80%

### 9.5 Excel Output Validation

Verify all 15 columns:

- [ ] Column 1: `Text DisplaCy entities visualization` (HTML)
- [ ] Column 2: `detected_diseases` (comma-separated)
- [ ] Column 3: `total_diseases_count` (integer)
- [ ] Column 4: `detected_genes` (comma-separated)
- [ ] Column 5: `total_gene_count` (integer)
- [ ] Column 6: `negated_entities` (with context)
- [ ] Column 7: `negated_entities_count` (integer)
- [ ] Column 8: `historical_entities` (with context)
- [ ] Column 9: `historical_entities_count` (integer)
- [ ] Column 10: `uncertain_entities` (with context)
- [ ] Column 11: `uncertain_entities_count` (integer)
- [ ] Column 12: `family_entities` (with context)
- [ ] Column 13: `family_entities_count` (integer)
- [ ] Column 14: `section_categories` (comma-separated)
- [ ] Column 15: `all_entities_json` (valid JSON)

**Formatting checks**:
- [ ] Entity highlighting in context sentences
- [ ] Font sizes appropriate
- [ ] Colors correct for entity types
- [ ] JSON is valid and parseable

### 9.6 Feature Validation

- [ ] Entity detection using 57,476 target rules
- [ ] Negation detection (99 patterns) working
- [ ] Historical detection (82 patterns) working
- [ ] Family detection (79 patterns) working
- [ ] Uncertainty detection (48 patterns) working
- [ ] Confirmed entity detection (138 patterns) working
- [ ] Scope reversal detection (103 patterns) working
- [ ] Section header detection working
- [ ] Context overlap resolved correctly (priority-based)
- [ ] Confidence scoring correct (0.3 curated vs 0.5 general)
- [ ] Proximity weighting applied
- [ ] False positive suppression working

### 9.7 Portability Validation

- [ ] Package works independently (no parent dependency)
- [ ] Can be copied to another location and still functions
- [ ] All paths relative to package root
- [ ] No hardcoded absolute paths
- [ ] Works on different machines
- [ ] Documentation complete and accurate

---

## 10. Success Criteria

The lean package is **READY FOR PRODUCTION** when:

✅ **Installation**: Environment setup completes without errors
✅ **CLI**: Pipeline executes and generates correct 15-column Excel output
✅ **Streamlit**: App launches and all features work
✅ **Testing**: All 7 test categories pass (>95% pass rate)
✅ **Consistency**: CLI and Streamlit outputs match 100%
✅ **Excel**: Output has all 15 columns with correct formatting
✅ **Features**: All detection rules working (entity, negation, scope, family, historical, etc.)
✅ **Models**: Models directory populated or download documented
✅ **Portability**: Package works independently
✅ **Documentation**: README provides complete setup/usage instructions
✅ **Performance**: Processing time acceptable (<1 min per 100 texts)

---

## 11. Key Implementation Steps Summary

### Step 1: Create Directory Structure
```bash
cd packages
mkdir -p medical_nlp_lean
# Create all subdirectories as per structure above
```

### Step 2: Copy Core Files
```bash
# Core Python scripts
cp enhanced_medical_ner_predictor.py packages/medical_nlp_lean/src/
cp enhanced_performance_analyzer.py packages/medical_nlp_lean/src/
cp enhanced_target_rules_loader.py packages/medical_nlp_lean/src/
cp scope_reversal_engine.py packages/medical_nlp_lean/src/

# Streamlit app
cp app/bio_ner_streamlit_app.py packages/medical_nlp_lean/app/medical_nlp_app.py

# CRITICAL: Copy models directory
cp -r models/ packages/medical_nlp_lean/models/

# Environment files
cp py311_bionlp_environment.yml packages/medical_nlp_lean/
cp activate_py311_bionlp.sh packages/medical_nlp_lean/activate_env.sh
```

### Step 3: Copy Template Files
```bash
cp data/external/*.xlsx packages/medical_nlp_lean/data/external/
```

### Step 4: Copy Test Files
```bash
# Master test script
cp run_all_tests.py packages/medical_nlp_lean/tests/master_test_script.py

# Copy all test files according to mapping in Section 5.1
# ... (40+ test files across 7 categories)
```

### Step 5: Copy Sample Data
```bash
# Primary sample input for running (REQUIRED)
cp data/raw/input_100texts.xlsx packages/medical_nlp_lean/data/raw/sample_input.xlsx

# Primary sample data for testing (REQUIRED)
cp data/raw/input_100texts.xlsx packages/medical_nlp_lean/tests/test_data/sample_data.xlsx

# Optional: Additional test files for specific feature testing
cp data/raw/confidence_comparison_test.xlsx packages/medical_nlp_lean/tests/test_data/
cp data/raw/context_test_results_complete.xlsx packages/medical_nlp_lean/tests/test_data/
cp data/raw/kif5a_test.xlsx packages/medical_nlp_lean/tests/test_data/
cp data/raw/template_override_test.xlsx packages/medical_nlp_lean/tests/test_data/
cp data/raw/test_visualization_2samples.xlsx packages/medical_nlp_lean/tests/test_data/
```

### Step 6: Update Path References
```bash
# In all Python files, update paths to be relative
# Pattern: PACKAGE_ROOT = Path(__file__).parent.parent
# Update template paths, model paths, output paths
```

### Step 7: Create Configuration Files
```bash
# Create requirements.txt, setup.py, pyproject.toml, Makefile, .gitignore
# See Section 7 for complete content
```

### Step 8: Create Execution Scripts
```bash
# Create run_ner_pipeline.sh, run_tests.sh, run_app.sh
# Make executable: chmod +x *.sh
```

### Step 9: Test Everything
```bash
cd packages/medical_nlp_lean

# Install
conda env create -f py311_bionlp_environment.yml
conda activate py311_bionlp
pip install -e .

# Download models
python -m spacy download en_core_web_sm
python -m spacy download en_ner_bc5cdr_md

# Run tests
./run_tests.sh

# Test CLI
./run_ner_pipeline.sh

# Test Streamlit
./run_app.sh
```

---

## 12. Post-Creation Tasks

After creating the package:

1. **Run full test suite**: `./run_tests.sh`
2. **Verify all outputs**: Check Excel, visualizations, logs, reports
3. **Test Streamlit export**: Upload file, export results, verify 15 columns
4. **Test portability**: Copy to different location, verify it works
5. **Update README**: Add any missing instructions
6. **Create distribution**: Package for deployment (with/without models)
7. **Document model download**: Clear instructions if models not included
8. **Create .gitkeep files**: For empty directories
9. **Test on clean environment**: Fresh conda env to validate setup
10. **Performance benchmark**: Time pipeline execution

---

## 13. Distribution Options

### Option A: Full Bundle (Recommended for Deployment)
- **Includes**: BioBERT models (~1.6GB) + templates + code
- **Size**: ~1.7GB total
- **Pros**: Users get working package immediately, no model download needed
- **Cons**: Large download size
- **Best for**: Production deployment, end users, offline environments

### Option B: Lean Package (Recommended for Development)
- **Includes**: Code + templates only
- **Excludes**: BioBERT models (models/pretrained/)
- **Size**: ~50-100MB
- **Pros**: Small download, fast git operations
- **Cons**: Users must download models separately (~1.6GB)
- **Best for**: Development, version control, CI/CD

**Model Download for Option B**:
```bash
# Users run these commands after installation:
cd packages/medical_nlp_lean/models/pretrained
git lfs install
git clone https://huggingface.co/alvaroalon2/biobert_diseases_ner Disease
git clone https://huggingface.co/alvaroalon2/biobert_chemical_ner Chemical
git clone https://huggingface.co/alvaroalon2/biobert_genetic_ner Gene
```

### Option C: Docker Image
- **Includes**: Everything (code + templates + models + environment)
- **Size**: ~3-4GB
- **Pros**: Easiest deployment, reproducible environment
- **Cons**: Largest size
- **Best for**: Production deployment, cloud hosting, containerized environments

**Docker Build**:
```dockerfile
# Includes BioBERT models in image
COPY models/pretrained/ /app/models/pretrained/
# Total image size: ~3-4GB
```

---

## 14. Critical Features to Validate

### Entity Detection
- ✅ 57,476 target medical terms
- ✅ Confidence thresholds: 0.3 (curated) vs 0.5 (general)
- ✅ Proximity weighting (max +0.3)
- ✅ False positive suppression
- ✅ Medical entity priority: Diseases > Genes > Drugs > Anatomy

### Context Classification
- ✅ Negation detection (99 patterns)
- ✅ Historical detection (82 patterns)
- ✅ Family history (79 patterns)
- ✅ Uncertainty detection (48 patterns)
- ✅ Confirmed entities (138 patterns)
- ✅ Priority hierarchy: Negated > Family > Historical > Uncertain > Confirmed
- ✅ No overlap: Each entity in exactly ONE category

### Scope Reversal
- ✅ 103 patterns with confidence 70-95%
- ✅ Adversative conjunctions ("but", "however", "although")
- ✅ Temporal transitions
- ✅ Exception patterns
- ✅ Concessive conjunctions

### Section Detection
- ✅ Chief Complaint
- ✅ Assessment
- ✅ Plan
- ✅ History of Present Illness
- ✅ Past Medical History
- ✅ Family History
- ✅ Social History
- ✅ Review of Systems

### Visualization
- ✅ Entity highlighting with colors
- ✅ Context sentence formatting
- ✅ Font size appropriateness
- ✅ DisplaCy HTML generation
- ✅ Text markers for Excel display

### Streamlit Features
- ✅ File upload (CSV, Excel, TXT)
- ✅ Text input
- ✅ Export Results button
- ✅ Download buttons
- ✅ Entity color display
- ✅ Real-time processing
- ✅ 100% consistency with CLI output

---

## 15. Reference Documents

- **Complete Implementation Guide**: `LEAN_PACKAGE_CREATION_GUIDE.md`
- **Main Project Documentation**: `CLAUDE.md`, `PROJECT_DOCUMENTATION.md`
- **Quick Reference**: `QUICK_REFERENCE.md`
- **Architecture**: `ENTITY_DETECTION_ARCHITECTURE.md`

---

**Version**: 2.0 (Comprehensive Merged Guide)
**Last Updated**: 2025-10-08
**Purpose**: Complete guide for creating production-ready Medical NLP lean package
**Maintainer**: Medical NLP Team
