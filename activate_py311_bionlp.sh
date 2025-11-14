#!/bin/bash
# activate_py311_bionlp.sh
# Activation script for the py311_bionlp environment

echo "🔬 Activating py311_bionlp environment..."

# Activate conda environment
# Initialize conda for shell use
if command -v conda &> /dev/null; then
    # Use conda from PATH
    eval "$(conda shell.bash hook)"
else
    # Try common conda installation paths
    for CONDA_PATH in "$HOME/anaconda3/bin/conda" "$HOME/miniconda3/bin/conda" "/opt/anaconda3/bin/conda" "/opt/miniconda3/bin/conda"; do
        if [ -f "$CONDA_PATH" ]; then
            eval "$($CONDA_PATH shell.bash hook)"
            break
        fi
    done
fi
conda activate py311_bionlp

echo "✅ py311_bionlp environment activated!"
echo "🏥 Available NLP models and packages:"
echo "   • spaCy 3.7.2 with en_core_web_sm model"
echo "   • negspaCy for negation detection"
echo "   • pandas, numpy, openpyxl for data processing"
echo ""
echo "🚀 To run the enhanced medical NER pipeline:"
echo "   python enhanced_medical_ner_predictor.py"
echo ""
echo "📁 Input file: data/raw/context_test_results_complete.xlsx"
echo "📁 Output directory: output/"