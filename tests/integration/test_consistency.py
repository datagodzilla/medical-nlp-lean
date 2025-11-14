#!/usr/bin/env python3
"""Test consistency between CLI and Streamlit predictions"""

import pandas as pd
from pathlib import Path
from enhanced_medical_ner_predictor import EnhancedMedicalNERPredictor

# Test text
test_text = "Patient has diabetes and hypertension. No chest pain."

print("="*80)
print("CONSISTENCY TEST: CLI vs Streamlit")
print("="*80)

# Initialize predictor (same as Streamlit does)
print("\n🔄 Initializing predictor...")
predictor = EnhancedMedicalNERPredictor(model_name='en_core_web_sm', use_gpu=False)

# Test 1: Direct entity extraction (what Streamlit does for text input)
print("\n" + "="*80)
print("TEST 1: Direct entity extraction (Streamlit text input mode)")
print("="*80)
result = predictor.extract_entities(test_text)

print(f"\n✅ Confirmed entities: {result.get('confirmed_entities', '')}")
print(f"   Context sentences: {result.get('confirmed_context_sentences', '')[:100]}...")

print(f"\n❌ Negated entities: {result.get('negated_entities', '')}")
print(f"   Context sentences: {result.get('negated_context_sentences', '')[:100]}...")

# Test 2: DataFrame prediction (what Streamlit does for file upload and CLI does)
print("\n" + "="*80)
print("TEST 2: DataFrame prediction (Streamlit file upload & CLI mode)")
print("="*80)

# Create test DataFrame
test_df = pd.DataFrame({
    'Index': [0],
    'Text': [test_text]
})

# Predict using DataFrame method (same as CLI)
df_result = predictor.predict_dataframe(test_df)

print(f"\n✅ Confirmed entities: {df_result['confirmed_entities'].iloc[0]}")
print(f"   Context sentences: {df_result['confirmed_context_sentences'].iloc[0][:100]}...")

print(f"\n❌ Negated entities: {df_result['negated_entities'].iloc[0]}")
print(f"   Context sentences: {df_result['negated_context_sentences'].iloc[0][:100]}...")

# Test 3: Verify columns match
print("\n" + "="*80)
print("TEST 3: DataFrame output columns")
print("="*80)
print(f"Total columns: {len(df_result.columns)}")

context_cols = [col for col in df_result.columns if 'context_sentences' in col]
print(f"\n📋 Context sentence columns ({len(context_cols)}):")
for col in context_cols:
    value = df_result[col].iloc[0]
    has_content = "✅ HAS CONTENT" if value and str(value) != 'nan' else "❌ EMPTY"
    print(f"   {col:40s} {has_content}")

# Test 4: Check predictor location
print("\n" + "="*80)
print("TEST 4: Verify same predictor module")
print("="*80)
import inspect
module_file = inspect.getfile(EnhancedMedicalNERPredictor)
print(f"Predictor location: {module_file}")
print(f"Same module as CLI: {'✅ YES' if 'Clindocs_NLP/enhanced_medical_ner_predictor.py' in module_file else '❌ NO'}")

print("\n" + "="*80)
print("✅ CONSISTENCY TEST COMPLETE")
print("="*80)
