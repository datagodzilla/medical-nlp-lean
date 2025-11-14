#!/usr/bin/env python3
"""
Bio-NER Streamlit Visualization App
===================================

A comprehensive Streamlit application for visualizing biomedical named entity recognition
using the enhanced medical NER pipeline with BioBERT models for diseases, chemicals, and genetics.

Features:
- Multi-domain NER (diseases, chemicals, genetics)
- Interactive text input and file upload
- Real-time entity visualization with color-coded highlighting
- Model performance metrics
- Export capabilities
- Automatic pipeline validation

Author: Enhanced Medical NER Pipeline Project
Date: September 30, 2025
"""

import sys
import os
from pathlib import Path

# Add current directory to path to import our enhanced predictor
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
src_dir = parent_dir / "src"
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(parent_dir))

import streamlit as st
import pandas as pd
import json
import io
from datetime import datetime
import base64
import numpy as np
from typing import Dict, List, Any

# Import our enhanced medical NER predictor
try:
    from enhanced_medical_ner_predictor import EnhancedMedicalNERPredictor, validate_pipeline_performance, save_formatted_excel
    PREDICTOR_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Could not import enhanced medical NER predictor: {e}")
    st.info("Make sure you're running this from the project root directory.")
    PREDICTOR_AVAILABLE = False

# Helper Functions for Export
def export_text_results_fast(texts, results):
    """Export text analysis results to Excel file - optimized version."""
    import pandas as pd
    from datetime import datetime
    from pathlib import Path
    import os

    # Create output directory if it doesn't exist
    output_dir = Path("output/exports")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output_results_{timestamp}.xlsx"
    filepath = output_dir / filename

    # Prepare data for export
    export_data = []

    # Handle both single and multiple results
    if not isinstance(texts, list):
        texts = [texts]
    if not isinstance(results, list):
        results = [results]

    for idx, (text, result) in enumerate(zip(texts, results)):
        # Extract entities from result
        entities = result.get('entities', [])

        # Add row for each entity
        for entity in entities:
            export_data.append({
                'Text_Index': idx + 1,
                'Original_Text': text[:100] + '...' if len(text) > 100 else text,
                'Entity': entity.get('text', ''),
                'Type': entity.get('label', ''),
                'Start_Position': entity.get('start', ''),
                'End_Position': entity.get('end', ''),
                'Confidence': entity.get('confidence', 1.0)
            })

        # If no entities, add a row with just the text
        if not entities:
            export_data.append({
                'Text_Index': idx + 1,
                'Original_Text': text[:100] + '...' if len(text) > 100 else text,
                'Entity': 'No entities found',
                'Type': '',
                'Start_Position': '',
                'End_Position': '',
                'Confidence': ''
            })

    # Create DataFrame and save to Excel
    df = pd.DataFrame(export_data)

    # Write to Excel with basic formatting
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='NER_Results', index=False)

        # Auto-adjust column widths
        worksheet = writer.sheets['NER_Results']
        for column in df:
            column_width = max(df[column].astype(str).map(len).max(), len(column))
            column_width = min(column_width, 50)  # Cap at 50 characters
            col_idx = df.columns.get_loc(column)
            worksheet.column_dimensions[chr(65 + col_idx)].width = column_width + 2

    return str(filepath)

def export_entire_dataframe(df, predictor):
    """Export entire DataFrame with NER predictions to Excel."""
    import pandas as pd
    from datetime import datetime
    from pathlib import Path

    # Create output directory if it doesn't exist
    output_dir = Path("output/exports")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output_results_{timestamp}.xlsx"
    filepath = output_dir / filename

    # Create a copy of the DataFrame
    export_df = df.copy()

    # Add NER results columns
    if predictor and 'text_column' in st.session_state:
        text_col = st.session_state.text_column

        # Process each row
        entities_list = []
        entity_counts = []

        for idx, row in export_df.iterrows():
            text = row[text_col] if text_col in row and pd.notna(row[text_col]) else ""

            if text:
                try:
                    # Process text with predictor
                    result = predictor.process_text(str(text))
                    entities = result.get('entities', [])

                    # Format entities for display
                    entity_strings = [f"{e['text']} ({e['label']})" for e in entities]
                    entities_list.append('; '.join(entity_strings))
                    entity_counts.append(len(entities))
                except:
                    entities_list.append('')
                    entity_counts.append(0)
            else:
                entities_list.append('')
                entity_counts.append(0)

        # Add new columns
        export_df['Detected_Entities'] = entities_list
        export_df['Entity_Count'] = entity_counts

    # Write to Excel
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        export_df.to_excel(writer, sheet_name='Data_with_NER', index=False)

        # Auto-adjust column widths
        worksheet = writer.sheets['Data_with_NER']
        for column in export_df:
            column_width = max(export_df[column].astype(str).map(len).max(), len(column))
            column_width = min(column_width, 50)  # Cap at 50 characters
            col_idx = export_df.columns.get_loc(column)
            if col_idx < 26:  # Basic column letter handling
                worksheet.column_dimensions[chr(65 + col_idx)].width = column_width + 2

    return str(filepath)

def process_text(text, predictor):
    """Process single text using the enhanced medical NER predictor."""
    if predictor is None:
        return {}

    try:
        # Use the correct method from the predictor
        result = predictor.extract_entities(text)

        # Parse entities from JSON if needed and format results
        if 'all_entities_json' in result:
            import json
            try:
                entities_data = json.loads(result['all_entities_json'])
                entities = entities_data.get('entities', [])
            except:
                entities = []
        else:
            entities = []

        # Create formatted text with entity highlighting
        formatted_text = result.get('Text Visualization', text)

        # Count entities by type
        entity_counts = {
            'diseases': result.get('total_diseases_count', 0),
            'genes': result.get('total_gene_count', 0),
            'chemicals': len(result.get('detected_chemicals', [])),
            'total': len(entities),
            'confirmed': len(result.get('confirmed_entities', '').split(', ')) if result.get('confirmed_entities') else 0,
            'negated': len(result.get('negated_entities', '').split(', ')) if result.get('negated_entities') else 0,
            'uncertain': len(result.get('uncertain_entities', '').split(', ')) if result.get('uncertain_entities') else 0,
            'historical': len(result.get('historical_entities', '').split(', ')) if result.get('historical_entities') else 0,
            'family': len(result.get('family_entities', '').split(', ')) if result.get('family_entities') else 0
        }

        return {
            'result': result,
            'entities': entities,
            'formatted_text': formatted_text,
            'entity_counts': entity_counts
        }

    except Exception as e:
        import streamlit as st
        st.error(f"❌ Error processing text: {str(e)}")
        return {}

def process_multiple_texts(texts, predictor):
    """Process multiple texts with the predictor - with timeout protection."""
    import streamlit as st
    import time

    results = []
    start_time = time.time()
    timeout_seconds = 300  # 5 minute timeout

    for idx, text in enumerate(texts):
        # Check timeout
        if time.time() - start_time > timeout_seconds:
            st.error(f"❌ Processing timeout after {timeout_seconds/60:.1f} minutes. Processed {idx}/{len(texts)} texts.")
            break

        try:
            # Show progress every 10 texts (only for large batches)
            if len(texts) > 10 and idx % 10 == 0 and idx > 0:
                elapsed = time.time() - start_time
                st.info(f"📊 Progress: {idx}/{len(texts)} texts processed ({elapsed:.1f}s elapsed)")

            # Use the process_text wrapper function for consistency
            result = process_text(text, predictor)
            results.append(result)

        except Exception as e:
            st.warning(f"⚠️ Error processing text {idx + 1}: {str(e)}")
            # Return empty result if processing fails
            results.append({
                'entities': [],
                'entity_counts': {'diseases': 0, 'genes': 0, 'chemicals': 0, 'total': 0}
            })

    return results

def _adjust_to_word_boundaries(text, start, end):
    """Adjust entity boundaries to word boundaries."""
    if start < 0 or end > len(text) or start >= end:
        return start, end

    # Adjust start to beginning of word
    adjusted_start = start
    while adjusted_start > 0 and not text[adjusted_start - 1].isspace():
        adjusted_start -= 1

    # Adjust end to end of word
    adjusted_end = end
    while adjusted_end < len(text) and not text[adjusted_end].isspace():
        adjusted_end += 1

    return adjusted_start, adjusted_end

def _find_predictor_for_entity(text, entity_text, start_pos, patterns):
    """Find which pattern predicted this entity."""
    # Simple implementation - in practice would check patterns
    # For now, return a generic response
    return "Pattern match"

# Page configuration
st.set_page_config(
    page_title="Enhanced Bio-NER Entity Visualizer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force dark theme
st._config.set_option("theme.base", "dark")

# Custom CSS for dark theme and better visualization
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Dark theme for main app */
    .stApp {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }

    .main > div {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }

    /* Dark theme for main content area */
    .block-container {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }

    /* Clean, simple sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #262730 !important;
    }

    section[data-testid="stSidebar"] > div {
        background-color: #262730 !important;
    }

    /* Sidebar text */
    section[data-testid="stSidebar"] * {
        color: #fafafa !important;
    }

    /* Simple working button styling */
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #ff6b6b !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.5rem 1rem !important;
        width: 100% !important;
        cursor: pointer !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #ff5252 !important;
    }

    /* Simple input styling */
    section[data-testid="stSidebar"] .stSelectbox select,
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stTextArea textarea {
        background-color: #1e1e1e !important;
        color: #fafafa !important;
        border: 1px solid #4a4a4a !important;
    }

    /* Configuration box styling - match sidebar background */
    section[data-testid="stSidebar"] .block-container {
        background-color: #262730 !important;
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Align Configuration text with main title level */
    section[data-testid="stSidebar"] h2:first-of-type {
        margin-top: 0 !important;
        padding-top: 1rem !important;
        margin-bottom: 1rem !important;
    }

    /* Remove extra padding from sidebar container */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Quick Examples dropdown styling - more specific selectors */
    section[data-testid="stSidebar"] div[data-testid="stExpander"] .stSelectbox > div > div {
        background-color: #262730 !important;
        border: 1px solid #4a4a4a !important;
        border-radius: 4px !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stExpander"] .stSelectbox select {
        background-color: #262730 !important;
        color: #fafafa !important;
        border: none !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stExpander"] .stSelectbox div[data-baseweb="select"] {
        background-color: #262730 !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stExpander"] .stSelectbox div[data-baseweb="select"] > div {
        background-color: #262730 !important;
        color: #fafafa !important;
        border: 1px solid #4a4a4a !important;
    }

    /* Dropdown menu background - the actual options list */
    section[data-testid="stSidebar"] div[data-testid="stExpander"] .stSelectbox [data-baseweb="popover"] {
        background-color: #262730 !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stExpander"] .stSelectbox [data-baseweb="menu"] {
        background-color: #262730 !important;
        border: 1px solid #4a4a4a !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stExpander"] .stSelectbox [data-baseweb="menu"] > ul {
        background-color: #262730 !important;
    }

    /* Individual dropdown options */
    section[data-testid="stSidebar"] div[data-testid="stExpander"] .stSelectbox [role="option"] {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    /* Selection highlight styling - salmon/pink for hover and selected */
    section[data-testid="stSidebar"] div[data-testid="stExpander"] .stSelectbox [role="option"]:hover {
        background-color: #ff6b6b !important;
        color: white !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stExpander"] .stSelectbox [aria-selected="true"] {
        background-color: #ff6b6b !important;
        color: white !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stExpander"] .stSelectbox [role="option"][aria-selected="true"] {
        background-color: #ff6b6b !important;
        color: white !important;
    }

    /* Override any default Streamlit dropdown styling */
    section[data-testid="stSidebar"] .stSelectbox ul[role="listbox"] {
        background-color: #262730 !important;
        border: 1px solid #4a4a4a !important;
    }

    section[data-testid="stSidebar"] .stSelectbox li[role="option"] {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    section[data-testid="stSidebar"] .stSelectbox li[role="option"]:hover {
        background-color: #ff6b6b !important;
        color: white !important;
    }

    /* Aggressive dropdown targeting - multiple approaches */
    .stSelectbox > div > div[data-baseweb="select"] > div[data-baseweb="popover"] {
        background-color: #262730 !important;
    }

    .stSelectbox [data-baseweb="menu"] {
        background-color: #262730 !important;
        border: 1px solid #4a4a4a !important;
    }

    .stSelectbox [data-baseweb="menu"] li {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    .stSelectbox [data-baseweb="menu"] li:hover {
        background-color: #ff6b6b !important;
        color: white !important;
    }

    /* Target all possible dropdown containers */
    [data-baseweb="popover"] [data-baseweb="menu"] {
        background-color: #262730 !important;
    }

    [data-baseweb="popover"] [role="listbox"] {
        background-color: #262730 !important;
    }

    [data-baseweb="popover"] [role="option"] {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    [data-baseweb="popover"] [role="option"]:hover {
        background-color: #ff6b6b !important;
        color: white !important;
    }

    /* Universal dropdown override */
    div[data-testid="stSelectbox"] [data-baseweb="menu"] {
        background-color: #262730 !important;
    }

    div[data-testid="stSelectbox"] [role="option"] {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    /* Style export download buttons - red/pink/salmon color */
    .stDownloadButton > button {
        background-color: #ff6b6b !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }

    .stDownloadButton > button:hover {
        background-color: #ff5252 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 8px rgba(255, 107, 107, 0.3) !important;
    }

    .stDownloadButton > button:active {
        transform: translateY(0px) !important;
        background-color: #ff4444 !important;
    }

    /* Target specific download buttons by key if needed */
    div[data-testid="stDownloadButton"] button {
        background-color: #ff6b6b !important;
        color: white !important;
        border: 1px solid #ff6b6b !important;
    }

    div[data-testid="stDownloadButton"] button:hover {
        background-color: #ff5252 !important;
        border-color: #ff5252 !important;
    }

    /* Style main Download button with violet color */
    button[data-testid="download_button"] {
        background-color: #8b5cf6 !important;
        color: white !important;
        border: 1px solid #8b5cf6 !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }

    button[data-testid="download_button"]:hover {
        background-color: #7c3aed !important;
        border-color: #7c3aed !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 8px rgba(139, 92, 246, 0.3) !important;
    }

    button[data-testid="download_button"]:active {
        transform: translateY(0px) !important;
        background-color: #6d28d9 !important;
    }

    /* Alternative selector for download button */
    .stDownloadButton button[key="download_button"] {
        background-color: #8b5cf6 !important;
        color: white !important;
        border: 1px solid #8b5cf6 !important;
    }

    .stDownloadButton button[key="download_button"]:hover {
        background-color: #7c3aed !important;
        border-color: #7c3aed !important;
    }

    /* Style Export Results button with red/salmon color */
    button:contains("💾 Export Results") {
        background-color: #ff6b6b !important;
        color: white !important;
        border: 1px solid #ff6b6b !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
    }

    button:contains("💾 Export Results"):hover {
        background-color: #ff5252 !important;
        border-color: #ff5252 !important;
    }

    /* Alternative selector for Export Results button */
    .stButton:has(button:contains("💾 Export Results")) button {
        background-color: #ff6b6b !important;
        color: white !important;
        border: 1px solid #ff6b6b !important;
    }

    .stButton:has(button:contains("💾 Export Results")) button:hover {
        background-color: #ff5252 !important;
        border-color: #ff5252 !important;
    }

    /* Target first option specifically - multiple approaches */
    [data-baseweb="menu"] [role="option"]:first-child {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    [data-baseweb="menu"] li:first-child {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    .stSelectbox [role="option"]:first-of-type {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    /* Target any option with placeholder/default text */
    [data-baseweb="menu"] [role="option"][aria-selected="false"]:first-child {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    /* Force all dropdown children to have consistent styling */
    [data-baseweb="menu"] > * {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    [data-baseweb="menu"] * {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    /* Target any disabled or placeholder options */
    [role="option"][aria-disabled="true"] {
        background-color: #262730 !important;
        color: #888888 !important;
    }

    /* Ultra-aggressive first option targeting */
    .stSelectbox [data-baseweb="menu"] > div:first-child {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    .stSelectbox [data-baseweb="menu"] > div:first-child * {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    /* Target the placeholder text specifically */
    [data-baseweb="menu"] [role="option"]:nth-child(1) {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    [data-baseweb="menu"] div:nth-child(1) {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    /* Override any Streamlit default first option styling */
    .stSelectbox div[role="option"]:first-child,
    .stSelectbox li:first-child,
    .stSelectbox [data-baseweb="menu"] > *:first-child {
        background: #262730 !important;
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    /* Nuclear option - force everything in dropdown to match */
    .stSelectbox [data-baseweb="menu"] div,
    .stSelectbox [data-baseweb="menu"] span,
    .stSelectbox [data-baseweb="menu"] li,
    .stSelectbox [data-baseweb="menu"] p {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    /* Test multiple first option approaches */

    /* Test 1: Target placeholder specifically */
    .stSelectbox [data-baseweb="menu"] [data-value=""] {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    /* Test 2: Target any text containing "Select" */
    .stSelectbox [data-baseweb="menu"] div:first-child[role="option"] {
        background: #262730 !important;
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    /* Test 3: Ultra-specific first child targeting */
    .stSelectbox > div > div[data-baseweb="select"] [data-baseweb="menu"] > div:first-child {
        background: #262730 !important;
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    /* Test 4: Target by content/text */
    .stSelectbox [data-baseweb="menu"] div:contains("Select") {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    /* Test 5: Override with highest specificity */
    section[data-testid="stSidebar"] .stSelectbox > div > div[data-baseweb="select"] [data-baseweb="menu"] > div:first-child,
    section[data-testid="stSidebar"] .stSelectbox [data-baseweb="menu"] div:first-of-type {
        background: #262730 !important;
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    /* Test 6: Force all children with wildcard */
    .stSelectbox [data-baseweb="menu"] > *:first-child,
    .stSelectbox [data-baseweb="menu"] > *:first-child > *,
    .stSelectbox [data-baseweb="menu"] > *:first-child * {
        background: #262730 !important;
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    /* Fix the dropdown button/input display (closed state) */
    .stSelectbox > div > div[data-baseweb="select"] {
        background-color: #262730 !important;
    }

    .stSelectbox > div > div[data-baseweb="select"] > div {
        background-color: #262730 !important;
        color: #fafafa !important;
        border: 1px solid #4a4a4a !important;
    }

    /* Target the input/control part specifically */
    .stSelectbox [data-baseweb="select"] [data-baseweb="input"] {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    .stSelectbox [data-baseweb="select"] > div[role="button"] {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    /* Target the control wrapper */
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #262730 !important;
        color: #fafafa !important;
        border: 1px solid #4a4a4a !important;
    }



    /* Dark theme for text inputs */
    .stTextInput > div > div > input {
        background-color: #262730 !important;
        color: #fafafa !important;
        border: 1px solid #4a4a4a !important;
    }

    .stTextArea > div > div > textarea {
        background-color: #262730 !important;
        color: #fafafa !important;
        border: 1px solid #4a4a4a !important;
    }

    /* Dark theme for selectbox */
    .stSelectbox > div > div > select {
        background-color: #262730 !important;
        color: #fafafa !important;
        border: 1px solid #4a4a4a !important;
    }

    /* Dark theme for file uploader */
    .stFileUploader > div {
        background-color: #262730 !important;
        color: #fafafa !important;
        border: 1px solid #4a4a4a !important;
    }

    /* Dark theme for dataframes */
    .stDataFrame {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    /* Dark theme for metrics */
    .metric-container {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    /* Dark theme for expanders */
    .streamlit-expanderHeader {
        background-color: #262730 !important;
        color: #fafafa !important;
    }

    .streamlit-expanderContent {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }

    .entity-disease {
        background-color: #ff6b6b;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        margin: 2px;
        display: inline-block;
    }

    .entity-gene {
        background-color: #52c41a;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        margin: 2px;
        display: inline-block;
    }

    .entity-chemical {
        background-color: #45b7d1;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        margin: 2px;
        display: inline-block;
    }

    .entity-anatomy {
        background-color: #f9ca24;
        color: black;
        padding: 2px 6px;
        border-radius: 4px;
        margin: 2px;
        display: inline-block;
    }

    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        cursor: pointer;
    }

    .stButton > button:hover {
        background-color: #45a049;
    }

    /* Export button styling */
    div[data-testid="column"]:nth-child(2) .stButton > button {
        background-color: #ff6b6b !important;
        color: white !important;
    }

    div[data-testid="column"]:nth-child(2) .stButton > button:hover {
        background-color: #ff5252 !important;
    }

    /* Download link styling */
    .download-link {
        background-color: #4CAF50;
        color: white !important;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        text-decoration: none;
        display: inline-block;
        margin-left: 10px;
    }

    .download-link:hover {
        background-color: #45a049;
        text-decoration: none;
    }

    /* Reduce subheading font size by 20% and add spacing */
    h2 {
        font-size: 1.6rem !important;  /* 20% smaller than default ~2rem */
        margin-top: 2rem !important;    /* Add 20% more space above */
        margin-bottom: 1rem !important;
    }

    h3 {
        font-size: 1.28rem !important; /* 20% smaller than default ~1.6rem */
        margin-top: 1.8rem !important;  /* Add 20% more space above */
        margin-bottom: 0.8rem !important;
    }

    /* Reduce empty space above title */
    .block-container {
        padding-top: 2rem !important;  /* Reduced by 50% from default ~4rem */
    }

    /* Reduce sidebar top padding to align with main title */
    section[data-testid="stSidebar"] > div {
        padding-top: 0rem !important;  /* Remove all top padding */
        margin-top: -1rem !important;   /* Negative margin to move up further */
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 0rem !important;
    }

    /* Move sidebar content up to align with main title */
    section[data-testid="stSidebar"] > div:first-child {
        margin-top: 0 !important;
        padding-top: 2rem !important;  /* Match main title padding */
    }

    /* Reduce spacing between sidebar sections */
    section[data-testid="stSidebar"] .element-container {
        margin-bottom: 0.25rem !important;
        margin-top: 0.25rem !important;
    }

    /* Reduce spacing in sidebar headers */
    section[data-testid="stSidebar"] h2 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
        padding-top: 0 !important;
    }

    section[data-testid="stSidebar"] h3 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
        padding-top: 0 !important;
    }

    /* Reduce spacing for sidebar header specifically */
    section[data-testid="stSidebar"] [data-testid="stHeader"] {
        padding-top: 0.5rem !important;
    }

    /* Make buttons closer together (0.5 inch apart) */
    div[data-testid="column"]:has(.stButton) {
        padding-left: 0.1rem !important;
        padding-right: 0.1rem !important;
    }

    /* Specific styling for button columns to reduce gap */
    div[data-testid="column"]:nth-child(1) {
        padding-right: 0.25rem !important;
    }

    div[data-testid="column"]:nth-child(2) {
        padding-left: 0.25rem !important;
        padding-right: 0.25rem !important;
    }

    div[data-testid="column"]:nth-child(3) {
        padding-left: 0.25rem !important;
    }

    /* Section header highlighting - violet background with white text */
    span.section-header {
        background-color: #8b5cf6 !important;  /* Violet */
        color: #ffffff !important;  /* White */
        padding: 2px 6px !important;
        border-radius: 4px !important;
        font-weight: bold !important;
        border: 2px solid #7c3aed !important;  /* Darker violet border */
        display: inline !important;
        white-space: pre-wrap !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'validation_passed' not in st.session_state:
    st.session_state.validation_passed = False

@st.cache_resource
def initialize_predictor():
    """Initialize the enhanced medical NER predictor with caching."""
    if not PREDICTOR_AVAILABLE:
        return None

    try:
        with st.spinner("🔄 Loading Enhanced Medical NER Pipeline..."):
            predictor = EnhancedMedicalNERPredictor(
                model_name="en_core_web_sm",
                use_gpu=False,
                batch_size=10
            )
            return predictor
    except Exception as e:
        st.error(f"❌ Failed to initialize predictor: {e}")
        return None

def validate_predictor_performance(predictor):
    """Run validation tests on the predictor."""
    if predictor is None:
        return False

    try:
        with st.spinner("🧪 Running pipeline validation tests..."):
            return validate_pipeline_performance(predictor)
    except Exception as e:
        st.error(f"❌ Validation failed: {e}")
        return False

def clean_text(text: str) -> str:
    """Remove HTML tags and clean text for display."""
    import re
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', '', text)
    # Remove extra whitespace
    clean = re.sub(r'\s+', ' ', clean)
    # Strip leading/trailing whitespace
    clean = clean.strip()
    return clean

def format_context_sentences(context_text: str) -> str:
    """Format context sentences with line breaks and convert text markers to HTML highlighting."""
    if not context_text:
        return ""

    # Define entity type colors (same as main visualization)
    entity_colors = {
        'DISEASE': '#ff6b6b',
        'GENE': '#52c41a',
        'DRUG': '#45b7d1',
        'CHEMICAL': '#45b7d1',
        'ANATOMY': '#f9ca24'
    }

    # Convert text markers ▶[entity]◀ to colored HTML
    import re

    # Extract entity type from format: [PREDICTOR | entity] (TYPE) ...
    def convert_line(line):
        # Extract entity type from parentheses
        type_match = re.search(r'\(([A-Z]+)\)', line)
        entity_type = type_match.group(1) if type_match else 'DISEASE'
        color = entity_colors.get(entity_type, '#888888')

        # Convert predictor markers ⟨pattern⟩ to italicized brackets
        line = re.sub(
            r'⟨([^⟩]+)⟩',
            lambda m: f'<i>[{m.group(1)}]</i>',
            line
        )

        # Convert text markers ▶[text]◀ to colored badges
        line = re.sub(
            r'▶\[([^\]]+)\]◀',
            lambda m: f'<span style="background-color: {color}; color: white; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{m.group(1)}</span>',
            line
        )

        # Color the entity in [PREDICTOR | entity] label - keep at 18px
        line = re.sub(
            r'\[([A-Z\s]+)\s+\|\s+([^\]]+)\]',
            lambda m: f'<span style="font-size: 18px;">[{m.group(1)} | <span style="color: {color}; font-weight: bold;">{m.group(2)}</span>]</span>',
            line
        )

        # Wrap rest of context in 14px
        return f'<span style="font-size: 14px;">{line}</span>'

    # Process each line
    lines = context_text.split('\n\n')
    formatted_lines = [convert_line(line) for line in lines]

    # Replace double newlines with <br><br> for blank line separation
    return '<br><br>'.join(formatted_lines)

def _find_predictor_for_entity(text: str, entity_text: str, entity_start: int, patterns: List[str]) -> str:
    """Find which predictor pattern triggered the classification for a specific entity with word boundaries."""
    if not patterns:
        return ""

    # Search ±200 char window around entity (matches predictor detection logic)
    start_idx = max(0, entity_start - 200)
    end_idx = min(len(text), entity_start + len(entity_text) + 200)
    context = text[start_idx:end_idx].lower()

    # Use word boundary matching to avoid false positives
    import re
    for pattern in patterns:
        pattern_lower = pattern.lower()
        # Multi-word patterns use simple substring
        if ' ' in pattern_lower:
            if pattern_lower in context:
                return pattern.upper()
        else:
            # Single-word patterns use word boundaries
            regex = r'\b' + re.escape(pattern_lower) + r'\b'
            if re.search(regex, context):
                return pattern.upper()

    return ""

def _adjust_to_word_boundaries(text: str, start: int, end: int) -> tuple:
    """
    Adjust entity boundaries to word boundaries to prevent cutting words mid-way.

    Args:
        text: The full text
        start: Entity start position
        end: Entity end position

    Returns:
        Tuple of (adjusted_start, adjusted_end)
    """
    # Ensure bounds are within text limits
    start = max(0, min(start, len(text)))
    end = max(0, min(end, len(text)))

    # Expand start backward to word boundary
    adjusted_start = start
    while adjusted_start > 0 and adjusted_start - 1 < len(text) and not text[adjusted_start - 1].isspace() and text[adjusted_start - 1] not in '.,;:!?()[]{}':
        adjusted_start -= 1

    # Expand end forward to word boundary
    adjusted_end = end
    while adjusted_end < len(text) and adjusted_end < len(text) and not text[adjusted_end].isspace() and text[adjusted_end] not in '.,;:!?()[]{}':
        adjusted_end += 1

    return adjusted_start, adjusted_end

def format_entities_html(text: str, entities: List[Dict], context_entities: Dict[str, Any] = None) -> str:
    """Format text with colored entity highlighting, context badges, and section headers.
    Entity boundaries are adjusted to word boundaries to prevent cutting words."""
    if not entities:
        return clean_text(text)

    # Clean the text first
    original_text = clean_text(text)

    # Build context mapping from entity text to context types
    context_map = {}
    context_patterns = {
        'confirmed': [],
        'negated': [],
        'uncertain': [],
        'historical': [],
        'family': []
    }

    if context_entities:
        # Extract pattern lists from predictor strings
        confirmed_predictors = context_entities.get('confirmed_entities_predictors', '')
        context_patterns['confirmed'] = [p.strip() for p in confirmed_predictors.split(',') if p.strip()]

        negated_predictors = context_entities.get('negated_entities_predictors', '')
        context_patterns['negated'] = [p.strip() for p in negated_predictors.split(',') if p.strip()]

        uncertain_predictors = context_entities.get('uncertain_entities_predictors', '')
        context_patterns['uncertain'] = [p.strip() for p in uncertain_predictors.split(',') if p.strip()]

        historical_predictors = context_entities.get('historical_entities_predictors', '')
        context_patterns['historical'] = [p.strip() for p in historical_predictors.split(',') if p.strip()]

        family_predictors = context_entities.get('family_entities_predictors', '')
        context_patterns['family'] = [p.strip() for p in family_predictors.split(',') if p.strip()]

        # Parse confirmed entities
        confirmed = context_entities.get('confirmed_entities', '').split(';')
        for ent in confirmed:
            ent = ent.strip()
            if ent and ent != 'NOT':
                context_map[ent.lower()] = context_map.get(ent.lower(), []) + ['confirmed']

        # Parse negated entities
        negated = context_entities.get('negated_entities', '').split(';')
        for ent in negated:
            ent = ent.strip().replace('NOT ', '')
            if ent:
                context_map[ent.lower()] = context_map.get(ent.lower(), []) + ['negated']

        # Parse uncertain entities
        uncertain = context_entities.get('uncertain_entities', '').split(';')
        for ent in uncertain:
            ent = ent.strip()
            if ent:
                context_map[ent.lower()] = context_map.get(ent.lower(), []) + ['uncertain']

        # Parse historical entities
        historical = context_entities.get('historical_entities', '').split(';')
        for ent in historical:
            ent = ent.strip()
            if ent:
                context_map[ent.lower()] = context_map.get(ent.lower(), []) + ['historical']

        # Parse family entities
        family = context_entities.get('family_entities', '').split(';')
        for ent in family:
            ent = ent.strip()
            if ent:
                context_map[ent.lower()] = context_map.get(ent.lower(), []) + ['family']

    # Define colors for different entity types
    color_map = {
        'DISEASE': '#ff6b6b',
        'GENE': '#52c41a',
        'DRUG': '#45b7d1',
        'CHEMICAL': '#45b7d1',
        'ANATOMY': '#f9ca24'
    }

    # Priority order for context classifications
    priority_order = ['negated', 'confirmed', 'uncertain', 'historical', 'family']
    context_icons = {
        'confirmed': '✅',
        'negated': '❌',
        'uncertain': '❓',
        'historical': '📅',
        'family': '👨‍👩‍👧'
    }

    # STEP 1: Remove overlapping entities (keep longer ones)
    sorted_by_start = sorted(entities, key=lambda x: x.get('start', 0))
    filtered_entities = []

    for entity in sorted_by_start:
        should_keep = True
        entity_start = entity.get('start', 0)
        entity_end = entity.get('end', 0)
        entity_len = entity_end - entity_start

        # Check against already kept entities
        for i, kept in enumerate(filtered_entities):
            # Skip None entries (already marked for removal)
            if kept is None:
                continue

            kept_start = kept.get('start', 0)
            kept_end = kept.get('end', 0)
            kept_len = kept_end - kept_start

            # Check for overlap
            if not (entity_end <= kept_start or entity_start >= kept_end):
                # There's an overlap - keep the longer entity
                if entity_len > kept_len:
                    # Current entity is longer, remove the kept one
                    filtered_entities[i] = None  # Mark for removal
                else:
                    # Kept entity is longer or equal, skip current
                    should_keep = False
                    break

        if should_keep:
            filtered_entities.append(entity)

    # Remove None entries (marked for removal)
    filtered_entities = [e for e in filtered_entities if e is not None]

    # STEP 2: Build formatted text from segments (no position drift!)
    segments = []
    last_position = 0

    for entity in sorted(filtered_entities, key=lambda x: x.get('start', 0)):
        start = entity.get('start', 0)
        end = entity.get('end', 0)
        label = entity.get('label', 'UNKNOWN')
        original_entity_text = entity.get('text', '')  # Keep original for context lookup

        # Adjust boundaries to word boundaries to prevent cutting words
        adjusted_start, adjusted_end = _adjust_to_word_boundaries(original_text, start, end)

        # Ensure bounds are within text limits before accessing
        adjusted_start = max(0, min(adjusted_start, len(original_text)))
        adjusted_end = max(0, min(adjusted_end, len(original_text)))

        # Get display text from adjusted boundaries
        display_text = original_text[adjusted_start:adjusted_end] if adjusted_end > adjusted_start else original_entity_text

        # Use adjusted boundaries for segment building
        start, end = adjusted_start, adjusted_end

        # Add plain text before this entity
        if start > last_position and last_position < len(original_text):
            safe_end = min(start, len(original_text))
            segments.append(original_text[last_position:safe_end])

        # Get color for this entity type
        color = color_map.get(label, '#95a5a6')

        # Check for context classifications using ORIGINAL entity text
        contexts = context_map.get(original_entity_text.lower(), [])
        context_badges = ''

        if contexts:
            # Show ALL applicable context badges instead of just highest priority
            badge_spans = []
            for priority_ctx in priority_order:
                if priority_ctx in contexts:
                    icon = context_icons.get(priority_ctx, '')

                    # Find the SPECIFIC predictor for THIS entity and context
                    predictor = _find_predictor_for_entity(
                        original_text,
                        original_entity_text,
                        start,
                        context_patterns.get(priority_ctx, [])
                    )

                    # Create tooltip with specific reason
                    if predictor:
                        tooltip = f"{priority_ctx.capitalize()} (Reason: {predictor})"
                    else:
                        tooltip = priority_ctx.capitalize()

                    if icon:
                        badge_spans.append(f'<span style="margin-left: 2px; font-size: 0.85em;" title="{tooltip}">{icon}</span>')

            # Combine all badges
            context_badges = ''.join(badge_spans)

        # Create HTML span with entity highlighting (improved styling for better flow)
        entity_html = f'<span class="entity-{label.lower()}" style="background-color: {color}; color: white; padding: 1px 4px; border-radius: 3px; white-space: pre-wrap; display: inline;">{display_text} ({label}){context_badges}</span>'

        segments.append(entity_html)
        last_position = end

    # Add remaining text after last entity
    if last_position < len(original_text):
        safe_start = min(last_position, len(original_text))
        segments.append(original_text[safe_start:])

    # Join all segments to create the formatted text
    formatted_text = ''.join(segments)

    # NOW highlight section headers AFTER entities (only detected sections)
    section_categories = context_entities.get('section_categories', '') if context_entities else ''
    if section_categories and section_categories != 'General Clinical Text':
        import re

        # Get list of sections to highlight (only those detected)
        # Handle both comma and semicolon separators
        if ';' in section_categories:
            detected_sections = [s.strip() for s in section_categories.split(';')]
        else:
            detected_sections = [s.strip() for s in section_categories.split(',')]

        # Map detected sections to common variations to catch different formats
        section_variations = {
            'Chief Complaint': ['Chief Complaint', 'CC', 'chief complaint'],
            'History of Present Illness': ['History of Present Illness', 'HPI', 'Present Illness', 'history of present illness'],
            'Past Medical History': ['Past Medical History', 'PMH', 'Medical History', 'past medical history'],
            'Family History': ['Family History', 'FH', 'FHx', 'family history'],
            'Social History': ['Social History', 'SH', 'social history'],
            'Review of Systems': ['Review of Systems', 'ROS', 'review of systems'],
            'Physical Examination': ['Physical Examination', 'PE', 'Physical Exam', 'Exam', 'physical examination'],
            'Assessment': ['Assessment', 'assessment'],
            'Plan': ['Plan', 'plan', 'treatment', 'therapy', 'management'],
            'Assessment and Plan': ['Assessment and Plan', 'A/P', 'A&P', 'assessment and plan'],
            'Subjective': ['Subjective', 'subjective'],
            'Objective': ['Objective', 'objective'],
            'Summary': ['Summary', 'summary', 'Clinical Summary'],
            'Introduction': ['Introduction', 'introduction', 'Background', 'Overview'],
            'Medications': ['Medications', 'Meds', 'medications'],
            'Allergies': ['Allergies', 'allergies'],
            'Vital Signs': ['Vital Signs', 'Vitals', 'vital signs'],
            'Laboratory': ['Laboratory', 'Labs', 'Lab Results', 'laboratory'],
            'Imaging': ['Imaging', 'Radiology', 'imaging'],
            'Diagnosis': ['Diagnosis', 'Dx', 'diagnosis'],
            'Treatment': ['Treatment', 'Tx', 'treatment'],
            'Follow-up': ['Follow-up', 'Follow up', 'F/U', 'follow-up']
        }

        # Collect all variations for detected sections - use only exact detected names
        sections_to_highlight = set(detected_sections)

        # Split formatted text into HTML and plain text parts
        # Replace section headers only in plain text (not inside <span> tags)
        if sections_to_highlight:
            parts = re.split(r'(<span[^>]*>.*?</span>)', formatted_text, flags=re.DOTALL)

            for i, part in enumerate(parts):
                # Only process non-HTML parts
                if not part.startswith('<span'):
                    # Highlight ONLY detected section headers in this plain text part
                    for section in sections_to_highlight:
                        # Create pattern - match word with optional colon/period
                        pattern = re.compile(rf'\b({re.escape(section)})\s*[:,.]?', re.IGNORECASE)

                        def make_section_html(match):
                            matched_text = match.group(0)  # Get full match including punctuation
                            return f'<span class="section-header" style="background-color: #8b5cf6 !important; color: #ffffff !important; padding: 2px 6px !important; border-radius: 4px !important; font-weight: bold !important; border: 2px solid #7c3aed !important; display: inline !important; white-space: pre-wrap !important;">📋 {matched_text}</span>'

                        parts[i] = pattern.sub(make_section_html, parts[i])

            formatted_text = ''.join(parts)

    return formatted_text

def process_text(text: str, predictor) -> Dict[str, Any]:
    """Process text using the enhanced medical NER predictor."""
    if predictor is None:
        return {}

    try:
        result = predictor.extract_entities(text)

        # Parse entities from JSON
        entities_json = result.get('all_entities_json', '[]')
        entities = json.loads(entities_json) if entities_json else []

        # Count entities by type
        entity_counts = {
            'diseases': len([e for e in entities if e.get('label') == 'DISEASE']),
            'genes': len([e for e in entities if e.get('label') == 'GENE']),
            'chemicals': len([e for e in entities if e.get('label') in ['DRUG', 'CHEMICAL']]),
            'total': len(entities),
            # Context entity counts
            'confirmed': result.get('confirmed_entities_count', 0),
            'negated': result.get('negated_entities_count', 0),
            'uncertain': result.get('uncertain_entities_count', 0),
            'historical': result.get('historical_entities_count', 0),
            'family': result.get('family_entities_count', 0)
        }

        # Use the predictor's built-in visualization which handles entity positioning correctly
        built_in_visualization = result.get('Text Visualization', '')

        # Check if built-in visualization has highlighting, otherwise use custom formatter
        has_highlighting = built_in_visualization and '<span' in built_in_visualization

        return {
            'entities': entities,
            'entity_counts': entity_counts,
            'result': result,
            'formatted_text': built_in_visualization if has_highlighting else format_entities_html(text, entities, result)
        }
    except Exception as e:
        st.error(f"❌ Error processing text: {e}")
        return {}

def process_multiple_texts(texts: List[str], predictor) -> List[Dict[str, Any]]:
    """Process multiple texts and return results for each."""
    results = []
    for text in texts:
        result = process_text(text, predictor)
        results.append(result)
    return results

def export_entire_dataframe(df: pd.DataFrame, predictor) -> str:
    """Export entire DataFrame with NER predictions using the enhanced pipeline's predict_dataframe method."""
    from pathlib import Path

    # Create output directory if it doesn't exist
    output_dir = Path('output/results')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'output_results_{timestamp}.xlsx'
    filepath = output_dir / filename

    # Ensure the DataFrame has the required columns for the pipeline
    if 'Index' not in df.columns:
        df = df.reset_index()
        df.rename(columns={'index': 'Index'}, inplace=True)

    if 'Text' not in df.columns:
        raise ValueError("DataFrame must have a 'Text' column for NER processing")

    # Use the enhanced NER pipeline's predict_dataframe method directly
    # This preserves all original columns and adds all prediction columns
    df_predicted = predictor.predict_dataframe(df, text_column='Text')

    # Use the same save_formatted_excel function as the enhanced NER pipeline
    save_formatted_excel(df_predicted, filepath)

    return str(filepath)

def export_results_to_file(texts: List[str], results: List[Dict[str, Any]], file_format: str = 'xlsx') -> str:
    """Export results to Excel file - fast version without complex formatting."""
    from pathlib import Path

    # Create output directory if it doesn't exist
    output_dir = Path('output/exports')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'output_results_{timestamp}.xlsx'
    filepath = output_dir / filename

    # Prepare data for export in the exact same format as the enhanced NER pipeline
    export_data = []
    for idx, (text, result) in enumerate(zip(texts, results)):
        result_dict = result.get('result', {})

        # Create row data with exact same columns as enhanced NER pipeline
        row_data = {
            'Index': idx,
            'Text': clean_text(text),
            'Text_Clean': clean_text(text),
            'Text Visualization': result_dict.get('Text Visualization', ''),
            'detected_diseases': result_dict.get('detected_diseases', ''),
            'total_diseases_count': result_dict.get('total_diseases_count', 0),
            'detected_genes': result_dict.get('detected_genes', ''),
            'total_gene_count': result_dict.get('total_gene_count', 0),
            'detected_chemicals': result_dict.get('detected_chemicals', ''),
            'total_chemicals_count': result_dict.get('total_chemicals_count', 0),
            'confirmed_entities': result_dict.get('confirmed_entities', ''),
            'confirmed_entities_count': result_dict.get('confirmed_entities_count', 0),
            'confirmed_context_sentences': result_dict.get('confirmed_context_sentences', ''),
            'negated_entities': result_dict.get('negated_entities', ''),
            'negated_entities_count': result_dict.get('negated_entities_count', 0),
            'negated_context_sentences': result_dict.get('negated_context_sentences', ''),
            'historical_entities': result_dict.get('historical_entities', ''),
            'historical_entities_count': result_dict.get('historical_entities_count', 0),
            'historical_context_sentences': result_dict.get('historical_context_sentences', ''),
            'uncertain_entities': result_dict.get('uncertain_entities', ''),
            'uncertain_entities_count': result_dict.get('uncertain_entities_count', 0),
            'uncertain_context_sentences': result_dict.get('uncertain_context_sentences', ''),
            'family_entities': result_dict.get('family_entities', ''),
            'family_entities_count': result_dict.get('family_entities_count', 0),
            'family_context_sentences': result_dict.get('family_context_sentences', ''),
            'section_categories': result_dict.get('section_categories', ''),
            'all_entities_json': result_dict.get('all_entities_json', '[]')
        }
        export_data.append(row_data)

    # Create DataFrame and save directly with pandas
    df = pd.DataFrame(export_data)

    # Fast save with pandas ExcelWriter
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='NER_Results', index=False)

    return str(filepath)

def export_text_results_fast(texts: List[str], results: List[Dict[str, Any]]) -> str:
    """Export text results with enhanced NER pipeline format to output/exports/ directory."""
    from pathlib import Path

    # Create output/exports directory if it doesn't exist
    output_dir = Path('output/exports')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'output_results_{timestamp}.xlsx'
    filepath = output_dir / filename

    # Prepare data in enhanced NER pipeline format
    export_data = []
    for idx, (text, result) in enumerate(zip(texts, results)):
        result_dict = result.get('result', {})

        # Create row data with exact same columns as enhanced NER pipeline
        row_data = {
            'Index': idx,
            'Text': clean_text(text),
            'Text_Clean': clean_text(text),
            'Text Visualization': result_dict.get('Text Visualization', ''),
            'detected_diseases': result_dict.get('detected_diseases', ''),
            'total_diseases_count': result_dict.get('total_diseases_count', 0),
            'detected_genes': result_dict.get('detected_genes', ''),
            'total_gene_count': result_dict.get('total_gene_count', 0),
            'detected_chemicals': result_dict.get('detected_chemicals', ''),
            'total_chemicals_count': result_dict.get('total_chemicals_count', 0),
            'confirmed_entities': result_dict.get('confirmed_entities', ''),
            'confirmed_entities_count': result_dict.get('confirmed_entities_count', 0),
            'confirmed_context_sentences': result_dict.get('confirmed_context_sentences', ''),
            'negated_entities': result_dict.get('negated_entities', ''),
            'negated_entities_count': result_dict.get('negated_entities_count', 0),
            'negated_context_sentences': result_dict.get('negated_context_sentences', ''),
            'historical_entities': result_dict.get('historical_entities', ''),
            'historical_entities_count': result_dict.get('historical_entities_count', 0),
            'historical_context_sentences': result_dict.get('historical_context_sentences', ''),
            'uncertain_entities': result_dict.get('uncertain_entities', ''),
            'uncertain_entities_count': result_dict.get('uncertain_entities_count', 0),
            'uncertain_context_sentences': result_dict.get('uncertain_context_sentences', ''),
            'family_entities': result_dict.get('family_entities', ''),
            'family_entities_count': result_dict.get('family_entities_count', 0),
            'family_context_sentences': result_dict.get('family_context_sentences', ''),
            'section_categories': result_dict.get('section_categories', ''),
            'all_entities_json': result_dict.get('all_entities_json', '[]')
        }
        export_data.append(row_data)

    # Create DataFrame with exact same columns as enhanced NER pipeline
    df = pd.DataFrame(export_data)

    # Save with formatting (text wrap, row heights, frozen header, filters, green header)
    try:
        import xlsxwriter

        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            # Write dataframe to Excel
            df.to_excel(writer, sheet_name='NER_Results', index=False, startrow=0, startcol=0)

            # Get workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['NER_Results']

            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'vcenter',
                'align': 'left',  # Left alignment for header
                'fg_color': '#90EE90',  # Light green
                'border': 1
            })

            cell_format = workbook.add_format({
                'text_wrap': True,
                'valign': 'top',
                'border': 1
            })

            # Calculate dynamic column widths (max 25, but adjust to content if smaller)
            for col_num, column_name in enumerate(df.columns):
                # Get column data
                column_data = df.iloc[:, col_num]

                # Calculate max length considering header and data
                max_len = len(str(column_name))  # Header length

                # Check first 100 rows for content width (for performance)
                for value in column_data.head(100):
                    if pd.notna(value):
                        # For long text, estimate based on first 100 chars
                        value_str = str(value)[:100]
                        max_len = max(max_len, len(value_str))

                # Set width: minimum 10, maximum 25, with padding
                column_width = min(25, max(10, max_len + 2))
                worksheet.set_column(col_num, col_num, column_width, cell_format)

            # Format header row (row 0)
            for col_num, column_name in enumerate(df.columns):
                worksheet.write(0, col_num, column_name, header_format)

            # Set header row height to 20 points
            worksheet.set_row(0, 20)

            # Set row height to 15 points for all data rows
            for row_num in range(1, len(df) + 1):
                worksheet.set_row(row_num, 15)

            # Freeze top row
            worksheet.freeze_panes(1, 0)

            # Add autofilter to the entire data range
            worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)

    except ImportError:
        # Fallback to basic export if xlsxwriter not available
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='NER_Results', index=False)

    return str(filepath)

def main():
    """Main Streamlit application."""

    # Main title
    st.title("🧬 Enhanced Bio-NER Entity Visualizer")

    # Description
    st.markdown("""
    ### A powerful biomedical named entity recognition tool using BioBERT models for:
    - **🩺 Diseases** - Medical conditions and pathologies
    - **🧬 Genes** - Genetic entities and biomarkers
    - **💊 Chemicals/Drugs** - Pharmaceutical compounds and chemicals
    - **🔍 Context Analysis** - Historical, negated, uncertain, and family history entities
    """)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Initialize predictor
    if st.session_state.predictor is None:
        st.session_state.predictor = initialize_predictor()

        # Auto-run validation on first load if predictor is available
        if st.session_state.predictor is not None and not st.session_state.validation_passed:
            with st.spinner("🧪 Running automatic pipeline validation..."):
                try:
                    validation_result = validate_predictor_performance(st.session_state.predictor)
                    st.session_state.validation_passed = validation_result
                except Exception as e:
                    st.warning(f"⚠️ Auto-validation skipped: {e}")
                    st.session_state.validation_passed = False

    # Validation section
    st.sidebar.subheader("🧪 Pipeline Validation")

    # Show validation status
    if st.session_state.validation_passed:
        st.sidebar.success("✅ Pipeline validated")
    else:
        st.sidebar.warning("⚠️ Pipeline not validated")

    # Manual validation button
    if st.sidebar.button("🔬 Run Validation Tests"):
        if st.session_state.predictor:
            validation_result = validate_predictor_performance(st.session_state.predictor)
            st.session_state.validation_passed = validation_result
            if validation_result:
                st.sidebar.success("✅ All validation tests passed!")
            else:
                st.sidebar.error("❌ Some validation tests failed!")
        else:
            st.sidebar.error("❌ Predictor not available")

    # Model configuration
    st.sidebar.subheader("🤖 Model Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)
    show_raw_results = st.sidebar.checkbox("Show Raw Results", False)

    # Loaded Models (expandable dropdown in sidebar with compact list display)
    with st.sidebar.expander("📦 Loaded Models", expanded=False):
        if st.session_state.predictor:
            st.markdown("""
            **Loaded Models:**
            - ✅ spaCy NLP Pipeline
            - ✅ BioBERT Chemical Model
            - ✅ BioBERT Disease Model
            - ✅ BioBERT Gene Model
            """)

            # Template information in compact list format
            st.markdown("**📋 Enhanced Rules:**")
            predictor = st.session_state.predictor

            rules_list = []
            if hasattr(predictor, 'historical_rules_enabled') and predictor.historical_rules_enabled:
                rules_list.append(f"- 📜 Historical: {len(predictor.historical_patterns)} patterns")

            if hasattr(predictor, 'negated_rules_enabled') and predictor.negated_rules_enabled:
                rules_list.append(f"- 🚫 Negated: {len(predictor.negated_patterns)} patterns")

            if hasattr(predictor, 'uncertainty_rules_enabled') and predictor.uncertainty_rules_enabled:
                rules_list.append(f"- ❓ Uncertainty: {len(predictor.uncertainty_patterns)} patterns")

            if rules_list:
                st.markdown("\n".join(rules_list))
        else:
            st.error("❌ Predictor not available")

    # Example Texts (dropdown in sidebar)
    st.sidebar.subheader("💡 Quick Examples")

    # Define example options first
    example_options = {
        "Select an example...": "",
        "Hypertension & Medication": "Patient diagnosed with hypertension and prescribed lisinopril.",
        "Genetic Mutation": "BRCA1 mutation associated with breast cancer risk.",
        "Diabetes Treatment": "Diabetes mellitus treated with metformin and insulin.",
        "Family History": "Family history of cardiovascular disease and stroke.",
        "Symptoms & Negation": "Patient denies chest pain but reports dyspnea."
    }

    # Initialize session state for text input with first real example
    if 'input_text' not in st.session_state:
        st.session_state.input_text = example_options["Hypertension & Medication"]

    # Initialize selected example in session state
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = "Hypertension & Medication"

    selected_example = st.sidebar.selectbox(
        "Choose an example:",
        options=list(example_options.keys()),
        index=list(example_options.keys()).index(st.session_state.selected_example),
        key="example_selector"
    )

    # Update text when example is selected
    if selected_example != "Select an example..." and example_options[selected_example]:
        st.session_state.input_text = example_options[selected_example]
        st.session_state.selected_example = selected_example

    # Entity type legend in sidebar
    st.sidebar.subheader("🏷️ Entity Types")
    st.sidebar.markdown("""
    <div style="margin: 10px 0;">
        <span class="entity-disease">DISEASE</span> Medical conditions<br>
        <span class="entity-gene">GENE</span> Genetic entities<br>
        <span class="entity-chemical">CHEMICAL</span> Drugs & compounds<br>
        <span class="entity-anatomy">ANATOMY</span> Body parts<br>
        <span style="background-color: #8b5cf6; color: #ffffff; padding: 2px 6px; border-radius: 4px; margin: 2px; font-weight: bold; border: 2px solid #7c3aed;">📋 SECTION</span> Section headers<br>
    </div>
    """, unsafe_allow_html=True)

    # Main content area (full width - no columns)
    st.header("📝 Text Input")

    # Input method selection
    input_method = st.radio("Choose input method:", ["Text Input", "File Upload"])

    if input_method == "Text Input":
        # Clear file upload session state when switching to text input mode
        if 'excel_mode' in st.session_state:
            st.session_state.excel_mode = None
        if 'excel_texts' in st.session_state:
            st.session_state.excel_texts = []
        if 'uploaded_df' in st.session_state:
            st.session_state.uploaded_df = None

        text_input = st.text_area(
            "Enter medical text for analysis:",
            value=st.session_state.input_text,
            height=150,
            placeholder="Enter medical text here...",
            key="text_input_area"
        )
        # Update session state with current text
        st.session_state.input_text = text_input
    else:
        # Clear text input session state when switching to file upload mode
        if 'input_text' in st.session_state:
            st.session_state.input_text = ""

        uploaded_file = st.file_uploader(
            "Upload a text or Excel file (.txt, .xlsx, .xls)",
            type=['txt', 'xlsx', 'xls'],
            help="Upload a plain text file or Excel file containing medical text"
        )

        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1].lower()

            if file_extension == 'txt':
                # Handle text files
                text_input = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", clean_text(text_input), height=300)
            elif file_extension in ['xlsx', 'xls']:
                # Handle Excel files
                try:
                    df = pd.read_excel(uploaded_file)
                    st.success(f"✅ Loaded Excel file with {len(df)} rows and {len(df.columns)} columns")

                    # Store the uploaded DataFrame in session state for export
                    st.session_state.uploaded_df = df.copy()

                    # Show available columns
                    st.write("**Available columns:**", ", ".join(df.columns.tolist()))

                    # Function to validate if a column contains text data
                    def is_text_column(df, column_name):
                        """Check if a column contains text data suitable for NER analysis."""
                        column = df[column_name]

                        # Check if column has any non-null values
                        if column.isnull().all():
                            return False, "Column contains only null values"

                        # Get sample of non-null values
                        sample_values = column.dropna().head(10)

                        # Check if values are strings or can be converted to strings meaningfully
                        text_count = 0
                        for value in sample_values:
                            if isinstance(value, str) and len(str(value).strip()) > 0:
                                text_count += 1
                            elif pd.api.types.is_numeric_dtype(type(value)) and len(str(value)) < 50:
                                # Probably not meaningful text (just numbers)
                                continue
                            elif not isinstance(value, str) and len(str(value).strip()) > 5:
                                # Non-string but has meaningful length
                                text_count += 1

                        # At least 70% should be meaningful text
                        if text_count / len(sample_values) >= 0.7:
                            return True, "Valid text column"
                        else:
                            return False, "Column does not contain sufficient text data"

                    # Always show column selector for better user control
                    st.subheader("📋 Column Selection")
                    available_columns = df.columns.tolist()

                    # Default selection logic
                    default_index = 0
                    if 'Text' in available_columns:
                        default_index = available_columns.index('Text')
                    elif 'text' in available_columns:
                        default_index = available_columns.index('text')
                    elif any('text' in col.lower() for col in available_columns):
                        # Find first column with 'text' in name
                        for i, col in enumerate(available_columns):
                            if 'text' in col.lower():
                                default_index = i
                                break

                    selected_column = st.selectbox(
                        "Select the column containing text for NER analysis:",
                        options=available_columns,
                        index=default_index,
                        help="Choose the column that contains the medical text you want to analyze"
                    )

                    # Validate the selected column
                    if selected_column:
                        is_valid, validation_message = is_text_column(df, selected_column)

                        if is_valid:
                            # Create a copy of the DataFrame and ensure the selected column is mapped to 'Text'
                            df_for_processing = df.copy()
                            if selected_column != 'Text':
                                df_for_processing['Text'] = df_for_processing[selected_column]

                            st.session_state.uploaded_df = df_for_processing
                            st.session_state.selected_text_column = selected_column

                            st.success(f"✅ Column '{selected_column}' selected for NER analysis")

                            # Show preview of the selected column
                            preview_data = df[selected_column].dropna().head(3)
                            if len(preview_data) > 0:
                                st.write("**Preview of selected text column:**")
                                for i, text in enumerate(preview_data):
                                    preview_text = str(text)[:200] + "..." if len(str(text)) > 200 else str(text)
                                    st.text_area(f"Row {i+1}:", preview_text, height=60, disabled=True)
                        else:
                            st.error(f"❌ Please select a column with text. {validation_message}")
                            st.session_state.uploaded_df = None
                            st.session_state.selected_text_column = None
                    else:
                        selected_column = 'Text'
                        st.info("✅ Using existing 'Text' column for NER processing")

                    # Let user select which row to analyze
                    row_options = ["Single row", "All rows (separately)", "All rows (combined)", "First 5 rows", "First 10 rows"]
                    analysis_mode = st.radio(
                        "Analysis mode:",
                        row_options,
                        horizontal=True
                    )

                    # Initialize session state for Excel data
                    if 'excel_texts' not in st.session_state:
                        st.session_state.excel_texts = []
                    if 'excel_mode' not in st.session_state:
                        st.session_state.excel_mode = 'single'

                    if analysis_mode == "Single row":
                        row_index = st.number_input(
                            "Select row number:",
                            min_value=0,
                            max_value=len(df)-1,
                            value=0
                        )
                        text_input = str(df.iloc[row_index][selected_column])
                        st.text_area("Selected text:", clean_text(text_input), height=300)
                        st.session_state.excel_mode = 'single'
                        st.session_state.excel_texts = []
                    elif analysis_mode == "All rows (separately)":
                        # Store all texts separately for individual analysis
                        st.session_state.excel_texts = df[selected_column].astype(str).tolist()
                        st.session_state.excel_mode = 'separate'
                        text_input = ""  # Will process separately
                        st.info(f"📊 Will analyze {len(df)} rows separately with individual visualizations")
                        preview_text = clean_text(st.session_state.excel_texts[0])
                        st.text_area("Preview (first row):", preview_text, height=200)
                    elif analysis_mode == "All rows (combined)":
                        # Combine all rows from selected column
                        text_input = "\n\n".join(df[selected_column].astype(str).tolist())
                        preview_length = min(500, len(text_input))
                        preview_text = clean_text(text_input)[:preview_length]
                        st.text_area("Combined text preview:", preview_text + ("..." if len(text_input) > preview_length else ""), height=300)
                        st.info(f"📊 Will analyze {len(df)} rows combined")
                        st.session_state.excel_mode = 'single'
                        st.session_state.excel_texts = []
                    elif analysis_mode == "First 5 rows":
                        st.session_state.excel_texts = df.head(5)[selected_column].astype(str).tolist()
                        st.session_state.excel_mode = 'separate'
                        text_input = ""
                        st.info(f"📊 Will analyze first 5 rows separately")
                        preview_text = clean_text(st.session_state.excel_texts[0])
                        st.text_area("Preview (first row):", preview_text, height=200)
                    elif analysis_mode == "First 10 rows":
                        st.session_state.excel_texts = df.head(10)[selected_column].astype(str).tolist()
                        st.session_state.excel_mode = 'separate'
                        text_input = ""
                        st.info(f"📊 Will analyze first 10 rows separately")
                        preview_text = clean_text(st.session_state.excel_texts[0])
                        st.text_area("Preview (first row):", preview_text, height=200)

                except Exception as e:
                    st.error(f"❌ Error reading Excel file: {e}")
                    text_input = ""
            else:
                text_input = ""
        else:
            text_input = ""

    # Initialize session state for storing results
    if 'last_analysis_results' not in st.session_state:
        st.session_state.last_analysis_results = None
    if 'last_analysis_texts' not in st.session_state:
        st.session_state.last_analysis_texts = None
    if 'export_file_path' not in st.session_state:
        st.session_state.export_file_path = None
    if 'uploaded_df' not in st.session_state:
        st.session_state.uploaded_df = None

    # Check if we have uploaded Excel data available for export
    has_excel_data = (hasattr(st.session_state, 'uploaded_df') and
                     st.session_state.uploaded_df is not None and
                     len(st.session_state.uploaded_df) > 0)

    # Check if we have results to export (check BEFORE creating buttons to ensure proper state)
    has_results = (hasattr(st.session_state, 'last_analysis_results') and
                  st.session_state.last_analysis_results is not None and
                  len(st.session_state.last_analysis_results) > 0)

    export_enabled = has_results or has_excel_data

    # Button layout: Always show both buttons horizontally aligned in a container to prevent flickering
    button_container = st.container()
    with button_container:
        col_btn1, col_btn2, _ = st.columns([1.5, 1.5, 3])

        with col_btn1:
            analyze_clicked = st.button("🚀 Analyze Text",
                                       disabled=not PREDICTOR_AVAILABLE,
                                       use_container_width=True,
                                       key="analyze_btn")

        with col_btn2:
            # Set button help text
            button_help = "Click to export results to Excel" if export_enabled else "Analyze text first to enable export"

            export_clicked = st.button("💾 Export Results",
                                      disabled=not export_enabled,
                                      key="export_btn",
                                      type="primary",
                                      use_container_width=True,
                                      help=button_help)

    # Apply salmon/red color styling to Export Results button - ALWAYS salmon colored
    # Also prevent button flickering with stable positioning
    st.markdown("""
    <style>
    /* Prevent button flickering by stabilizing container */
    .stButton {
        position: relative !important;
        min-height: 40px !important;
        transition: none !important;
    }

    /* Target Export Results button by its text content */
    button[kind="primary"] {
        background-color: #ff6b6b !important;
        color: white !important;
        border: none !important;
        transition: opacity 0.15s ease !important;
    }
    button[kind="primary"]:hover:not(:disabled) {
        background-color: #ff5252 !important;
        color: white !important;
    }
    button[kind="primary"]:disabled {
        background-color: #ff6b6b !important;
        color: white !important;
        opacity: 0.6 !important;
        cursor: not-allowed !important;
    }
    /* Alternative: Target all buttons containing Export emoji */
    button p:contains("💾") {
        color: white !important;
    }
    /* Target the second column button specifically */
    div[data-testid="column"]:nth-child(2) button {
        background-color: #ff6b6b !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Import Path for export functionality
    from pathlib import Path

    # Handle export button click for top button
    if export_clicked:
        try:
            with st.spinner("💾 Generating Excel file..."):
                if has_excel_data:
                    # Export entire uploaded DataFrame with NER predictions
                    filepath = export_entire_dataframe(st.session_state.uploaded_df, st.session_state.predictor)
                elif st.session_state.last_analysis_results is not None:
                    # Export single text analysis results - fast version
                    filepath = export_text_results_fast(
                        st.session_state.last_analysis_texts,
                        st.session_state.last_analysis_results
                    )
                else:
                    st.error("❌ No data to export")
                    filepath = None

                if filepath and Path(filepath).exists():
                    # Get filename for display
                    filename = Path(filepath).name

                    # Auto-open the exports folder and select the file
                    import subprocess
                    import platform

                    try:
                        abs_filepath = Path(filepath).resolve()

                        if platform.system() == "Darwin":  # macOS
                            # Use 'open -R' to reveal/select the file in Finder
                            subprocess.run(["open", "-R", str(abs_filepath)])
                        elif platform.system() == "Windows":  # Windows
                            # Use 'explorer /select,' to select the file in Explorer
                            subprocess.run(["explorer", "/select,", str(abs_filepath)])
                        else:  # Linux
                            # Most Linux file managers don't support file selection via command line
                            # Fall back to opening the directory
                            exports_dir = abs_filepath.parent
                            subprocess.run(["xdg-open", str(exports_dir)])

                    except Exception as e:
                        # If folder opening fails, just continue with success message
                        pass

                    # Show simple success message
                    st.success(f"✅ Export completed! File saved to output/exports/{filename}")

                else:
                    st.error("❌ Failed to create export file")

        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")
            import traceback
            st.text(traceback.format_exc())


    # Process button
    if analyze_clicked:
        # Check if we're processing multiple texts separately
        is_multi_mode = (hasattr(st.session_state, 'excel_mode') and
                        st.session_state.excel_mode == 'separate' and
                        len(st.session_state.excel_texts) > 0)

        if text_input.strip() or is_multi_mode:
            if st.session_state.predictor is None:
                st.error("❌ Predictor not available. Please check the error messages above.")
            else:
                results = None  # Initialize results variable
                if is_multi_mode:
                    # Process multiple texts separately
                    num_texts = len(st.session_state.excel_texts)

                    # Create placeholder for warning message that can be cleared
                    warning_placeholder = st.empty()

                    # Show warning for large datasets
                    if num_texts > 100:
                        warning_placeholder.warning(f"⚠️ Processing {num_texts} texts. This may take several minutes. Consider using batch processing for very large datasets.")
                    elif num_texts > 20:
                        warning_placeholder.warning(f"⚠️ Processing {num_texts} texts. This may take up to a minute...")

                    try:
                        with st.spinner(f"🔬 Analyzing {num_texts} texts with BioBERT models..."):
                            all_results = process_multiple_texts(st.session_state.excel_texts, st.session_state.predictor)

                        # Store results for export
                        st.session_state.last_analysis_results = all_results
                        st.session_state.last_analysis_texts = st.session_state.excel_texts

                        # Clear warning message after analysis completes
                        warning_placeholder.empty()

                    except Exception as e:
                        # Clear warning on error too
                        warning_placeholder.empty()
                        st.error(f"❌ Error during multi-text processing: {str(e)}")
                        import traceback
                        st.text(traceback.format_exc())

                    # Display results for each text with index and divider
                    st.header("🎯 Entity Recognition Results")

                    # Calculate overall statistics
                    total_diseases = sum(r.get('entity_counts', {}).get('diseases', 0) for r in all_results)
                    total_genes = sum(r.get('entity_counts', {}).get('genes', 0) for r in all_results)
                    total_chemicals = sum(r.get('entity_counts', {}).get('chemicals', 0) for r in all_results)
                    total_entities = sum(r.get('entity_counts', {}).get('total', 0) for r in all_results)

                    # Context entity statistics
                    total_confirmed = sum(r.get('entity_counts', {}).get('confirmed', 0) for r in all_results)
                    total_negated = sum(r.get('entity_counts', {}).get('negated', 0) for r in all_results)
                    total_uncertain = sum(r.get('entity_counts', {}).get('uncertain', 0) for r in all_results)
                    total_historical = sum(r.get('entity_counts', {}).get('historical', 0) for r in all_results)
                    total_family = sum(r.get('entity_counts', {}).get('family', 0) for r in all_results)

                    # Overall statistics - Main entities
                    st.subheader("🔬 Entity Detection Summary")
                    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                    with col_stats1:
                        st.metric("🩺 Total Diseases", total_diseases)
                    with col_stats2:
                        st.metric("🧬 Total Genes", total_genes)
                    with col_stats3:
                        st.metric("💊 Total Chemicals", total_chemicals)
                    with col_stats4:
                        st.metric("📊 Total Entities", total_entities)

                    # Context entity statistics
                    st.subheader("🎯 Context Classification Summary")
                    col_ctx1, col_ctx2, col_ctx3, col_ctx4, col_ctx5 = st.columns(5)
                    with col_ctx1:
                        st.metric("✅ Confirmed", total_confirmed)
                    with col_ctx2:
                        st.metric("❌ Negated", total_negated)
                    with col_ctx3:
                        st.metric("❓ Uncertain", total_uncertain)
                    with col_ctx4:
                        st.metric("📅 Historical", total_historical)
                    with col_ctx5:
                        st.metric("👨‍👩‍👧 Family", total_family)

                    st.markdown("---")

                    # Display each row's results
                    for idx, (text, results) in enumerate(zip(st.session_state.excel_texts, all_results)):
                        # Row header with index
                        st.subheader(f"📄 Row {idx} Results")

                        # Entity counts for this row
                        entity_counts = results.get('entity_counts', {})
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Diseases", entity_counts.get('diseases', 0))
                        with col2:
                            st.metric("Genes", entity_counts.get('genes', 0))
                        with col3:
                            st.metric("Chemicals", entity_counts.get('chemicals', 0))
                        with col4:
                            st.metric("Entities", entity_counts.get('total', 0))

                        # Context entities for this row
                        col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns(5)
                        with col_c1:
                            st.metric("✅ Confirmed", entity_counts.get('confirmed', 0))
                        with col_c2:
                            st.metric("❌ Negated", entity_counts.get('negated', 0))
                        with col_c3:
                            st.metric("❓ Uncertain", entity_counts.get('uncertain', 0))
                        with col_c4:
                            st.metric("📅 Historical", entity_counts.get('historical', 0))
                        with col_c5:
                            st.metric("👨‍👩‍👧 Family", entity_counts.get('family', 0))

                        # Section categories display
                        result_dict = results.get('result', {})
                        section_categories = result_dict.get('section_categories', '')
                        if section_categories and section_categories != 'General Clinical Text':
                            st.info(f"**📋 Sections:** {section_categories}")

                        # Highlighted text
                        st.caption("**Context Icons:** ✅ Confirmed | ❌ Negated | ❓ Uncertain | 📅 Historical | 👨‍👩‍👧 Family History · Hover for reason")
                        formatted_text = results.get('formatted_text', text)
                        st.markdown(formatted_text, unsafe_allow_html=True)

                        # Entity table for this row
                        entities = results.get('entities', [])
                        if entities:
                            with st.expander(f"📋 View {len(entities)} entities for Row {idx}", expanded=False):
                                entity_data = []
                                for entity in entities:
                                    entity_data.append({
                                        'Text': entity.get('text', ''),
                                        'Label': entity.get('label', ''),
                                        'Confidence': entity.get('confidence', 'N/A')
                                    })
                                entity_df = pd.DataFrame(entity_data)
                                st.dataframe(entity_df, use_container_width=True)

                        # Context entities details for this row
                        result_dict = results.get('result', {})
                        context_entities_exist = (
                            result_dict.get('confirmed_entities_count', 0) > 0 or
                            result_dict.get('negated_entities_count', 0) > 0 or
                            result_dict.get('uncertain_entities_count', 0) > 0 or
                            result_dict.get('historical_entities_count', 0) > 0 or
                            result_dict.get('family_entities_count', 0) > 0
                        )

                        if context_entities_exist:
                            with st.expander(f"🎯 View context classifications for Row {idx}", expanded=False):
                                if result_dict.get('confirmed_entities_count', 0) > 0:
                                    st.markdown(f"<span style='font-size: 18px;'><strong>✅ Confirmed ({result_dict.get('confirmed_entities_count', 0)}):</strong></span> <span style='font-size: 14px;'>{result_dict.get('confirmed_entities', '')}</span>", unsafe_allow_html=True)
                                    if result_dict.get('confirmed_entities_predictors'):
                                        st.markdown(f"<span style='font-size: 14px; color: #808080;'>Predictors: {result_dict.get('confirmed_entities_predictors', '')}</span>", unsafe_allow_html=True)
                                    if result_dict.get('confirmed_context_sentences'):
                                        st.markdown(f"<span style='font-size: 18px;'><strong>Context:</strong></span><br>{format_context_sentences(result_dict.get('confirmed_context_sentences', ''))}", unsafe_allow_html=True)

                                if result_dict.get('negated_entities_count', 0) > 0:
                                    st.markdown(f"<span style='font-size: 18px;'><strong>❌ Negated ({result_dict.get('negated_entities_count', 0)}):</strong></span> <span style='font-size: 14px;'>{result_dict.get('negated_entities', '')}</span>", unsafe_allow_html=True)
                                    if result_dict.get('negated_entities_predictors'):
                                        st.markdown(f"<span style='font-size: 14px; color: #808080;'>Predictors: {result_dict.get('negated_entities_predictors', '')}</span>", unsafe_allow_html=True)
                                    if result_dict.get('negated_context_sentences'):
                                        st.markdown(f"<span style='font-size: 18px;'><strong>Context:</strong></span><br>{format_context_sentences(result_dict.get('negated_context_sentences', ''))}", unsafe_allow_html=True)

                                if result_dict.get('uncertain_entities_count', 0) > 0:
                                    st.markdown(f"<span style='font-size: 18px;'><strong>❓ Uncertain ({result_dict.get('uncertain_entities_count', 0)}):</strong></span> <span style='font-size: 14px;'>{result_dict.get('uncertain_entities', '')}</span>", unsafe_allow_html=True)
                                    if result_dict.get('uncertain_entities_predictors'):
                                        st.markdown(f"<span style='font-size: 14px; color: #808080;'>Predictors: {result_dict.get('uncertain_entities_predictors', '')}</span>", unsafe_allow_html=True)
                                    if result_dict.get('uncertain_context_sentences'):
                                        st.markdown(f"<span style='font-size: 18px;'><strong>Context:</strong></span><br>{format_context_sentences(result_dict.get('uncertain_context_sentences', ''))}", unsafe_allow_html=True)

                                if result_dict.get('historical_entities_count', 0) > 0:
                                    st.markdown(f"<span style='font-size: 18px;'><strong>📅 Historical ({result_dict.get('historical_entities_count', 0)}):</strong></span> <span style='font-size: 14px;'>{result_dict.get('historical_entities', '')}</span>", unsafe_allow_html=True)
                                    if result_dict.get('historical_entities_predictors'):
                                        st.markdown(f"<span style='font-size: 14px; color: #808080;'>Predictors: {result_dict.get('historical_entities_predictors', '')}</span>", unsafe_allow_html=True)
                                    if result_dict.get('historical_context_sentences'):
                                        st.markdown(f"<span style='font-size: 18px;'><strong>Context:</strong></span><br>{format_context_sentences(result_dict.get('historical_context_sentences', ''))}", unsafe_allow_html=True)

                                if result_dict.get('family_entities_count', 0) > 0:
                                    st.markdown(f"<span style='font-size: 18px;'><strong>👨‍👩‍👧 Family History ({result_dict.get('family_entities_count', 0)}):</strong></span> <span style='font-size: 14px;'>{result_dict.get('family_entities', '')}</span>", unsafe_allow_html=True)
                                    if result_dict.get('family_entities_predictors'):
                                        st.markdown(f"<span style='font-size: 14px; color: #808080;'>Predictors: {result_dict.get('family_entities_predictors', '')}</span>", unsafe_allow_html=True)
                                    if result_dict.get('family_context_sentences'):
                                        st.markdown(f"<span style='font-size: 18px;'><strong>Context:</strong></span><br>{format_context_sentences(result_dict.get('family_context_sentences', ''))}", unsafe_allow_html=True)

                        # Divider between rows
                        st.markdown("---")

                    # Add export button AFTER all visualizations are displayed for multi-text mode
                    st.markdown("---")
                    st.subheader("💾 Export Results")

                    export_multi_btn = st.button(
                        "📥 Download All Results as Excel",
                        type="primary",
                        use_container_width=False,
                        key="export_multi_bottom"
                    )

                    if export_multi_btn:
                        try:
                            with st.spinner("💾 Generating Excel file..."):
                                # Export entire uploaded DataFrame with NER predictions
                                filepath = export_entire_dataframe(st.session_state.uploaded_df, st.session_state.predictor)

                            if filepath and Path(filepath).exists():
                                # Get filename for display
                                filename = Path(filepath).name

                                # Auto-open the exports folder and select the file
                                import subprocess
                                import platform

                                try:
                                    abs_filepath = Path(filepath).resolve()

                                    if platform.system() == "Darwin":  # macOS
                                        subprocess.run(["open", "-R", str(abs_filepath)])
                                    elif platform.system() == "Windows":  # Windows
                                        subprocess.run(["explorer", "/select,", str(abs_filepath)])
                                    else:  # Linux
                                        exports_dir = abs_filepath.parent
                                        subprocess.run(["xdg-open", str(exports_dir)])

                                except Exception as e:
                                    pass

                                st.success(f"✅ Export completed! File saved to output/exports/{filename}")
                            else:
                                st.error("❌ Failed to create export file")

                        except Exception as e:
                            st.error(f"❌ Export failed: {str(e)}")
                            import traceback
                            st.text(traceback.format_exc())

                else:
                    # Single text processing
                    with st.spinner("🔬 Analyzing medical entities with BioBERT models..."):
                        results = process_text(text_input, st.session_state.predictor)

                    # Store results for export
                    st.session_state.last_analysis_results = [results]
                    st.session_state.last_analysis_texts = [text_input]

                if results and not is_multi_mode:
                    # Entity visualization
                    st.header("🎯 Entity Recognition Results")

                    # Entity statistics
                    entity_counts = results.get('entity_counts', {})

                    st.subheader("🔬 Entity Detection")
                    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                    with col_stats1:
                        st.metric("🩺 Diseases", entity_counts.get('diseases', 0))
                    with col_stats2:
                        st.metric("🧬 Genes", entity_counts.get('genes', 0))
                    with col_stats3:
                        st.metric("💊 Chemicals", entity_counts.get('chemicals', 0))
                    with col_stats4:
                        st.metric("📊 Total", entity_counts.get('total', 0))

                    # Context entity statistics
                    st.subheader("🎯 Context Classification")
                    col_ctx1, col_ctx2, col_ctx3, col_ctx4, col_ctx5 = st.columns(5)
                    with col_ctx1:
                        st.metric("✅ Confirmed", entity_counts.get('confirmed', 0))
                    with col_ctx2:
                        st.metric("❌ Negated", entity_counts.get('negated', 0))
                    with col_ctx3:
                        st.metric("❓ Uncertain", entity_counts.get('uncertain', 0))
                    with col_ctx4:
                        st.metric("📅 Historical", entity_counts.get('historical', 0))
                    with col_ctx5:
                        st.metric("👨‍👩‍👧 Family", entity_counts.get('family', 0))

                    # Section categories display
                    result_dict = results.get('result', {})
                    section_categories = result_dict.get('section_categories', '')
                    if section_categories and section_categories != 'General Clinical Text':
                        st.info(f"**📋 Detected Clinical Sections:** {section_categories}")

                    # Highlighted text display
                    st.subheader("🔍 Text with Entity Highlighting")

                    # Add legend for context icons
                    st.caption("**Context Icons:** ✅ Confirmed | ❌ Negated | ❓ Uncertain | 📅 Historical | 👨‍👩‍👧 Family History")
                    st.caption("💡 **Tip:** Hover over the context icons to see WHY an entity was classified (predictor patterns used)")

                    formatted_text = results.get('formatted_text', text_input)
                    st.markdown(formatted_text, unsafe_allow_html=True)

                    # Entity details table
                    entities = results.get('entities', [])
                    if entities:
                        st.subheader("📋 Detected Entities")

                        # Create DataFrame for entity display
                        entity_data = []
                        for entity in entities:
                            entity_data.append({
                                'Text': entity.get('text', ''),
                                'Label': entity.get('label', ''),
                                'Start': entity.get('start', 0),
                                'End': entity.get('end', 0),
                                'Confidence': entity.get('confidence', 'N/A')
                            })

                        entity_df = pd.DataFrame(entity_data)
                        st.dataframe(entity_df, use_container_width=True)

                    # Context entities details
                    result_dict = results.get('result', {})
                    context_entities_exist = (
                        result_dict.get('confirmed_entities_count', 0) > 0 or
                        result_dict.get('negated_entities_count', 0) > 0 or
                        result_dict.get('uncertain_entities_count', 0) > 0 or
                        result_dict.get('historical_entities_count', 0) > 0 or
                        result_dict.get('family_entities_count', 0) > 0
                    )

                    if context_entities_exist:
                        st.subheader("🎯 Context Classifications Details")

                        if result_dict.get('confirmed_entities_count', 0) > 0:
                            with st.expander(f"✅ Confirmed Entities ({result_dict.get('confirmed_entities_count', 0)})", expanded=True):
                                st.markdown(f"**Entities:** {result_dict.get('confirmed_entities', '')}")
                                if result_dict.get('confirmed_entities_predictors'):
                                    st.caption(f"**Predictors:** {result_dict.get('confirmed_entities_predictors', '')}")
                                if result_dict.get('confirmed_context_sentences'):
                                    st.markdown(f"**Context:**<br>{format_context_sentences(result_dict.get('confirmed_context_sentences', ''))}", unsafe_allow_html=True)

                        if result_dict.get('negated_entities_count', 0) > 0:
                            with st.expander(f"❌ Negated Entities ({result_dict.get('negated_entities_count', 0)})", expanded=True):
                                st.markdown(f"**Entities:** {result_dict.get('negated_entities', '')}")
                                if result_dict.get('negated_entities_predictors'):
                                    st.caption(f"**Predictors:** {result_dict.get('negated_entities_predictors', '')}")
                                if result_dict.get('negated_context_sentences'):
                                    st.markdown(f"**Context:**<br>{format_context_sentences(result_dict.get('negated_context_sentences', ''))}", unsafe_allow_html=True)

                        if result_dict.get('uncertain_entities_count', 0) > 0:
                            with st.expander(f"❓ Uncertain Entities ({result_dict.get('uncertain_entities_count', 0)})", expanded=True):
                                st.markdown(f"**Entities:** {result_dict.get('uncertain_entities', '')}")
                                if result_dict.get('uncertain_entities_predictors'):
                                    st.caption(f"**Predictors:** {result_dict.get('uncertain_entities_predictors', '')}")
                                if result_dict.get('uncertain_context_sentences'):
                                    st.markdown(f"**Context:**<br>{format_context_sentences(result_dict.get('uncertain_context_sentences', ''))}", unsafe_allow_html=True)

                        if result_dict.get('historical_entities_count', 0) > 0:
                            with st.expander(f"📅 Historical Entities ({result_dict.get('historical_entities_count', 0)})", expanded=True):
                                st.markdown(f"**Entities:** {result_dict.get('historical_entities', '')}")
                                if result_dict.get('historical_entities_predictors'):
                                    st.caption(f"**Predictors:** {result_dict.get('historical_entities_predictors', '')}")
                                if result_dict.get('historical_context_sentences'):
                                    st.markdown(f"**Context:**<br>{format_context_sentences(result_dict.get('historical_context_sentences', ''))}", unsafe_allow_html=True)

                        if result_dict.get('family_entities_count', 0) > 0:
                            with st.expander(f"👨‍👩‍👧 Family History Entities ({result_dict.get('family_entities_count', 0)})", expanded=True):
                                st.markdown(f"**Entities:** {result_dict.get('family_entities', '')}")
                                if result_dict.get('family_entities_predictors'):
                                    st.caption(f"**Predictors:** {result_dict.get('family_entities_predictors', '')}")
                                if result_dict.get('family_context_sentences'):
                                    st.markdown(f"**Context:**<br>{format_context_sentences(result_dict.get('family_context_sentences', ''))}", unsafe_allow_html=True)

                    # Raw results
                    if show_raw_results:
                        st.subheader("🔧 Raw Analysis Results")
                        raw_result = results.get('result', {})
                        st.json(raw_result)

                    # Add export button AFTER visualizations are displayed
                    st.markdown("---")
                    st.subheader("💾 Export Results")

                    export_results_btn = st.button(
                        "📥 Download Results as Excel",
                        type="primary",
                        use_container_width=False,
                        key="export_results_bottom"
                    )

                    if export_results_btn:
                        try:
                            with st.spinner("💾 Generating Excel file..."):
                                # Export single text analysis results - fast version
                                filepath = export_text_results_fast(
                                    st.session_state.last_analysis_texts,
                                    st.session_state.last_analysis_results
                                )

                            if filepath and Path(filepath).exists():
                                # Get filename for display
                                filename = Path(filepath).name

                                # Auto-open the exports folder and select the file
                                import subprocess
                                import platform

                                try:
                                    abs_filepath = Path(filepath).resolve()

                                    if platform.system() == "Darwin":  # macOS
                                        subprocess.run(["open", "-R", str(abs_filepath)])
                                    elif platform.system() == "Windows":  # Windows
                                        subprocess.run(["explorer", "/select,", str(abs_filepath)])
                                    else:  # Linux
                                        exports_dir = abs_filepath.parent
                                        subprocess.run(["xdg-open", str(exports_dir)])

                                except Exception as e:
                                    pass

                                st.success(f"✅ Export completed! File saved to output/exports/{filename}")
                            else:
                                st.error("❌ Failed to create export file")

                        except Exception as e:
                            st.error(f"❌ Export failed: {str(e)}")
                            import traceback
                            st.text(traceback.format_exc())

        else:
            st.warning("⚠️ Please enter some text to analyze")


if __name__ == "__main__":
    if PREDICTOR_AVAILABLE:
        main()
    else:
        st.error("❌ Enhanced Medical NER Predictor not available. Please run from the project root directory.")
        st.info("Make sure `enhanced_medical_ner_predictor.py` is accessible in the parent directory.")