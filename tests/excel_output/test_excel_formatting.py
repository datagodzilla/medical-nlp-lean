#!/usr/bin/env python3
"""Test Excel export formatting with dynamic column width and left alignment"""

import pandas as pd
from pathlib import Path
from datetime import datetime

# Test data with varying column content lengths
test_data = [
    {
        'Index': 0,
        'Text': 'Patient has diabetes and hypertension. No chest pain.',
        'detected_diseases': 'diabetes; hypertension',
        'total_diseases_count': 2,
        'detected_genes': 'KIF5A',
        'total_gene_count': 1,
        'confirmed_entities': 'diabetes; hypertension',
        'confirmed_context_sentences': '[HAS | diabetes] ...Patient has diabetes and hypertension. No chest pain or shortness of breath....',
        'negated_entities': 'chest pain',
        'negated_context_sentences': '[NO | chest pain] ...No chest pain or shortness of breath....'
    },
    {
        'Index': 1,
        'Text': 'Patient diagnosed with KIF5A mutation and Parkinsons disease.',
        'detected_diseases': 'parkinsons disease',
        'total_diseases_count': 1,
        'detected_genes': 'KIF5A',
        'total_gene_count': 1,
        'confirmed_entities': 'KIF5A; parkinsons disease',
        'confirmed_context_sentences': '[DIAGNOSED WITH | parkinsons disease] ...Patient diagnosed with KIF5A mutation and Parkinsons disease....'
    }
]

# Create DataFrame
df = pd.DataFrame(test_data)

# Create output directory
output_dir = Path('output/exports')
output_dir.mkdir(parents=True, exist_ok=True)

# Generate filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'test_formatted_export_v2_{timestamp}.xlsx'
filepath = output_dir / filename

print(f"Creating test export with dynamic column widths: {filepath}\n")

# Apply formatting
try:
    import xlsxwriter

    with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
        # Write dataframe
        df.to_excel(writer, sheet_name='NER_Results', index=False, startrow=0, startcol=0)

        # Get workbook and worksheet
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

        # Calculate dynamic column widths
        print("Column widths:")
        for col_num, column_name in enumerate(df.columns):
            # Get column data
            column_data = df.iloc[:, col_num]

            # Calculate max length considering header and data
            max_len = len(str(column_name))  # Header length

            # Check first 100 rows for content width
            for value in column_data.head(100):
                if pd.notna(value):
                    # For long text, estimate based on first 100 chars
                    value_str = str(value)[:100]
                    max_len = max(max_len, len(value_str))

            # Set width: minimum 10, maximum 25, with padding
            column_width = min(25, max(10, max_len + 2))
            worksheet.set_column(col_num, col_num, column_width, cell_format)

            print(f"   {column_name:35s} -> width={column_width:2d} (content_max={max_len})")

        # Format header row
        for col_num, column_name in enumerate(df.columns):
            worksheet.write(0, col_num, column_name, header_format)

        # Set header row height to 20 points
        worksheet.set_row(0, 20)

        # Set data row height to 15 points
        for row_num in range(1, len(df) + 1):
            worksheet.set_row(row_num, 15)

        # Freeze top row
        worksheet.freeze_panes(1, 0)

        # Add autofilter
        worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)

    print("\n✅ Export successful with formatting!")
    print(f"   - Header row: height=20, green background, LEFT aligned, text wrap, bold")
    print(f"   - Data rows: height=15, text wrap")
    print(f"   - Column widths: dynamic (10-25 based on content)")
    print(f"   - Top row frozen")
    print(f"   - Filters applied")
    print(f"\n📁 File location: {filepath.resolve()}")

except ImportError:
    print("❌ xlsxwriter not available - formatting not applied")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
