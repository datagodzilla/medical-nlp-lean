#!/usr/bin/env python3
"""Test colored HTML visualization"""

import re

def convert_text_markers_to_colored_html(text_viz: str) -> str:
    """Convert text markers ▶[entity]◀ (TYPE | CONTEXT) to colored HTML spans."""
    # Define colors for entity types
    entity_colors = {
        'DISEASE': '#ff6b6b',
        'GENE': '#52c41a',
        'DRUG': '#45b7d1',
        'CHEMICAL': '#45b7d1',
        'ANATOMY': '#f9ca24',
        'SYMPTOM': '#ff9999',
        'TREATMENT': '#99ccff'
    }

    # Define colors for context classifications
    context_colors = {
        'CONFIRMED': '#4CAF50',
        'NEGATED': '#f44336',
        'UNCERTAIN': '#ff9800',
        'HISTORICAL': '#9c27b0',
        'FAMILY': '#2196F3'
    }

    # Pattern to match: ▶[entity_text]◀ (ENTITY_TYPE | CONTEXT1 | CONTEXT2)
    pattern = r'▶\[([^\]]+)\]◀\s*\(([^)]+)\)'

    def replace_marker(match):
        entity_text = match.group(1)
        type_and_context = match.group(2)

        # Parse type and contexts
        parts = [p.strip() for p in type_and_context.split('|')]
        entity_type = parts[0] if parts else 'UNKNOWN'
        contexts = parts[1:] if len(parts) > 1 else []

        # Get entity color
        entity_color = entity_colors.get(entity_type, '#95a5a6')

        # Build HTML
        html = f'<span style="background-color: {entity_color}; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; margin: 0 2px;">{entity_text}</span>'

        # Add context badges
        if contexts:
            for ctx in contexts:
                ctx_color = context_colors.get(ctx, '#757575')
                badge = f'<span style="background-color: {ctx_color}; color: white; padding: 1px 4px; border-radius: 3px; font-size: 0.85em; margin-left: 2px;">{ctx}</span>'
                html += badge

        return html

    # Replace all markers
    html_text = re.sub(pattern, replace_marker, text_viz)

    # Wrap in div
    styled_html = f'''<div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; border: 1px solid #dee2e6; font-family: sans-serif; line-height: 1.8; font-size: 15px;">{html_text}</div>'''

    return styled_html

# Test with sample visualization
test_viz = """📋 SECTION: General Clinical Text

================================================================================
LEGEND: ▶[Entity]◀ (TYPE | CONTEXT)
CONTEXTS: CONFIRMED | NEGATED | UNCERTAIN | HISTORICAL | FAMILY
================================================================================

Patient has ▶[diabetes]◀ (DISEASE | CONFIRMED) and ▶[hypertension]◀ (DISEASE | CONFIRMED). No ▶[chest pain]◀ (DISEASE | NEGATED).

Patient diagnosed with ▶[KIF5A mutation]◀ (GENE | FAMILY) and ▶[parkinson]◀ (DISEASE | FAMILY)'s disease.

No ▶[fever]◀ (DISEASE | NEGATED) but patient has persistent ▶[cough]◀ (DISEASE | NEGATED) and ▶[headache]◀ (DISEASE | NEGATED)."""

print("Converting text markers to colored HTML...")
html_output = convert_text_markers_to_colored_html(test_viz)

# Save to HTML file for viewing
with open('output/test_colored_viz.html', 'w', encoding='utf-8') as f:
    f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Colored Entity Visualization Test</title>
</head>
<body>
    <h2>Colored Entity Visualization Test</h2>
    {html_output}
</body>
</html>
""")

print("✅ HTML file created: output/test_colored_viz.html")
print("\nHTML preview:")
print(html_output[:500] + "...")
