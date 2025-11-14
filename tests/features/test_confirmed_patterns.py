#!/usr/bin/env python3
import pandas as pd

# Check what confirmed patterns are in the template
df = pd.read_excel('data/external/confirmed_rules_template.xlsx')
print("Confirmed patterns that might match 'but reports' or 'reports':")
for i, pattern in enumerate(df['pattern']):
    if 'report' in str(pattern).lower():
        print(f"  {i}: '{pattern}'")

print("\nConfirmed patterns that start with 'but':")
for i, pattern in enumerate(df['pattern']):
    if str(pattern).lower().startswith('but'):
        print(f"  {i}: '{pattern}'")

print(f"\nTotal confirmed patterns: {len(df)}")
print("\nFirst 20 confirmed patterns:")
for i, pattern in enumerate(df['pattern'][:20]):
    print(f"  {i}: '{pattern}'")