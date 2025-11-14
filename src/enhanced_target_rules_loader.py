#!/usr/bin/env python3
"""
enhanced_target_rules_loader.py

Enhanced target rules loader that processes the target_rules_template.xlsx
and creates comprehensive entity mapping rules for improved disease, gene, and drug prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple
import logging
import re

logger = logging.getLogger(__name__)

# Get project root directory
project_root = Path(__file__).parent

class EnhancedTargetRulesLoader:
    """Enhanced loader for target rules with improved entity mapping"""

    def __init__(self):
        """Initialize the enhanced target rules loader"""
        self.target_rules_file = project_root / "data/external/target_rules_template.xlsx"
        self.disease_terms = set()
        self.gene_terms = set()
        self.drug_terms = set()
        self.anatomical_terms = set()
        self.symptom_terms = set()
        self.treatment_terms = set()

        # Comprehensive drug/chemical terms (since template doesn't have them)
        self.additional_drug_terms = self._get_comprehensive_drug_terms()

    def load_target_rules(self) -> Dict[str, Set[str]]:
        """Load and process target rules from the template file"""
        try:
            if not self.target_rules_file.exists():
                logger.warning(f"Target rules file not found: {self.target_rules_file}")
                return self._get_fallback_terms()

            # Read the target rules template
            df = pd.read_excel(self.target_rules_file)
            logger.info(f"✅ Loaded {len(df)} target rules from template")

            # Process terms by category
            self._process_disease_terms(df)
            self._process_gene_terms(df)
            self._process_other_terms(df)

            # Add comprehensive drug terms
            self.drug_terms.update(self.additional_drug_terms)

            # Log statistics
            logger.info(f"📊 Processed target rules:")
            logger.info(f"   - Disease terms: {len(self.disease_terms)}")
            logger.info(f"   - Gene terms: {len(self.gene_terms)}")
            logger.info(f"   - Drug terms: {len(self.drug_terms)}")
            logger.info(f"   - Anatomical terms: {len(self.anatomical_terms)}")
            logger.info(f"   - Symptom terms: {len(self.symptom_terms)}")
            logger.info(f"   - Treatment terms: {len(self.treatment_terms)}")

            return {
                'diseases': self.disease_terms,
                'genes': self.gene_terms,
                'drugs': self.drug_terms,
                'anatomical': self.anatomical_terms,
                'symptoms': self.symptom_terms,
                'treatments': self.treatment_terms
            }

        except Exception as e:
            logger.error(f"Error loading target rules: {e}")
            return self._get_fallback_terms()

    def _process_disease_terms(self, df: pd.DataFrame):
        """Process disease-related terms from the dataframe"""
        # Disease categories to include
        disease_categories = [
            'Disease', 'Malformation syndrome', 'Clinical subtype',
            'Etiological subtype', 'Histopathological subtype',
            'Clinical syndrome', 'Clinical group', 'Cancer',
            'Neurological', 'Cardiovascular', 'Respiratory', 'Metabolic'
        ]

        disease_labels = ['DISEASE', 'PROBLEM']

        # Extract disease terms
        disease_mask = (
            df['Category'].isin(disease_categories) |
            df['Entity_Type'].isin(disease_labels)
        )

        disease_df = df[disease_mask]

        for _, row in disease_df.iterrows():
            term = str(row['Pattern']).strip()
            if term and len(term) > 2 and term.lower() != 'nan':
                # Clean and normalize term
                clean_term = self._clean_medical_term(term)
                if clean_term:
                    self.disease_terms.add(clean_term.lower())

                    # Add variations
                    variations = self._generate_term_variations(clean_term)
                    self.disease_terms.update(variations)

    def _process_gene_terms(self, df: pd.DataFrame):
        """Process gene-related terms from the dataframe"""
        gene_mask = df['Entity_Type'] == 'GENE'
        gene_df = df[gene_mask]

        for _, row in gene_df.iterrows():
            term = str(row['Pattern']).strip()
            if term and len(term) > 1 and term.lower() != 'nan':
                # Clean and normalize term
                clean_term = self._clean_gene_term(term)
                if clean_term:
                    self.gene_terms.add(clean_term.lower())

                    # Add gene name variations (uppercase, with/without numbers)
                    self.gene_terms.add(clean_term.upper())

                    # Extract gene symbols from complex names
                    gene_symbols = self._extract_gene_symbols(clean_term)
                    self.gene_terms.update(gene_symbols)

    def _process_other_terms(self, df: pd.DataFrame):
        """Process other medical terms (anatomical, symptoms, treatments)"""
        # Anatomical terms
        anatomical_mask = df['Entity_Type'] == 'ANATOMICAL_LOCATION'
        anatomical_df = df[anatomical_mask]
        for _, row in anatomical_df.iterrows():
            term = self._clean_medical_term(str(row['Pattern']))
            if term:
                self.anatomical_terms.add(term.lower())

        # Symptom terms
        symptom_mask = df['Entity_Type'] == 'SYMPTOM'
        symptom_df = df[symptom_mask]
        for _, row in symptom_df.iterrows():
            term = self._clean_medical_term(str(row['Pattern']))
            if term:
                self.symptom_terms.add(term.lower())

        # Treatment terms
        treatment_mask = df['Entity_Type'] == 'TREATMENT'
        treatment_df = df[treatment_mask]
        for _, row in treatment_df.iterrows():
            term = self._clean_medical_term(str(row['Pattern']))
            if term:
                self.treatment_terms.add(term.lower())

    def _clean_medical_term(self, term: str) -> str:
        """Clean and normalize medical terms"""
        if not term or term.lower() in ['nan', 'none', '']:
            return ""

        # Remove extra whitespace and special characters
        clean_term = re.sub(r'\s+', ' ', term.strip())

        # Remove parenthetical information but keep the main term
        # Example: "diabetes mellitus (type 2)" -> "diabetes mellitus"
        main_term = re.sub(r'\s*\([^)]*\)\s*', ' ', clean_term).strip()

        # Keep both full term and main term if different
        if main_term and len(main_term) > 2:
            return main_term

        return clean_term if len(clean_term) > 2 else ""

    def _clean_gene_term(self, term: str) -> str:
        """Clean and normalize gene terms"""
        if not term or term.lower() in ['nan', 'none', '']:
            return ""

        # Remove extra whitespace
        clean_term = re.sub(r'\s+', ' ', term.strip())

        # For gene terms, keep everything but clean up formatting
        return clean_term if len(clean_term) > 1 else ""

    def _extract_gene_symbols(self, gene_term: str) -> List[str]:
        """Extract gene symbols from complex gene names"""
        symbols = []

        # Pattern for gene symbols in parentheses
        parentheses_matches = re.findall(r'\(([A-Z0-9]+[A-Z0-9\-]*)\)', gene_term)
        symbols.extend([match.lower() for match in parentheses_matches if len(match) > 1])

        # Pattern for short gene names (2-6 uppercase letters/numbers)
        short_gene_pattern = re.findall(r'\b([A-Z]{2,6}[0-9]*)\b', gene_term)
        symbols.extend([match.lower() for match in short_gene_pattern])

        return list(set(symbols))

    def _generate_term_variations(self, term: str) -> List[str]:
        """Generate common variations of medical terms"""
        variations = []

        # Add plural/singular variations
        if term.endswith('s') and len(term) > 3:
            variations.append(term[:-1])  # Remove 's'
        elif not term.endswith('s'):
            variations.append(term + 's')  # Add 's'

        # Add hyphenated variations
        if ' ' in term:
            variations.append(term.replace(' ', '-'))
        if '-' in term:
            variations.append(term.replace('-', ' '))

        # Add abbreviation if term has multiple words
        words = term.split()
        if len(words) > 1:
            abbreviation = ''.join([word[0].upper() for word in words if len(word) > 0])
            if len(abbreviation) > 1:
                variations.append(abbreviation.lower())

        return [v.lower() for v in variations if len(v) > 1]

    def _get_comprehensive_drug_terms(self) -> Set[str]:
        """Get comprehensive list of drug/chemical terms"""
        drug_terms = {
            # Common medications
            'acetaminophen', 'paracetamol', 'aspirin', 'ibuprofen', 'naproxen', 'diclofenac',
            'metformin', 'insulin', 'glyburide', 'glipizide', 'pioglitazone', 'sitagliptin',
            'lisinopril', 'enalapril', 'losartan', 'valsartan', 'amlodipine', 'nifedipine',
            'hydrochlorothiazide', 'furosemide', 'spironolactone', 'atenolol', 'metoprolol',
            'simvastatin', 'atorvastatin', 'rosuvastatin', 'pravastatin', 'lovastatin',
            'omeprazole', 'lansoprazole', 'pantoprazole', 'ranitidine', 'famotidine',
            'albuterol', 'fluticasone', 'budesonide', 'montelukast', 'theophylline',
            'levothyroxine', 'methimazole', 'propylthiouracil', 'prednisone', 'prednisolone',
            'warfarin', 'heparin', 'clopidogrel', 'rivaroxaban', 'dabigatran',
            'sertraline', 'fluoxetine', 'paroxetine', 'citalopram', 'escitalopram',
            'lorazepam', 'diazepam', 'alprazolam', 'clonazepam', 'zolpidem',
            'morphine', 'oxycodone', 'hydrocodone', 'tramadol', 'fentanyl',
            'amoxicillin', 'azithromycin', 'ciprofloxacin', 'levofloxacin', 'clindamycin',

            # Chemotherapy and cancer drugs
            'cisplatin', 'carboplatin', 'oxaliplatin', 'doxorubicin', 'cyclophosphamide',
            'methotrexate', 'fluorouracil', '5-fu', 'paclitaxel', 'docetaxel',
            'bevacizumab', 'trastuzumab', 'rituximab', 'cetuximab', 'imatinib',

            # Biologics and immunosuppressants
            'adalimumab', 'infliximab', 'etanercept', 'rituximab', 'tocilizumab',
            'azathioprine', 'mycophenolate', 'tacrolimus', 'cyclosporine', 'sirolimus',

            # Vitamins and supplements
            'vitamin d', 'vitamin b12', 'folic acid', 'iron', 'calcium', 'magnesium',
            'potassium', 'zinc', 'omega-3', 'coq10', 'biotin', 'thiamine',

            # Chemical compounds
            'sodium chloride', 'potassium chloride', 'calcium carbonate', 'magnesium sulfate',
            'dextrose', 'lactose', 'sucrose', 'mannitol', 'sorbitol', 'glycerin',

            # Drug classes
            'beta blocker', 'ace inhibitor', 'arb', 'calcium channel blocker', 'diuretic',
            'statin', 'ppi', 'nsaid', 'ssri', 'snri', 'benzodiazepine', 'opioid',
            'antibiotic', 'antifungal', 'antiviral', 'chemotherapy', 'immunosuppressant'
        }

        # Add variations and brand names
        expanded_terms = set(drug_terms)

        # Add common variations
        for term in list(drug_terms):
            # Add with/without spaces and hyphens
            if ' ' in term:
                expanded_terms.add(term.replace(' ', ''))
                expanded_terms.add(term.replace(' ', '-'))
            if '-' in term:
                expanded_terms.add(term.replace('-', ' '))
                expanded_terms.add(term.replace('-', ''))

        return expanded_terms

    def _get_fallback_terms(self) -> Dict[str, Set[str]]:
        """Get fallback terms if target rules file is not available"""
        logger.warning("Using fallback medical terms")

        fallback_diseases = {
            'diabetes', 'diabetes mellitus', 'hypertension', 'cancer', 'infection', 'fever',
            'pneumonia', 'asthma', 'copd', 'heart failure', 'stroke', 'myocardial infarction',
            'sepsis', 'kidney disease', 'liver disease', 'alzheimer', 'parkinson', 'epilepsy',
            'depression', 'anxiety', 'schizophrenia', 'bipolar', 'autism', 'adhd', 'obesity',
            'malignancy', 'metastatic disease', 'lymphadenopathy', 'community-acquired pneumonia',
            'chest pain', 'bleeding', 'headache', 'nausea', 'vomiting', 'diarrhea', 'constipation',
            'fatigue', 'weakness', 'pain', 'inflammation', 'thrombosis', 'embolism', 'edema'
        }

        fallback_genes = {
            'brca1', 'brca2', 'tp53', 'apoe', 'cftr', 'fmr1', 'htt', 'dmd', 'f8', 'f9',
            'pah', 'hfe', 'mthfr', 'cyp2d6', 'cyp2c19', 'aldh2', 'vkorc1', 'slc6a4'
        }

        return {
            'diseases': fallback_diseases,
            'genes': fallback_genes,
            'drugs': self.additional_drug_terms,
            'anatomical': set(),
            'symptoms': set(),
            'treatments': set()
        }

def load_enhanced_target_rules() -> Dict[str, Set[str]]:
    """Convenience function to load enhanced target rules"""
    loader = EnhancedTargetRulesLoader()
    return loader.load_target_rules()

if __name__ == "__main__":
    # Test the loader
    loader = EnhancedTargetRulesLoader()
    rules = loader.load_target_rules()

    print("Enhanced Target Rules Loaded:")
    for category, terms in rules.items():
        print(f"{category.upper()}: {len(terms)} terms")
        if terms:
            sample_terms = list(terms)[:5]
            print(f"  Sample: {sample_terms}")
        print()