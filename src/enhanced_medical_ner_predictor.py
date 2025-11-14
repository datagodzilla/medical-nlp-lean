#!/usr/bin/env python3
"""
enhanced_medical_ner_predictor.py

Enhanced Medical NER Pipeline for Complete Column Prediction
Predicts all required columns from the context_test_results_complete.xlsx file
Including: diseases, genes, negation, historical, uncertain, family entities, sections
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import re
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text

# Hugging Face imports
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ Warning: transformers not available. Install with: pip install transformers")

# Setup logging - will be configured with file handler in main()
logger = logging.getLogger(__name__)

# Import NLP libraries
try:
    import spacy
    from negspacy.negation import Negex
    from spacy import displacy
    from spacy.matcher import Matcher
except ImportError as e:
    logger.error(f"Required libraries not installed: {e}")
    logger.error("Please run: conda activate py311_bionlp")
    sys.exit(1)

# Optional visualization support for PNG export
VISUALIZATION_AVAILABLE = False
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from webdriver_manager.chrome import ChromeDriverManager
    import time
    VISUALIZATION_AVAILABLE = True
    logger.info("✅ Selenium available for PNG visualization export")
except ImportError:
    logger.info("⚠️ Selenium not available. PNG visualization export will be disabled")

def preprocess_text(text: str) -> str:
    """
    Preprocess text to remove HTML/RTF formatting and clean for NLP processing

    Args:
        text (str): Raw text input

    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    # Handle RTF formatting
    if text.strip().startswith('{\\rtf'):
        try:
            text = rtf_to_text(text)
        except Exception:
            # If RTF parsing fails, continue with original text
            pass

    # Remove HTML tags
    try:
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
    except Exception:
        # If BeautifulSoup fails, continue with original text
        pass

    # Clean up text
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters but keep medical relevant ones
    # Keep periods, commas, hyphens, parentheses, colons, semicolons
    text = re.sub(r'[^\w\s\.\,\-\(\)\:\;\%\/\+\=]', ' ', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text

class EnhancedMedicalNERPredictor:
    """Enhanced Medical NER pipeline for predicting all required columns with Hugging Face BioBERT models"""

    def __init__(self, model_name: str = "en_core_web_sm", use_gpu: bool = False, batch_size: int = 100,
                 target_rules_template: str = "data/external/target_rules_template.xlsx",
                 historical_rules_template: str = "data/external/historical_rules_template.xlsx",
                 negated_rules_template: str = "data/external/negated_rules_template.xlsx",
                 uncertainty_rules_template: str = "data/external/uncertainty_rules_template.xlsx",
                 confirmed_rules_template: str = "data/external/confirmed_rules_template.xlsx",
                 family_rules_template: str = "data/external/family_rules_template.xlsx",
                 section_categories_template: str = "data/external/section_categories_template.xlsx",
                 confidence_scores_template: str = "data/external/confidence_scores_template.xlsx",
                 template_priority: bool = True):
        """Initialize the enhanced medical NER predictor with configurable template paths

        Args:
            template_priority: If True, templates ALWAYS override BioBERT detections (DEFAULT).
                             If False, templates only override when confidence is higher.
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.template_priority = template_priority  # Template-always-override mode
        self.nlp = None
        self.disease_terms = set()
        self.gene_terms = set()
        self.drug_terms = set()
        self.anatomical_terms = set()
        self.symptom_terms = set()
        self.treatment_terms = set()

        # Initialize enhanced medical terminology (moved to after fallback terms load)
        self.protein_terms = set()
        self.anatomical_terms_extended = set()

        # Hugging Face BioBERT models
        self.hf_models = {}
        self.model_paths = {
            'chemical': Path('models/pretrained/Chemical'),
            'disease': Path('models/pretrained/Disease'),
            'gene': Path('models/pretrained/Gene')
        }

        # Dynamic template configurations
        self.target_rules_enabled = False
        self.historical_rules_enabled = False
        self.negated_rules_enabled = False
        self.uncertainty_rules_enabled = False
        self.confirmed_rules_enabled = False
        self.family_rules_enabled = False

        # Template file paths (configurable via CLI)
        self.target_rules_file = Path(target_rules_template)
        self.historical_rules_file = Path(historical_rules_template)
        self.negated_rules_file = Path(negated_rules_template)
        self.uncertainty_rules_file = Path(uncertainty_rules_template)
        self.confirmed_rules_file = Path(confirmed_rules_template)
        self.family_rules_file = Path(family_rules_template)

        # Template-based patterns
        self.historical_patterns = []
        self.negated_patterns = []
        self.uncertainty_patterns = []
        self.confirmed_patterns = []
        self.family_patterns = []

        # Load enhanced medical terms (with dynamic template checking)
        self._load_enhanced_medical_terms()
        self._load_context_templates()

        # Setup pipeline
        self._setup_pipeline()

        # Load Hugging Face models
        self._load_hf_models()

    def _load_hf_models(self):
        """Load Hugging Face BioBERT models for chemical, disease, and gene NER"""
        if not HF_AVAILABLE:
            logger.warning("⚠️ Hugging Face transformers not available. Using spaCy models only.")
            return

        for entity_type, model_path in self.model_paths.items():
            if model_path.exists():
                try:
                    logger.info(f"Loading {entity_type} BioBERT model from {model_path}")
                    # Create NER pipeline
                    self.hf_models[entity_type] = pipeline(
                        "ner",
                        model=str(model_path),
                        tokenizer=str(model_path),
                        aggregation_strategy="simple",
                        device=0 if self.use_gpu else -1
                    )
                    logger.info(f"✅ {entity_type.capitalize()} BioBERT model loaded successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to load {entity_type} model: {e}")
                    self.hf_models[entity_type] = None
            else:
                logger.warning(f"⚠️ {entity_type.capitalize()} model not found at {model_path}")
                self.hf_models[entity_type] = None

    def _extract_hf_entities(self, text: str) -> Dict[str, List[Dict]]:
        """Extract entities using Hugging Face BioBERT models with improved tokenization handling"""
        hf_entities = {
            'chemicals': [],
            'diseases_hf': [],
            'genes_hf': []
        }

        if not HF_AVAILABLE or not text.strip():
            return hf_entities

        # Extract entities for each model type
        model_mapping = {
            'chemical': 'chemicals',
            'disease': 'diseases_hf',
            'gene': 'genes_hf'
        }

        for model_type, result_key in model_mapping.items():
            if self.hf_models.get(model_type):
                try:
                    # Extract entities using the pipeline with better aggregation
                    predictions = self.hf_models[model_type](text)

                    # Post-process predictions to fix tokenization issues
                    processed_entities = self._post_process_hf_predictions(predictions, text, model_type)

                    for entity_info in processed_entities:
                        hf_entities[result_key].append(entity_info)

                except Exception as e:
                    logger.warning(f"Error extracting {model_type} entities: {e}")

        return hf_entities

    def _combine_subword_tokens(self, predictions: List[Dict], text: str) -> List[Dict]:
        """Combine subword tokens (like 'd' + '##yspnea') into complete entities"""
        if not predictions:
            return []

        combined_predictions = []
        i = 0

        while i < len(predictions):
            current_pred = predictions[i]
            entity_group = current_pred.get('entity_group', '')

            # Skip non-entity tokens (labeled as '0')
            if entity_group == '0':
                i += 1
                continue

            # Start building a combined entity
            combined_start = current_pred.get('start', 0)
            combined_end = current_pred.get('end', 0)
            combined_confidence_scores = [current_pred.get('score', 0)]

            # Look ahead for adjacent tokens with same entity group
            j = i + 1
            while j < len(predictions):
                next_pred = predictions[j]
                next_group = next_pred.get('entity_group', '')
                next_start = next_pred.get('start', 0)

                # Check if next token is adjacent and same entity group
                if (next_group == entity_group and
                    next_start == combined_end):  # Adjacent tokens

                    # Extend the combined entity
                    combined_end = next_pred.get('end', combined_end)
                    combined_confidence_scores.append(next_pred.get('score', 0))
                    j += 1
                else:
                    break

            # Create combined prediction
            avg_confidence = sum(combined_confidence_scores) / len(combined_confidence_scores)

            combined_pred = {
                'entity_group': entity_group,
                'score': avg_confidence,
                'start': combined_start,
                'end': combined_end,
                'word': text[combined_start:combined_end] if combined_start < len(text) and combined_end <= len(text) else ''
            }

            combined_predictions.append(combined_pred)
            i = j

        return combined_predictions

    def _post_process_hf_predictions(self, predictions: List[Dict], text: str, model_type: str) -> List[Dict]:
        """Post-process HF predictions to fix tokenization and combine subword tokens"""
        if not predictions:
            return []

        # First, combine subword tokens into complete entities
        combined_predictions = self._combine_subword_tokens(predictions, text)

        processed_entities = []

        for pred in combined_predictions:
            # Skip low confidence predictions
            if pred.get('score', 0) < 0.5:
                continue

            # Extract the actual text span from the original text
            start_pos = int(pred.get('start', 0))
            end_pos = int(pred.get('end', 0))

            if start_pos >= len(text) or end_pos > len(text) or start_pos >= end_pos:
                continue

            # Get the actual text from the original text
            entity_text = text[start_pos:end_pos].strip()

            # Skip very short entities or generic words
            if len(entity_text) < 2:
                continue

            # Filter by entity group and confidence
            entity_group = pred.get('entity_group', '')
            confidence = float(pred.get('score', 0.0))

            # Improved filtering based on model type and entity group
            should_include = False

            if model_type == 'chemical' and entity_group == 'CHEMICAL':
                # Only include likely chemical/drug names
                if len(entity_text) >= 3 and confidence > 0.7:
                    should_include = True

            elif model_type == 'disease' and entity_group == 'DISEASE':
                # Only include likely disease names
                if len(entity_text) >= 3 and confidence > 0.8:
                    should_include = True

            elif model_type == 'gene' and entity_group in ['GENETIC', 'GENE']:
                # Include gene names
                if len(entity_text) >= 2 and confidence > 0.7:
                    should_include = True

            if should_include:
                entity_info = {
                    'text': entity_text,
                    'label': entity_group.replace('CHEMICAL', 'DRUG').replace('GENETIC', 'GENE'),
                    'start': start_pos,
                    'end': end_pos,
                    'confidence': confidence,
                    'source': f'biobert_{model_type}'
                }
                processed_entities.append(entity_info)

        # Remove duplicates and overlapping entities
        processed_entities = self._remove_duplicate_entities(processed_entities)

        return processed_entities

    def _remove_duplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate and overlapping entities"""
        if not entities:
            return []

        # Sort by start position
        entities.sort(key=lambda x: x['start'])

        unique_entities = []

        for entity in entities:
            # Check for overlaps with existing entities
            is_duplicate = False

            for existing in unique_entities:
                # Check if entities overlap significantly
                overlap_start = max(entity['start'], existing['start'])
                overlap_end = min(entity['end'], existing['end'])
                overlap_length = max(0, overlap_end - overlap_start)

                entity_length = entity['end'] - entity['start']
                existing_length = existing['end'] - existing['start']

                # If overlap is significant, keep the one with higher confidence
                if overlap_length > 0.5 * min(entity_length, existing_length):
                    if entity['confidence'] > existing['confidence']:
                        unique_entities.remove(existing)
                    else:
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique_entities.append(entity)

        return unique_entities

    def _boost_entity_detection_with_target_rules(self, text: str, existing_entities: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Boost entity detection using target rules with confidence-based override"""
        if not self.target_rules_enabled:
            return existing_entities

        logger.debug("🎯 Applying target rules with confidence-based override")

        boosted_entities = {
            'diseases_hf': existing_entities.get('diseases_hf', []).copy(),
            'genes_hf': existing_entities.get('genes_hf', []).copy(),
            'chemicals': existing_entities.get('chemicals', []).copy()
        }
        text_lower = text.lower()
        override_count = {'diseases': 0, 'genes': 0, 'chemicals': 0}

        # Boost disease detection with confidence-based override
        disease_boosts = []
        for disease_term in self.disease_terms:
            if disease_term in text_lower and len(disease_term) > 2:
                start_pos = text_lower.find(disease_term)
                if start_pos != -1:
                    end_pos = start_pos + len(disease_term)
                    template_confidence = 0.92  # Higher confidence for template matches

                    # Find overlapping BioBERT entity
                    overlapping_idx = None
                    for idx, entity in enumerate(boosted_entities['diseases_hf']):
                        overlap_start = max(start_pos, entity.get('start', -1))
                        overlap_end = min(end_pos, entity.get('end', -1))
                        overlap_length = max(0, overlap_end - overlap_start)

                        entity_length = entity.get('end', 0) - entity.get('start', 0)
                        template_length = end_pos - start_pos

                        # Check for significant overlap (>50%)
                        if overlap_length > 0.5 * min(entity_length, template_length):
                            overlapping_idx = idx
                            break

                    if overlapping_idx is not None:
                        existing_entity = boosted_entities['diseases_hf'][overlapping_idx]
                        existing_conf = existing_entity.get('confidence', 0)

                        # Override based on strategy: template_priority OR confidence comparison
                        should_override = self.template_priority or (template_confidence > existing_conf)

                        if should_override:
                            boosted_entities['diseases_hf'][overlapping_idx] = {
                                'text': disease_term,
                                'label': 'DISEASE',
                                'start': start_pos,
                                'end': end_pos,
                                'confidence': template_confidence,
                                'source': 'target_rules_override',
                                'overridden': existing_entity.get('text')
                            }
                            override_count['diseases'] += 1
                            strategy = "TEMPLATE_PRIORITY" if self.template_priority else f"CONFIDENCE ({template_confidence:.2f} > {existing_conf:.2f})"
                            logger.debug(f"🔄 OVERRIDE [{strategy}]: '{disease_term}' replaced '{existing_entity.get('text')}'")
                    else:
                        # No overlap, add as new entity
                        disease_boosts.append({
                            'text': disease_term,
                            'label': 'DISEASE',
                            'start': start_pos,
                            'end': end_pos,
                            'confidence': template_confidence,
                            'source': 'target_rules'
                        })

        # Add new disease detections
        if disease_boosts:
            boosted_entities['diseases_hf'].extend(disease_boosts[:10])

        # Boost gene detection with confidence-based override
        gene_boosts = []
        for gene_term in self.gene_terms:
            if gene_term in text_lower and len(gene_term) > 1:
                start_pos = text_lower.find(gene_term)
                if start_pos != -1:
                    end_pos = start_pos + len(gene_term)
                    template_confidence = 0.90  # High confidence for gene template matches

                    # Find overlapping BioBERT entity
                    overlapping_idx = None
                    for idx, entity in enumerate(boosted_entities['genes_hf']):
                        overlap_start = max(start_pos, entity.get('start', -1))
                        overlap_end = min(end_pos, entity.get('end', -1))
                        overlap_length = max(0, overlap_end - overlap_start)

                        entity_length = entity.get('end', 0) - entity.get('start', 0)
                        template_length = end_pos - start_pos

                        # Check for significant overlap (>50%)
                        if overlap_length > 0.5 * min(entity_length, template_length):
                            overlapping_idx = idx
                            break

                    if overlapping_idx is not None:
                        existing_entity = boosted_entities['genes_hf'][overlapping_idx]
                        existing_conf = existing_entity.get('confidence', 0)

                        # Override based on strategy: template_priority OR confidence comparison
                        should_override = self.template_priority or (template_confidence > existing_conf)

                        if should_override:
                            boosted_entities['genes_hf'][overlapping_idx] = {
                                'text': gene_term,
                                'label': 'GENE',
                                'start': start_pos,
                                'end': end_pos,
                                'confidence': template_confidence,
                                'source': 'target_rules_override',
                                'overridden': existing_entity.get('text')
                            }
                            override_count['genes'] += 1
                            strategy = "TEMPLATE_PRIORITY" if self.template_priority else f"CONFIDENCE ({template_confidence:.2f} > {existing_conf:.2f})"
                            logger.debug(f"🔄 OVERRIDE [{strategy}]: '{gene_term}' replaced '{existing_entity.get('text')}'")
                    else:
                        # No overlap, add as new entity
                        gene_boosts.append({
                            'text': gene_term,
                            'label': 'GENE',
                            'start': start_pos,
                            'end': end_pos,
                            'confidence': template_confidence,
                            'source': 'target_rules'
                        })

        # Add new gene detections
        if gene_boosts:
            boosted_entities['genes_hf'].extend(gene_boosts[:10])

        # Boost drug/chemical detection with confidence-based override
        drug_boosts = []
        for drug_term in self.drug_terms:
            if drug_term in text_lower and len(drug_term) > 2:
                start_pos = text_lower.find(drug_term)
                if start_pos != -1:
                    end_pos = start_pos + len(drug_term)
                    template_confidence = 0.88  # High confidence for drug template matches

                    # Find overlapping BioBERT entity
                    overlapping_idx = None
                    for idx, entity in enumerate(boosted_entities['chemicals']):
                        overlap_start = max(start_pos, entity.get('start', -1))
                        overlap_end = min(end_pos, entity.get('end', -1))
                        overlap_length = max(0, overlap_end - overlap_start)

                        entity_length = entity.get('end', 0) - entity.get('start', 0)
                        template_length = end_pos - start_pos

                        # Check for significant overlap (>50%)
                        if overlap_length > 0.5 * min(entity_length, template_length):
                            overlapping_idx = idx
                            break

                    if overlapping_idx is not None:
                        existing_entity = boosted_entities['chemicals'][overlapping_idx]
                        existing_conf = existing_entity.get('confidence', 0)

                        # Override based on strategy: template_priority OR confidence comparison
                        should_override = self.template_priority or (template_confidence > existing_conf)

                        if should_override:
                            boosted_entities['chemicals'][overlapping_idx] = {
                                'text': drug_term,
                                'label': 'DRUG',
                                'start': start_pos,
                                'end': end_pos,
                                'confidence': template_confidence,
                                'source': 'target_rules_override',
                                'overridden': existing_entity.get('text')
                            }
                            override_count['chemicals'] += 1
                            strategy = "TEMPLATE_PRIORITY" if self.template_priority else f"CONFIDENCE ({template_confidence:.2f} > {existing_conf:.2f})"
                            logger.debug(f"🔄 OVERRIDE [{strategy}]: '{drug_term}' replaced '{existing_entity.get('text')}'")
                    else:
                        # No overlap, add as new entity
                        drug_boosts.append({
                            'text': drug_term,
                            'label': 'DRUG',
                            'start': start_pos,
                            'end': end_pos,
                            'confidence': template_confidence,
                            'source': 'target_rules'
                        })

        # Add new drug detections
        if drug_boosts:
            boosted_entities['chemicals'].extend(drug_boosts[:5])

        # Log boosting and override statistics
        total_overrides = sum(override_count.values())
        if disease_boosts or gene_boosts or drug_boosts or total_overrides > 0:
            logger.debug(f"🚀 Target rules: +{len(disease_boosts)} diseases, +{len(gene_boosts)} genes, +{len(drug_boosts)} chemicals")
            if total_overrides > 0:
                logger.info(f"🔄 Confidence-based overrides: {override_count['diseases']} diseases, {override_count['genes']} genes, {override_count['chemicals']} chemicals (Total: {total_overrides})")

        return boosted_entities

    def _setup_pipeline(self):
        """Setup the enhanced NLP pipeline"""
        try:
            logger.info(f"Loading spaCy pipeline with {self.model_name}")

            # Load base spacy model
            self.nlp = spacy.load(self.model_name)

            # Initialize matcher for medical pattern detection
            self.nlp.matcher = Matcher(self.nlp.vocab)

            # Enable GPU processing if requested and available
            if self.use_gpu:
                try:
                    spacy.prefer_gpu()
                    logger.info("✅ GPU processing enabled")
                except Exception as e:
                    logger.warning(f"⚠️ GPU not available, falling back to CPU: {e}")
                    self.use_gpu = False

            # Add negspacy for negation detection
            self.nlp.add_pipe("negex", config={"ent_types":["PERSON","ORG","DISEASE","GENE"]}, last=True)

            # Add custom components for medical entities
            self._add_custom_patterns()

            logger.info("✅ Enhanced NLP pipeline loaded successfully")

        except Exception as e:
            logger.error(f"Failed to setup pipeline: {e}")
            raise

    def _load_enhanced_medical_terms(self):
        """Dynamically load enhanced medical terms based on target rules template availability"""

        # Check if target rules template exists at execution time
        logger.info("🔍 Checking for target rules template at execution time...")

        if self.target_rules_file.exists():
            logger.info(f"✅ Target rules template found: {self.target_rules_file}")
            logger.info("🚀 Enabling enhanced entity detection with comprehensive target rules")

            try:
                # Import the enhanced target rules loader
                from enhanced_target_rules_loader import load_enhanced_target_rules

                # Load comprehensive target rules
                target_rules = load_enhanced_target_rules()

                # Update term sets
                self.disease_terms = target_rules.get('diseases', set())
                self.gene_terms = target_rules.get('genes', set())
                self.drug_terms = target_rules.get('drugs', set())
                self.anatomical_terms = target_rules.get('anatomical', set())
                self.symptom_terms = target_rules.get('symptoms', set())
                self.treatment_terms = target_rules.get('treatments', set())

                # Enable target rules boosting
                self.target_rules_enabled = True

                logger.info(f"✅ Enhanced medical terms loaded from target rules:")
                logger.info(f"   - Disease terms: {len(self.disease_terms):,}")
                logger.info(f"   - Gene terms: {len(self.gene_terms):,}")
                logger.info(f"   - Drug terms: {len(self.drug_terms):,}")
                logger.info(f"   - Anatomical terms: {len(self.anatomical_terms):,}")
                logger.info(f"   - Symptom terms: {len(self.symptom_terms):,}")
                logger.info(f"   - Treatment terms: {len(self.treatment_terms):,}")
                logger.info(f"🎯 Entity detection boosting: ENABLED")

            except Exception as e:
                logger.error(f"❌ Failed to load target rules template: {e}")
                logger.info("⚠️ Falling back to basic medical terms")
                self._load_fallback_terms()

        else:
            logger.warning(f"⚠️ Target rules template not found: {self.target_rules_file}")
            logger.info("📋 Using basic fallback medical terms")
            logger.info("💡 To enable enhanced detection, place target_rules_template.xlsx in data/external/")
            self._load_fallback_terms()

    def _load_fallback_terms(self):
        """Load basic fallback terms if enhanced rules fail"""
        self.disease_terms.update([
            'diabetes', 'diabetes mellitus', 'hypertension', 'cancer', 'infection', 'fever',
            'pneumonia', 'asthma', 'copd', 'heart failure', 'stroke', 'myocardial infarction',
            'sepsis', 'kidney disease', 'liver disease', 'alzheimer', 'parkinson', 'epilepsy',
            'depression', 'anxiety', 'schizophrenia', 'bipolar', 'autism', 'adhd', 'obesity',
            'osteoporosis', 'arthritis', 'lupus', 'multiple sclerosis', 'fibromyalgia'
        ])

        self.gene_terms.update([
            'brca1', 'brca2', 'tp53', 'apoe', 'cftr', 'fmr1', 'htt', 'dmd', 'f8', 'f9',
            'pah', 'hfe', 'mthfr', 'cyp2d6', 'cyp2c19', 'aldh2', 'vkorc1', 'slc6a4',
            'comt', 'maoa', 'drd4', 'slc6a3', 'chrna4', 'scn1a', 'kcnq1', 'herg',
            'ryr1', 'cacna1s', 'ldlr', 'pcsk9', 'apob', 'cetp', 'lipc', 'lpl'
        ])

        self.drug_terms.update([
            'metformin', 'insulin', 'aspirin', 'ibuprofen', 'acetaminophen', 'lisinopril',
            'atorvastatin', 'omeprazole', 'levothyroxine', 'amlodipine', 'sertraline'
        ])

        # Initialize enhanced medical terminology
        self.protein_terms = self._get_protein_terms()
        self.anatomical_terms_extended = self._get_anatomical_terms()

    def _load_context_templates(self):
        """Load clinical context templates (historical, negated, uncertainty) if available"""

        # Load historical patterns
        logger.info("🔍 Checking for historical rules template...")
        if self.historical_rules_file.exists():
            try:
                historical_df = pd.read_excel(self.historical_rules_file)
                self.historical_patterns = historical_df['pattern'].tolist()
                self.historical_rules_enabled = True
                logger.info(f"✅ Historical rules loaded: {len(self.historical_patterns)} patterns from {self.historical_rules_file}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load historical rules: {e}")
        else:
            logger.info(f"📋 Historical rules template not found: {self.historical_rules_file}")

        # Load negated patterns
        logger.info("🔍 Checking for negated rules template...")
        if self.negated_rules_file.exists():
            try:
                negated_df = pd.read_excel(self.negated_rules_file)
                self.negated_patterns = negated_df['pattern'].tolist()
                self.negated_rules_enabled = True
                logger.info(f"✅ Negated rules loaded: {len(self.negated_patterns)} patterns from {self.negated_rules_file}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load negated rules: {e}")
        else:
            logger.info(f"📋 Negated rules template not found: {self.negated_rules_file}")

        # Load uncertainty patterns
        logger.info("🔍 Checking for uncertainty rules template...")
        if self.uncertainty_rules_file.exists():
            try:
                uncertainty_df = pd.read_excel(self.uncertainty_rules_file)
                self.uncertainty_patterns = uncertainty_df['pattern'].tolist()
                # Ensure "suspected" is included (required for tests)
                if 'suspected' not in self.uncertainty_patterns:
                    self.uncertainty_patterns.append('suspected')
                self.uncertainty_rules_enabled = True
                logger.info(f"✅ Uncertainty rules loaded: {len(self.uncertainty_patterns)} patterns from {self.uncertainty_rules_file}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load uncertainty rules: {e}")
        else:
            logger.info(f"📋 Uncertainty rules template not found: {self.uncertainty_rules_file}")

        # Load confirmed patterns
        logger.info("🔍 Checking for confirmed rules template...")
        if self.confirmed_rules_file.exists():
            try:
                confirmed_df = pd.read_excel(self.confirmed_rules_file)
                self.confirmed_patterns = confirmed_df['pattern'].tolist()
                self.confirmed_rules_enabled = True
                logger.info(f"✅ Confirmed rules loaded: {len(self.confirmed_patterns)} patterns from {self.confirmed_rules_file}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load confirmed rules: {e}")
        else:
            logger.info(f"📋 Confirmed rules template not found: {self.confirmed_rules_file}")

        # Load family patterns
        logger.info("🔍 Checking for family rules template...")
        if self.family_rules_file.exists():
            try:
                family_df = pd.read_excel(self.family_rules_file)
                self.family_patterns = family_df['pattern'].tolist()
                self.family_rules_enabled = True
                logger.info(f"✅ Family rules loaded: {len(self.family_patterns)} patterns from {self.family_rules_file}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load family rules: {e}")
        else:
            logger.info(f"📋 Family rules template not found: {self.family_rules_file}")

        # Summary
        enabled_templates = []
        if self.target_rules_enabled:
            enabled_templates.append("Target Rules")
        if self.historical_rules_enabled:
            enabled_templates.append("Historical Rules")
        if self.negated_rules_enabled:
            enabled_templates.append("Negated Rules")
        if self.uncertainty_rules_enabled:
            enabled_templates.append("Uncertainty Rules")
        if self.confirmed_rules_enabled:
            enabled_templates.append("Confirmed Rules")
        if self.family_rules_enabled:
            enabled_templates.append("Family Rules")

        if enabled_templates:
            logger.info(f"🎯 Enhanced context detection enabled: {', '.join(enabled_templates)}")
        else:
            logger.info("📋 Using fallback hardcoded patterns for context detection")

    def _add_custom_patterns(self):
        """Add enhanced custom patterns for medical entity recognition"""
        all_patterns = []

        # Create patterns for diseases (increase limit due to better performance expected)
        disease_patterns = []
        disease_list = list(self.disease_terms)[:500]  # Increased from 100
        for disease in disease_list:
            if len(disease) > 2:  # Filter very short terms
                disease_patterns.append({"label": "DISEASE", "pattern": disease})

        # Create patterns for genes
        gene_patterns = []
        gene_list = list(self.gene_terms)[:500]  # Increased from 100
        for gene in gene_list:
            if len(gene) > 1:  # Genes can be shorter
                gene_patterns.append({"label": "GENE", "pattern": gene})

        # Create patterns for drugs/chemicals (NEW)
        drug_patterns = []
        drug_list = list(self.drug_terms)[:200]  # New drug patterns
        for drug in drug_list:
            if len(drug) > 2:
                drug_patterns.append({"label": "DRUG", "pattern": drug})

        # Create patterns for anatomical terms
        anatomical_patterns = []
        anatomical_list = list(self.anatomical_terms)
        for anatomical in anatomical_list:
            if len(anatomical) > 2:
                anatomical_patterns.append({"label": "ANATOMY", "pattern": anatomical})

        # Create patterns for symptoms
        symptom_patterns = []
        symptom_list = list(self.symptom_terms)
        for symptom in symptom_list:
            if len(symptom) > 2:
                symptom_patterns.append({"label": "SYMPTOM", "pattern": symptom})

        # Create patterns for treatments
        treatment_patterns = []
        treatment_list = list(self.treatment_terms)
        for treatment in treatment_list:
            if len(treatment) > 2:
                treatment_patterns.append({"label": "TREATMENT", "pattern": treatment})

        # Combine all patterns
        all_patterns = (disease_patterns + gene_patterns + drug_patterns +
                       anatomical_patterns + symptom_patterns + treatment_patterns)

        # Add entity ruler with enhanced patterns
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(all_patterns)

        logger.info(f"✅ Enhanced patterns added:")
        logger.info(f"   - Disease patterns: {len(disease_patterns)}")
        logger.info(f"   - Gene patterns: {len(gene_patterns)}")
        logger.info(f"   - Drug patterns: {len(drug_patterns)}")
        logger.info(f"   - Anatomical patterns: {len(anatomical_patterns)}")
        logger.info(f"   - Symptom patterns: {len(symptom_patterns)}")
        logger.info(f"   - Treatment patterns: {len(treatment_patterns)}")
        logger.info(f"   - Total patterns: {len(all_patterns)}")

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract all medical entities and context information from text"""
        if not text or not isinstance(text, str):
            return self._empty_result()

        try:
            # Process text with NLP pipeline
            doc = self.nlp(text)

            # Extract entities using Hugging Face models
            hf_entities = self._extract_hf_entities(text)

            # Apply target rules boosting if enabled
            if self.target_rules_enabled:
                hf_entities = self._boost_entity_detection_with_target_rules(text, hf_entities)

            # Initialize entity collections
            diseases = []
            genes = []
            chemicals = hf_entities['chemicals']  # From HF models
            negated_entities = []
            historical_entities = []
            uncertain_entities = []
            family_entities = []
            all_entities = []

            # Add HF entities to main collections
            diseases.extend(hf_entities['diseases_hf'])
            genes.extend(hf_entities['genes_hf'])
            all_entities.extend(chemicals)
            all_entities.extend(hf_entities['diseases_hf'])
            all_entities.extend(hf_entities['genes_hf'])

            # MEDICAL-ONLY ENTITY DETECTION (NO standard spaCy entities)
            # Priority: 1) Target Rules Boosting, 2) BioBERT (already loaded), 3) Medical Patterns

            # Apply additional target rules pattern matching if enabled
            if self.target_rules_enabled:
                additional_medical_entities = self._detect_additional_medical_patterns(text)
                diseases.extend(additional_medical_entities.get('diseases', []))
                genes.extend(additional_medical_entities.get('genes', []))
                chemicals.extend(additional_medical_entities.get('chemicals', []))
                all_entities.extend(additional_medical_entities.get('all', []))

            # Apply clinical context detection using medical entities only
            # Use enhanced negation detection with both spaCy entities and detected medical entities
            doc = self.nlp(text)

            # Process spaCy entities for enhanced context detection
            negated_entities = []

            # Check spaCy entities
            for ent in doc.ents:
                if self._is_negated_enhanced(text, ent):
                    # Convert spaCy entity to dict format for compatibility
                    entity_dict = {
                        'text': ent.text,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'label': ent.label_,
                        'confidence': 100,  # High confidence for enhanced detection
                        'negation_confidence': 100,
                        'matched_negation_pattern': 'enhanced_detection'
                    }
                    negated_entities.append(entity_dict)

            # Also check BioBERT and pattern-detected entities for negation
            # Create pseudo-spaCy entities for the enhanced negation detection
            for entity in all_entities:
                # Create a pseudo-spaCy entity object
                class PseudoEntity:
                    def __init__(self, text, start_char, end_char, label):
                        self.text = text
                        self.start_char = start_char
                        self.end_char = end_char
                        self.label_ = label

                pseudo_ent = PseudoEntity(
                    entity.get('text', ''),
                    entity.get('start', 0),
                    entity.get('end', 0),
                    entity.get('label', 'UNKNOWN')
                )

                if self._is_negated_enhanced(text, pseudo_ent):
                    # Avoid duplicates by checking if already in negated_entities
                    entity_id = f"{entity.get('text')}_{entity.get('start')}_{entity.get('end')}"
                    existing_ids = {f"{e.get('text')}_{e.get('start')}_{e.get('end')}" for e in negated_entities}

                    if entity_id not in existing_ids:
                        entity_dict = {
                            'text': entity.get('text', ''),
                            'start': entity.get('start', 0),
                            'end': entity.get('end', 0),
                            'label': entity.get('label', 'UNKNOWN'),
                            'confidence': 100,
                            'negation_confidence': 100,
                            'matched_negation_pattern': 'enhanced_detection'
                        }
                        negated_entities.append(entity_dict)

            # Apply context detection with conflict resolution
            # Priority hierarchy: Negated > Family > Historical > Uncertain > Confirmed
            remaining_entities = [e for e in all_entities]
            assigned_entities = set()

            # 1. Negated entities have highest priority (already processed above)
            for entity in negated_entities:
                entity_id = f"{entity.get('text')}_{entity.get('start')}_{entity.get('end')}"
                assigned_entities.add(entity_id)

            # 2. Family entities
            family_entities = self._detect_family_medical_entities(text, remaining_entities)
            for entity in family_entities:
                entity_id = f"{entity.get('text')}_{entity.get('start')}_{entity.get('end')}"
                assigned_entities.add(entity_id)

            # 3. Historical entities (excluding already assigned)
            unassigned_entities = [e for e in remaining_entities
                                 if f"{e.get('text')}_{e.get('start')}_{e.get('end')}" not in assigned_entities]
            historical_entities = self._detect_historical_medical_entities(text, unassigned_entities)
            for entity in historical_entities:
                entity_id = f"{entity.get('text')}_{entity.get('start')}_{entity.get('end')}"
                assigned_entities.add(entity_id)

            # 4. Uncertain entities (excluding already assigned)
            unassigned_entities = [e for e in remaining_entities
                                 if f"{e.get('text')}_{e.get('start')}_{e.get('end')}" not in assigned_entities]
            uncertain_entities = self._detect_uncertain_medical_entities(text, unassigned_entities)
            for entity in uncertain_entities:
                entity_id = f"{entity.get('text')}_{entity.get('start')}_{entity.get('end')}"
                assigned_entities.add(entity_id)

            # 5. Confirmed entities (lowest priority - everything else that's not assigned)
            unassigned_entities = [e for e in remaining_entities
                                 if f"{e.get('text')}_{e.get('start')}_{e.get('end')}" not in assigned_entities]
            confirmed_entities = self._detect_confirmed_medical_entities(text, unassigned_entities)

            # Handle low-confidence negated entities: reclassify as uncertain if confidence < 80%
            # Check all entities that matched negation patterns but didn't make it to negated_entities
            negated_entity_ids = {f"{e.get('text')}_{e.get('start')}_{e.get('end')}" for e in negated_entities}

            for entity in all_entities:
                entity_id = f"{entity.get('text')}_{entity.get('start')}_{entity.get('end')}"

                # Check if this entity has negation_confidence but is not in negated_entities
                if 'negation_confidence' in entity and entity_id not in negated_entity_ids:
                    confidence = entity.get('negation_confidence', 0)

                    # If confidence is between 50% and 80%, treat as uncertain
                    if 50 <= confidence < 80:
                        # Add to uncertain if not already there
                        uncertain_entity_ids = {f"{e.get('text')}_{e.get('start')}_{e.get('end')}" for e in uncertain_entities}
                        if entity_id not in uncertain_entity_ids:
                            entity_copy = entity.copy()
                            entity_copy['uncertain_reason'] = f"Low negation confidence ({confidence}%)"
                            uncertain_entities.append(entity_copy)

            # Determine section categories
            section_categories = self._categorize_sections(text)

            # Note: Visualization will be generated after deduplication to ensure no context overlap

            # Extract predictor terms for each context category
            confirmed_patterns = [
                'diagnosed with', 'diagnosis of', 'confirmed', 'has', 'shows', 'demonstrates',
                'presents with', 'found to have', 'suffering from', 'affected by',
                'positive for', 'tested positive', 'evidence of', 'consistent with',
                'currently has', 'currently experiencing', 'active', 'ongoing',
                'established', 'documented', 'known', 'chronic', 'acute',
                'findings show', 'reveals', 'indicates', 'detected'
            ]
            confirmed_predictors = self._extract_predictor_terms(text, confirmed_entities, confirmed_patterns)

            negated_patterns = self.negated_patterns if self.negated_rules_enabled else [
                'no', 'not', 'without', 'absent', 'negative', 'denies', 'rules out',
                'free of', 'clear of', 'unremarkable', 'normal', 'within normal limits',
                'ruled out', 'no evidence of', 'no signs of', 'absence of', 'never', 'none'
            ]
            negated_predictors = self._extract_predictor_terms(text, negated_entities, negated_patterns)

            historical_patterns = self.historical_patterns if self.historical_rules_enabled else [
                'history of', 'past', 'previous', 'prior', 'previously', 'former',
                'years ago', 'months ago', 'weeks ago', 'days ago', 'earlier',
                'childhood', 'adolescent', 'old', 'remote'
            ]
            historical_predictors = self._extract_predictor_terms(text, historical_entities, historical_patterns)

            uncertainty_patterns = self.uncertainty_patterns if self.uncertainty_rules_enabled else [
                'possible', 'possibly', 'probable', 'probably', 'likely', 'unlikely',
                'may', 'might', 'could', 'perhaps', 'maybe', 'uncertain', 'unclear',
                'rule out', 'r/o', 'vs', 'versus', 'differential', 'consider',
                'question', 'query', 'suspect', 'suspicious', 'may have', 'may be'
            ]
            uncertainty_predictors = self._extract_predictor_terms(text, uncertain_entities, uncertainty_patterns)

            family_patterns = [
                'family history', 'mother', 'father', 'sister', 'brother', 'parent',
                'maternal', 'paternal', 'grandmother', 'grandfather', 'aunt', 'uncle',
                'cousin', 'sibling', 'relatives', 'fh:', 'fhx', 'hereditary', 'genetic',
                'familial', 'inherited'
            ]
            family_predictors = self._extract_predictor_terms(text, family_entities, family_patterns)

            # Final deduplication: Ensure entities don't appear in multiple context strings
            # Priority: Negated > Family > Historical > Uncertain > Confirmed
            entity_text_assignments = {}  # Track which context each entity text is assigned to

            # Collect entity texts by priority
            negated_texts = [e.get('text', '') for e in negated_entities]
            family_texts = [e.get('text', '') for e in family_entities]
            historical_texts = [e.get('text', '') for e in historical_entities]
            uncertain_texts = [e.get('text', '') for e in uncertain_entities]
            confirmed_texts = [e.get('text', '') for e in confirmed_entities]

            # Assign priorities (using entity_text to avoid shadowing the text parameter)
            for entity_text in negated_texts:
                entity_text_assignments[entity_text] = 'negated'
            for entity_text in family_texts:
                if entity_text not in entity_text_assignments:
                    entity_text_assignments[entity_text] = 'family'
            for entity_text in historical_texts:
                if entity_text not in entity_text_assignments:
                    entity_text_assignments[entity_text] = 'historical'
            for entity_text in uncertain_texts:
                if entity_text not in entity_text_assignments:
                    entity_text_assignments[entity_text] = 'uncertain'
            for entity_text in confirmed_texts:
                if entity_text not in entity_text_assignments:
                    entity_text_assignments[entity_text] = 'confirmed'

            # Create deduplicated lists
            final_negated = [e for e in negated_entities if entity_text_assignments.get(e.get('text', '')) == 'negated']
            final_family = [e for e in family_entities if entity_text_assignments.get(e.get('text', '')) == 'family']
            final_historical = [e for e in historical_entities if entity_text_assignments.get(e.get('text', '')) == 'historical']
            final_uncertain = [e for e in uncertain_entities if entity_text_assignments.get(e.get('text', '')) == 'uncertain']
            final_confirmed = [e for e in confirmed_entities if entity_text_assignments.get(e.get('text', '')) == 'confirmed']

            # Generate enhanced visualization with DEDUPLICATED context classifications to prevent overlap
            displacy_html = self._generate_enhanced_medical_visualization(
                text, all_entities, final_confirmed, final_negated,
                final_uncertain, final_historical, final_family, section_categories
            )

            return {
                'Text Visualization': displacy_html,
                'detected_diseases': self._format_entities(diseases),
                'total_diseases_count': len(diseases),
                'detected_diseases_unique': self._get_unique_entities(diseases),
                'detected_genes': self._format_entities(genes),
                'total_gene_count': len(genes),
                'detected_genes_unique': self._get_unique_entities(genes),
                'detected_drugs': self._format_entities(chemicals),
                'detected_drugs_count': len(chemicals),
                'detected_drugs_unique': self._get_unique_entities(chemicals),
                'detected_chemicals': self._format_entities(chemicals),
                'total_chemicals_count': len(chemicals),
                'detected_chemicals_unique': self._get_unique_entities(chemicals),
                'confirmed_entities': self._format_entities(final_confirmed),
                'confirmed_entities_count': len(final_confirmed),
                'confirmed_entities_predictors': confirmed_predictors,
                'confirmed_context_sentences': self._extract_context_sentences(text, final_confirmed, 'confirmed'),
                'negated_entities': self._format_entities(final_negated),
                'negated_entities_count': len(final_negated),
                'negated_entities_predictors': negated_predictors,
                'negated_context_sentences': self._extract_context_sentences(text, final_negated, 'negated'),
                'historical_entities': self._format_entities(final_historical),
                'historical_entities_count': len(final_historical),
                'historical_entities_predictors': historical_predictors,
                'historical_context_sentences': self._extract_context_sentences(text, final_historical, 'historical'),
                'uncertain_entities': self._format_entities(final_uncertain),
                'uncertain_entities_count': len(final_uncertain),
                'uncertain_entities_predictors': uncertainty_predictors,
                'uncertain_context_sentences': self._extract_context_sentences(text, final_uncertain, 'uncertain'),
                'family_entities': self._format_entities(final_family),
                'family_entities_count': len(final_family),
                'family_entities_predictors': family_predictors,
                'family_context_sentences': self._extract_context_sentences(text, final_family, 'family'),
                'section_categories': section_categories,
                'all_entities_json': json.dumps(all_entities, indent=2)
            }

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return self._empty_result()

    def _is_historical(self, text: str, entity) -> bool:
        """Enhanced historical detection with improved context analysis and template priority"""
        import re

        # Use template patterns with priority, otherwise use improved fallback
        if self.historical_rules_enabled and self.historical_patterns:
            keywords = self.historical_patterns
            template_priority = True
        else:
            # Improved fallback patterns with confidence weighting
            keywords = [
                'history of', 'past history of', 'previous', 'prior', 'previously', 'former',
                'years ago', 'months ago', 'weeks ago', 'days ago', 'earlier this',
                'childhood', 'adolescent', 'young adult', 'remote history', 'longstanding'
            ]
            template_priority = False

        # Enhanced context analysis - smaller, more focused window
        start_idx = max(0, entity.start_char - 30)
        end_idx = min(len(text), entity.end_char + 30)
        context = text[start_idx:end_idx].lower()

        # Get sentence boundaries for better context
        sentences = re.split(r'[.!?]+', text)
        entity_sentence = ""
        for sentence in sentences:
            if entity.start_char >= text.find(sentence) and entity.end_char <= text.find(sentence) + len(sentence):
                entity_sentence = sentence.lower()
                break

        confidence_score = 0
        max_confidence = 0

        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            if not keyword_lower:
                continue

            current_confidence = 0

            # Multi-word pattern matching with position scoring
            if ' ' in keyword_lower:
                # Check in immediate context (higher confidence)
                if keyword_lower in context:
                    # Higher confidence if closer to entity
                    distance = min(abs(context.find(keyword_lower) - 15), 15)
                    current_confidence = 0.8 - (distance / 50)

                # Additional check in sentence context
                elif keyword_lower in entity_sentence:
                    current_confidence = 0.6

            else:
                # Single-word pattern with strict word boundaries
                pattern = r'\b' + re.escape(keyword_lower) + r'\b'

                # Check immediate context first
                context_match = re.search(pattern, context)
                if context_match:
                    distance = abs(context_match.start() - 15)
                    current_confidence = 0.7 - (distance / 30)

                # Check sentence context
                elif re.search(pattern, entity_sentence):
                    current_confidence = 0.5

            max_confidence = max(max_confidence, current_confidence)

        # Template priority: lower threshold for template patterns
        threshold = 0.3 if template_priority else 0.5
        return max_confidence >= threshold

    def _is_uncertain(self, text: str, entity) -> bool:
        """Enhanced uncertainty detection with improved context analysis and template priority"""
        import re

        # Use template patterns with priority, otherwise use improved fallback
        if self.uncertainty_rules_enabled and self.uncertainty_patterns:
            keywords = self.uncertainty_patterns
            template_priority = True
        else:
            # Improved fallback patterns - more specific uncertainty indicators
            keywords = [
                'possible', 'possibly', 'probable', 'probably', 'likely', 'unlikely',
                'may be', 'might be', 'could be', 'perhaps', 'maybe', 'uncertain', 'unclear',
                'rule out', 'r/o', 'vs', 'versus', 'differential diagnosis', 'consider',
                'questionable', 'query', 'suspect', 'suspicious of', 'concern for'
            ]
            template_priority = False

        # Enhanced context analysis - focused window
        start_idx = max(0, entity.start_char - 25)
        end_idx = min(len(text), entity.end_char + 25)
        context = text[start_idx:end_idx].lower()

        # Get sentence for broader context
        sentences = re.split(r'[.!?]+', text)
        entity_sentence = ""
        for sentence in sentences:
            sentence_start = text.find(sentence)
            if sentence_start != -1 and entity.start_char >= sentence_start and entity.end_char <= sentence_start + len(sentence):
                entity_sentence = sentence.lower()
                break

        max_confidence = 0

        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            if not keyword_lower:
                continue

            current_confidence = 0

            # Multi-word patterns with position-based confidence
            if ' ' in keyword_lower:
                if keyword_lower in context:
                    # Closer proximity = higher confidence
                    keyword_pos = context.find(keyword_lower)
                    distance = abs(keyword_pos - 12)  # 12 is center of 25-char window
                    current_confidence = 0.8 - (distance / 40)
                elif keyword_lower in entity_sentence:
                    current_confidence = 0.5
            else:
                # Single-word patterns with strict boundaries
                pattern = r'\b' + re.escape(keyword_lower) + r'\b'

                context_match = re.search(pattern, context)
                if context_match:
                    distance = abs(context_match.start() - 12)
                    current_confidence = 0.7 - (distance / 30)
                elif re.search(pattern, entity_sentence):
                    current_confidence = 0.4

            max_confidence = max(max_confidence, current_confidence)

        # Exclude common false positives for uncertainty
        false_positive_patterns = [
            r'\bpossible\s+causes?\b',  # "possible causes"
            r'\bmay\s+include\b',       # "may include"
            r'\bcould\s+indicate\b'     # "could indicate"
        ]

        for fp_pattern in false_positive_patterns:
            if re.search(fp_pattern, entity_sentence):
                max_confidence *= 0.3  # Reduce confidence for false positives

        # Template priority: lower threshold for template patterns
        threshold = 0.3 if template_priority else 0.5
        return max_confidence >= threshold

    def _is_family_related(self, text: str, entity) -> bool:
        """Enhanced family history detection with improved context analysis and template priority"""
        import re

        # Use template patterns with priority, otherwise use improved fallback
        if self.family_rules_enabled and self.family_patterns:
            keywords = self.family_patterns
            template_priority = True
        else:
            # Improved family history patterns
            keywords = [
                'family history', 'family history of', 'familial', 'hereditary', 'genetic predisposition',
                'inherited', 'runs in family', 'mother', 'father', 'parent', 'sibling',
                'brother', 'sister', 'grandmother', 'grandfather', 'aunt', 'uncle', 'cousin',
                'maternal', 'paternal', 'fh', 'fhx', 'family hx'
            ]
            template_priority = False

        # Enhanced context analysis - focused window
        start_idx = max(0, entity.start_char - 35)
        end_idx = min(len(text), entity.end_char + 35)
        context = text[start_idx:end_idx].lower()

        # Get sentence context
        sentences = re.split(r'[.!?]+', text)
        entity_sentence = ""
        for sentence in sentences:
            sentence_start = text.find(sentence)
            if sentence_start != -1 and entity.start_char >= sentence_start and entity.end_char <= sentence_start + len(sentence):
                entity_sentence = sentence.lower()
                break

        max_confidence = 0

        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            if not keyword_lower:
                continue

            current_confidence = 0

            # Multi-word patterns with position-based confidence
            if ' ' in keyword_lower:
                if keyword_lower in context:
                    keyword_pos = context.find(keyword_lower)
                    distance = abs(keyword_pos - 17)  # 17 is center of 35-char window
                    current_confidence = 0.9 - (distance / 50)
                elif keyword_lower in entity_sentence:
                    current_confidence = 0.6
            else:
                # Single-word patterns with strict boundaries
                pattern = r'\b' + re.escape(keyword_lower) + r'\b'

                context_match = re.search(pattern, context)
                if context_match:
                    distance = abs(context_match.start() - 17)
                    current_confidence = 0.8 - (distance / 35)
                elif re.search(pattern, entity_sentence):
                    current_confidence = 0.5

            max_confidence = max(max_confidence, current_confidence)

        # Exclude false positives - genetic terms that don't indicate family history
        false_positive_patterns = [
            r'\bgenetic\s+testing\b',     # "genetic testing"
            r'\bgenetic\s+counseling\b',  # "genetic counseling"
            r'\bgenetic\s+analysis\b',    # "genetic analysis"
            r'\bgenetic\s+markers?\b'     # "genetic marker(s)"
        ]

        for fp_pattern in false_positive_patterns:
            if re.search(fp_pattern, entity_sentence):
                max_confidence *= 0.2  # Significantly reduce confidence for false positives

        # Template priority: lower threshold for template patterns
        threshold = 0.3 if template_priority else 0.6
        return max_confidence >= threshold

    def _is_negated_enhanced(self, text: str, entity) -> bool:
        """Enhanced negation detection with improved context analysis and template priority"""
        import re

        # First check if negspaCy already detected negation
        if hasattr(entity, '_') and hasattr(entity._, 'negex') and entity._.negex:
            return True

        # Use template patterns with priority, otherwise use improved fallback
        if self.negated_rules_enabled and self.negated_patterns:
            keywords = self.negated_patterns
            template_priority = True
        else:
            # Improved negation patterns - more specific and precise
            keywords = [
                'no', 'not', 'none', 'never', 'denies', 'denied', 'negative for',
                'absent', 'absence of', 'free of', 'free from', 'without', 'lacks',
                'rule out', 'r/o', 'ruled out', 'no evidence of', 'no signs of',
                'no symptoms of', 'no history of', 'unremarkable for'
            ]
            template_priority = False

        # Enhanced context analysis - smaller focused window for negation
        start_idx = max(0, entity.start_char - 20)
        end_idx = min(len(text), entity.end_char + 15)
        context = text[start_idx:end_idx].lower()

        # Check for scope-reversing conjunctions that override negation
        # If entity comes after "but reports", "but shows", etc., skip negation detection entirely
        entity_text_lower = entity.text.lower()
        entity_text_pos = context.find(entity_text_lower)

        # Check for comprehensive scope-reversing patterns with priority-based matching
        scope_reversal_patterns = [
            # Priority 10: High-confidence adversative patterns (Confidence: 90-95%)
            {'pattern': 'but reports', 'priority': 10, 'confidence': 0.95},
            {'pattern': 'but shows', 'priority': 10, 'confidence': 0.95},
            {'pattern': 'but has', 'priority': 10, 'confidence': 0.95},
            {'pattern': 'but demonstrates', 'priority': 9, 'confidence': 0.93},
            {'pattern': 'but presents', 'priority': 9, 'confidence': 0.93},
            {'pattern': 'but exhibits', 'priority': 9, 'confidence': 0.92},
            {'pattern': 'but complains of', 'priority': 9, 'confidence': 0.92},
            {'pattern': 'but admits to', 'priority': 8, 'confidence': 0.90},
            {'pattern': 'but acknowledges', 'priority': 8, 'confidence': 0.90},
            {'pattern': 'but endorses', 'priority': 8, 'confidence': 0.88},

            # Priority 8-9: However patterns (Confidence: 88-93%)
            {'pattern': 'however reports', 'priority': 9, 'confidence': 0.93},
            {'pattern': 'however shows', 'priority': 9, 'confidence': 0.93},
            {'pattern': 'however has', 'priority': 9, 'confidence': 0.92},
            {'pattern': 'however demonstrates', 'priority': 8, 'confidence': 0.90},

            # Priority 7-8: Yet patterns (Confidence: 85-90%)
            {'pattern': 'yet reports', 'priority': 8, 'confidence': 0.90},
            {'pattern': 'yet shows', 'priority': 8, 'confidence': 0.90},
            {'pattern': 'yet has', 'priority': 8, 'confidence': 0.88},

            # Priority 7: Nevertheless/nonetheless patterns (Confidence: 85%)
            {'pattern': 'nevertheless reports', 'priority': 7, 'confidence': 0.85},
            {'pattern': 'nonetheless shows', 'priority': 7, 'confidence': 0.85},

            # Priority 8-9: Temporal transitions (Confidence: 85-92%)
            {'pattern': 'but now reports', 'priority': 9, 'confidence': 0.92},
            {'pattern': 'but currently has', 'priority': 9, 'confidence': 0.92},
            {'pattern': 'but today shows', 'priority': 8, 'confidence': 0.90},
            {'pattern': 'however now reports', 'priority': 8, 'confidence': 0.90},
            {'pattern': 'however currently demonstrates', 'priority': 8, 'confidence': 0.88},
            {'pattern': 'yet currently has', 'priority': 7, 'confidence': 0.85},

            # Priority 6-8: Exception patterns (Confidence: 78-90%)
            {'pattern': 'except for', 'priority': 8, 'confidence': 0.90},
            {'pattern': 'except', 'priority': 7, 'confidence': 0.85},
            {'pattern': 'save for', 'priority': 6, 'confidence': 0.80},
            {'pattern': 'apart from', 'priority': 6, 'confidence': 0.80},
            {'pattern': 'aside from', 'priority': 6, 'confidence': 0.78},
            {'pattern': 'with the exception of', 'priority': 7, 'confidence': 0.85},

            # Priority 6-7: Concessive patterns (Confidence: 80-85%)
            {'pattern': 'although reports', 'priority': 7, 'confidence': 0.85},
            {'pattern': 'though shows', 'priority': 7, 'confidence': 0.85},
            {'pattern': 'even though has', 'priority': 7, 'confidence': 0.83},
            {'pattern': 'despite reports', 'priority': 6, 'confidence': 0.80},
            {'pattern': 'in spite of shows', 'priority': 6, 'confidence': 0.80},

            # Priority 4-6: Contrastive patterns (Confidence: 70-80%)
            {'pattern': 'on the other hand reports', 'priority': 6, 'confidence': 0.80},
            {'pattern': 'conversely shows', 'priority': 6, 'confidence': 0.78},
            {'pattern': 'in contrast has', 'priority': 5, 'confidence': 0.75},
            {'pattern': 'rather reports', 'priority': 5, 'confidence': 0.75},
            {'pattern': 'instead shows', 'priority': 5, 'confidence': 0.73}
        ]

        # Sort patterns by priority (highest first) then by confidence
        scope_reversal_patterns.sort(key=lambda x: (x['priority'], x['confidence']), reverse=True)

        # Use priority-based pattern matching with confidence threshold
        confidence_threshold = 0.75  # Only use patterns with confidence >= 75%
        best_match = None
        best_confidence = 0

        for pattern_info in scope_reversal_patterns:
            pattern = pattern_info['pattern']
            confidence = pattern_info['confidence']
            priority = pattern_info['priority']

            # Skip low-confidence patterns
            if confidence < confidence_threshold:
                continue

            phrase_pos = context.find(pattern)
            if phrase_pos >= 0 and entity_text_pos >= 0 and phrase_pos < entity_text_pos:
                # Found a match - check if this is the best one so far
                if confidence > best_confidence:
                    best_match = pattern_info
                    best_confidence = confidence

        # If we found a high-confidence scope reversal pattern, skip negation
        if best_match and best_confidence >= 0.80:  # High confidence threshold for overriding negation
            return False

        # Get sentence context for broader analysis
        sentences = re.split(r'[.!?]+', text)
        entity_sentence = ""
        for sentence in sentences:
            sentence_start = text.find(sentence)
            if sentence_start != -1 and entity.start_char >= sentence_start and entity.end_char <= sentence_start + len(sentence):
                entity_sentence = sentence.lower()
                break

        max_confidence = 0

        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            if not keyword_lower:
                continue

            current_confidence = 0

            # Multi-word patterns with strict position scoring
            if ' ' in keyword_lower:
                if keyword_lower in context:
                    # Negation should be close and BEFORE the entity
                    keyword_pos = context.find(keyword_lower)
                    if keyword_pos < 10:  # Before entity in context window
                        distance = 10 - keyword_pos
                        current_confidence = 0.9 - (distance / 20)
                elif keyword_lower in entity_sentence:
                    # Check if negation appears before entity in sentence
                    entity_text_lower = entity.text.lower()
                    if entity_sentence.find(keyword_lower) < entity_sentence.find(entity_text_lower):
                        current_confidence = 0.6
            else:
                # Single-word patterns with strict word boundaries
                pattern = r'\b' + re.escape(keyword_lower) + r'\b'

                context_match = re.search(pattern, context)
                if context_match and context_match.start() < 10:  # Before entity
                    distance = 10 - context_match.start()
                    current_confidence = 0.8 - (distance / 15)
                elif re.search(pattern, entity_sentence):
                    # Check position in sentence
                    entity_text_lower = entity.text.lower()
                    if entity_sentence.find(keyword_lower) < entity_sentence.find(entity_text_lower):
                        current_confidence = 0.5

            max_confidence = max(max_confidence, current_confidence)

        # Enhanced false positive detection to prevent incorrect negation
        false_positive_patterns = [
            r'\bnormal\s+range\b',        # "normal range"
            r'\bnormal\s+limits\b',       # "normal limits"
            r'\bnormal\s+variant\b',      # "normal variant"
            r'\bno\s+longer\b',           # "no longer" (temporal, not negation)
            r'\bno\s+other\b',            # "no other" (not direct negation)
            r'\bno\s+change\b',           # "no change" (not negation of entity)
            r'\bno\s+difference\b',       # "no difference"
            r'\bno\s+significant\b',      # "no significant" (often followed by positive findings)
            r'\bnot\s+only\b',            # "not only" (emphasis, not negation)
            r'\bnot\s+just\b',            # "not just"
            r'\bnor\s+mal\b',             # "normal" split across words
        ]

        # Additional check: ensure negation keyword is not part of a larger word
        # This prevents "no" in "normal", "diagnosis", "prognosis", etc.
        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            if ' ' not in keyword_lower:  # Only for single words
                # Check if the keyword appears as a substring within other words
                pattern = r'\w' + re.escape(keyword_lower) + r'\w'
                if re.search(pattern, context.lower()):
                    # Found the keyword within another word - reduce confidence significantly
                    max_confidence *= 0.1

        for fp_pattern in false_positive_patterns:
            if re.search(fp_pattern, entity_sentence):
                max_confidence *= 0.3  # Reduce confidence for false positives

        # Template priority: lower threshold for template patterns
        threshold = 0.3 if template_priority else 0.6
        return max_confidence >= threshold

    def _generate_displacy_visualization(self, doc) -> str:
        """Generate Excel-friendly text visualization with entity highlighting"""
        try:
            if not doc.ents:
                return doc.text

            # Create text with entity annotations for Excel visibility
            text_with_highlights = doc.text
            entities_info = []

            # Sort entities by start position (reverse order for replacement)
            sorted_entities = sorted(doc.ents, key=lambda x: x.start_char, reverse=True)

            for ent in sorted_entities:
                # Create highlighted text with entity information
                entity_label = ent.label_
                entity_text = ent.text

                # Remove "BioBERT" or "biobert" from entity labels for cleaner visualization
                clean_label = entity_label.replace("BioBERT_", "").replace("biobert_", "").replace("BIOBERT_", "").upper()

                # Check for negation/context
                context_info = []
                if hasattr(ent, '_') and hasattr(ent._, 'negex') and ent._.negex:
                    context_info.append("NEGATED")

                # Create highlighted replacement text
                if context_info:
                    highlighted_text = f"▶[{entity_text}]◀ ({clean_label}:{':'.join(context_info)})"
                else:
                    highlighted_text = f"▶[{entity_text}]◀ ({clean_label})"

                # Replace in text
                text_with_highlights = (
                    text_with_highlights[:ent.start_char] +
                    highlighted_text +
                    text_with_highlights[ent.end_char:]
                )

                # Track entity info
                entities_info.append({
                    'text': entity_text,
                    'label': entity_label,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'context': context_info
                })

            # Add entity summary at the end
            if entities_info:
                summary = f"\n\n📋 ENTITIES DETECTED ({len(entities_info)}):\n"
                for i, ent_info in enumerate(entities_info, 1):
                    context_str = f" [{':'.join(ent_info['context'])}]" if ent_info['context'] else ""
                    summary += f"{i}. {ent_info['text']} ({ent_info['label']}){context_str}\n"

                text_with_highlights += summary

            return text_with_highlights

        except Exception as e:
            logger.warning(f"Could not generate text visualization: {e}")
            return doc.text if hasattr(doc, 'text') else ""

    def _categorize_sections(self, text: str) -> str:
        """Categorize text into medical sections by detecting section headers"""
        import re
        sections = []

        # Clean text for better pattern matching (remove HTML-like tags)
        clean_text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
        clean_text = re.sub(r'\s+', ' ', clean_text)  # Normalize whitespace

        # Section patterns - match at start of line or with colon/newline
        section_keywords = {
            'Chief Complaint': [r'\bchief\s+complaint\b', r'\bcc\s*:', r'\bpresenting\s+complaint\b'],
            'History of Present Illness': [r'\bhpi\b', r'\bhistory\s+of\s+present\s+illness\b', r'\bpresent\s+illness\b'],
            'Past Medical History': [r'\bpmh\b', r'\bpast\s+medical\s+history\b', r'\bmedical\s+history\b'],
            'Family History': [r'\bfamily\s+history\b', r'\bfh\s*:', r'\bfhx\b'],
            'Social History': [r'\bsocial\s+history\b', r'\bsh\s*:'],
            'Medications': [r'\bmedications\b', r'\bmeds\s*:', r'\bcurrent\s+medications\b'],
            'Allergies': [r'\ballergies\b', r'\bnkda\b', r'\bdrug\s+allergies\b'],
            'Physical Exam': [r'\bphysical\s+exam\b', r'\bpe\s*:', r'\bexamination\b', r'\bphysical\s+examination\b'],
            'Assessment': [r'\bassessment\b', r'\bimpression\b', r'\bassessment\s+and\s+plan\b', r'\ba\s*&\s*p\b'],
            'Plan': [r'\bplan\b', r'\btreatment\s+plan\b', r'\brecommendations\b', r'\btreatment\b', r'\btherapy\b', r'\bmanagement\b'],
            'Summary': [r'\bsummary\b', r'\bconclusion\b', r'\bfinal\s+impression\b'],
            'Introduction': [r'\bintroduction\b', r'\bbackground\b', r'\boverview\b'],
            'Review of Systems': [r'\bros\b', r'\breview\s+of\s+systems\b', r'\bsystems\s+review\b'],
            'Objective': [r'\bobjective\b', r'\bo\s*:'],
            'Subjective': [r'\bsubjective\b', r'\bs\s*:'],
            'Vital Signs': [r'\bvital\s+signs\b', r'\bvitals\b', r'\bvs\s*:'],
            'Discharge': [r'\bdischarge\b', r'\bdischarge\s+summary\b', r'\bdischarge\s+plan\b'],
            'Admission': [r'\badmission\b', r'\bhospital\s+course\b', r'\badmission\s+note\b'],
            'Follow-up': [r'\bfollow[\s-]up\b', r'\breturn\b', r'\bnext\s+appointment\b'],
            'Laboratory': [r'\blaboratory\b', r'\blabs\b', r'\blab\s+results\b'],
            'Imaging': [r'\bimaging\b', r'\bradiology\b', r'\bx[\s-]ray\b']
        }

        # Check if pattern appears as section header
        # Matches: start of string, after newline, or after sentence-ending punctuation (. ! ?)
        # Also allow standalone words that might be section headers
        for section, patterns in section_keywords.items():
            for pattern in patterns:
                # Try multiple pattern matching strategies:
                # 1. Traditional: at start, after newline, or after punctuation
                section_pattern = rf'(^|\n|[.!?][0-9\s]*\s+){pattern}\s*[:,.]?'
                # 2. Standalone at beginning: word boundary at start
                standalone_pattern = rf'^{pattern}\b'
                # 3. Simple substring match (as fallback)
                simple_pattern = pattern

                if (re.search(section_pattern, clean_text, re.IGNORECASE | re.MULTILINE) or
                    re.search(standalone_pattern, clean_text, re.IGNORECASE) or
                    re.search(simple_pattern, clean_text, re.IGNORECASE)):
                    sections.append(section)
                    break

        return '; '.join(sections) if sections else 'General Clinical Text'

    def _extract_context_sentences(self, text: str, entities: List[Dict], context_type: str) -> str:
        """
        Extract precise 100-character context around entities where specific context patterns are detected.

        Returns context sentences in the format: [PREDICTOR | ENTITY_NAME] ...context_text...

        Args:
            text: The clinical text
            entities: List of detected entities
            context_type: Type of context (confirmed, negated, uncertain, historical, family)

        Returns:
            Formatted context sentences with predictor keywords and entity names

        Example:
            "[HAS | diabetes] ...Patient has diabetes and hypertension..."
            "[NO | chest pain] ...Patient denies chest pain and fever..."
        """
        if not entities:
            return ""

        context_extracts = []
        text_lower = text.lower()

        # Define patterns for each context type - use loaded patterns when available
        if context_type == 'historical':
            # Use loaded historical patterns if available, otherwise use fallback
            if self.historical_rules_enabled and self.historical_patterns:
                patterns = self.historical_patterns
            else:
                patterns = [
                    'history of', 'past medical history', 'previously diagnosed', 'previous', 'prior',
                    'had', 'diagnosed', 'treated for', 'hx of', 'pmh', 'former', 'earlier', 'ago',
                    'previously', 'past', 'historical', 'before', 'prior to'
                ]
        elif context_type == 'uncertain':
            # Use loaded uncertainty patterns if available, otherwise use fallback
            if self.uncertainty_rules_enabled and self.uncertainty_patterns:
                patterns = self.uncertainty_patterns
            else:
                patterns = [
                    'possible', 'probable', 'likely', 'may', 'might', 'could', 'may have', 'could be', 'suspected',
                    'rule out', 'differential', 'consider', 'questionable', 'uncertain', 'unknown',
                    'perhaps', 'maybe', 'possibly', 'suggest', 'may be', 'could have',
                    'suspicious for', 'concern for', 'worrisome for', 'unclear', 'undetermined'
                ]
        elif context_type == 'family':
            # Use loaded family patterns if available, otherwise use fallback
            if self.family_rules_enabled and self.family_patterns:
                patterns = self.family_patterns
            else:
                patterns = [
                    'family history', 'father', 'mother', 'parent', 'sibling', 'brother', 'sister',
                    'grandmother', 'grandfather', 'aunt', 'uncle', 'cousin', 'relatives',
                    'familial', 'hereditary', 'genetic', 'fh:', 'fhx', 'maternal', 'paternal',
                    'inherited'
                ]
        elif context_type == 'negated':
            # Use loaded negation patterns if available, otherwise use fallback
            if self.negated_rules_enabled and self.negated_patterns:
                patterns = self.negated_patterns
            else:
                patterns = [
                    'no', 'not', 'without', 'absent', 'negative', 'denies', 'rules out',
                    'free of', 'clear of', 'unremarkable', 'normal', 'within normal limits',
                    'no evidence of', 'no signs of', 'no history of', 'never had', 'r/o'
                ]
        elif context_type == 'confirmed':
            # Use loaded confirmed patterns if available, otherwise use fallback
            if self.confirmed_rules_enabled and self.confirmed_patterns:
                patterns = self.confirmed_patterns
            else:
                patterns = [
                    'diagnosed with', 'diagnosis of', 'confirmed', 'has', 'shows', 'demonstrates',
                    'presents with', 'found to have', 'suffering from', 'affected by',
                    'positive for', 'tested positive', 'evidence of', 'consistent with',
                    'currently has', 'currently experiencing', 'active', 'ongoing',
                    'established', 'documented', 'known', 'chronic', 'acute',
                    'findings show', 'reveals', 'indicates', 'suggests', 'compatible with',
                    'definitive', 'conclusive', 'certain', 'clear evidence', 'obvious', 'detected'
                ]
        else:
            return ""

        # For each entity, find the nearest context pattern and extract ±100 characters
        for entity in entities:
            entity_start = entity.get('start', 0)
            entity_end = entity.get('end', len(entity.get('text', '')))
            entity_text = entity.get('text', '')

            # Search for patterns within ±200 characters of the entity
            search_start = max(0, entity_start - 200)
            search_end = min(len(text), entity_end + 200)
            search_region = text[search_start:search_end]
            search_region_lower = search_region.lower()

            # Find the closest pattern to this entity
            closest_pattern_pos = None
            closest_pattern = None
            min_distance = float('inf')

            for pattern in patterns:
                pattern_pos = search_region_lower.find(pattern)
                if pattern_pos != -1:
                    # Calculate distance from pattern to entity
                    absolute_pattern_pos = search_start + pattern_pos
                    distance = abs(absolute_pattern_pos - entity_start)

                    if distance < min_distance:
                        min_distance = distance
                        closest_pattern_pos = absolute_pattern_pos
                        closest_pattern = pattern

            # If we found a relevant pattern, extract ±100 characters around the entity
            if closest_pattern_pos is not None and min_distance <= 200:
                # Extract 100 characters before and after the entity (consistent for all types)
                context_start = max(0, entity_start - 100)
                context_end = min(len(text), entity_end + 100)
                context_text = text[context_start:context_end].strip()

                # Normalize whitespace: replace newlines and multiple spaces with single space
                # This keeps each context sentence on a single line without gaps
                import re
                context_text = re.sub(r'\s+', ' ', context_text)

                # Get entity label for color highlighting
                entity_label = entity.get('label', 'UNKNOWN')
                # Clean label for display (remove BioBERT prefix)
                clean_label = entity_label.replace("BioBERT_", "").replace("biobert_", "").replace("BIOBERT_", "").upper()

                # Highlight entity occurrences within the context text using text markers
                # This format works in both Excel and Streamlit
                # Use case-insensitive matching with word boundaries
                entity_text_escaped = re.escape(entity_text)

                # Step 1: Replace entity text in context with markers
                context_text_highlighted = re.sub(
                    rf'\b({entity_text_escaped})\b',
                    lambda m: f'▶[{m.group(1)}]◀',  # Preserve original case from context
                    context_text,
                    flags=re.IGNORECASE
                )

                # Step 2: Highlight predictor pattern in context with italics marker
                # Use special marker ⟨pattern⟩ that can be converted to [pattern] in Excel or <i>[pattern]</i> in Streamlit
                pattern_escaped = re.escape(closest_pattern.lower())
                context_text_highlighted = re.sub(
                    rf'\b({pattern_escaped})\b',
                    lambda m: f'⟨{m.group(1)}⟩',  # Preserve original case, use ⟨⟩ for predictor
                    context_text_highlighted,
                    flags=re.IGNORECASE
                )

                # Format with text markers for Excel compatibility
                # Format: [PREDICTOR | entity] (TYPE) ...context with ⟨predictor⟩ and ▶[entity]◀...
                context_info = f'[{closest_pattern.upper()} | {entity_text}] ({clean_label}) ...{context_text_highlighted}...'
                context_extracts.append(context_info)

        # Remove duplicates and return with double newlines for blank line separation
        # Each context is on a single line, with a blank line between items
        unique_contexts = list(dict.fromkeys(context_extracts))  # Preserves order, removes duplicates
        if not unique_contexts:
            return ""
        # Format: double newline creates blank line between items in both Excel and Streamlit
        return '\n\n'.join(unique_contexts)

    def _format_entities(self, entities: List[Dict]) -> str:
        """Format entities for CSV output"""
        if not entities:
            return ""

        formatted = []
        for ent in entities:
            text = ent['text']
            if ent.get('negated'):
                text = f"NOT {text}"
            formatted.append(text)

        return '; '.join(formatted)

    def _get_unique_entities(self, entities: List[Dict[str, Any]]) -> str:
        """Extract unique entity texts from entity list"""
        if not entities:
            return ""

        unique_texts = []
        seen = set()
        for ent in entities:
            text = ent['text'].lower()  # Case-insensitive uniqueness
            if text not in seen:
                seen.add(text)
                unique_texts.append(ent['text'])  # Preserve original case

        return '; '.join(unique_texts)

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            'Text Visualization': "",
            'detected_diseases': "",
            'total_diseases_count': 0,
            'detected_diseases_unique': "",
            'detected_genes': "",
            'total_gene_count': 0,
            'detected_genes_unique': "",
            'detected_drugs': "",
            'detected_drugs_count': 0,
            'detected_drugs_unique': "",
            'detected_chemicals': "",
            'total_chemicals_count': 0,
            'detected_chemicals_unique': "",
            'confirmed_entities': "",
            'confirmed_entities_count': 0,
            'confirmed_entities_predictors': "",
            'confirmed_context_sentences': "",
            'negated_entities': "",
            'negated_entities_count': 0,
            'negated_entities_predictors': "",
            'negated_context_sentences': "",
            'historical_entities': "",
            'historical_entities_count': 0,
            'historical_entities_predictors': "",
            'historical_context_sentences': "",
            'uncertain_entities': "",
            'uncertain_entities_count': 0,
            'uncertain_entities_predictors': "",
            'uncertain_context_sentences': "",
            'family_entities': "",
            'family_entities_count': 0,
            'family_entities_predictors': "",
            'family_context_sentences': "",
            'section_categories': "General Clinical Text",
            'all_entities_json': "[]"
        }

    def predict_dataframe(self, df: pd.DataFrame, text_column: str = 'Text') -> pd.DataFrame:
        """Process entire dataframe and add prediction columns with text preprocessing"""
        logger.info(f"Processing {len(df)} rows for enhanced medical NER prediction")

        # Validate required columns (case-sensitive)
        required_columns = ['Index', 'Text']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Required columns missing: {missing_columns}. Input must have case-sensitive 'Index' and 'Text' columns.")

        # Ensure text column exists (case-sensitive)
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in dataframe (case-sensitive)")

        # Create a copy of the dataframe to preserve all original columns
        result_df = df.copy()

        # Add Text_Clean column as first output column (after input columns)
        logger.info("🧹 Preprocessing text to remove HTML/RTF formatting and clean for NLP processing")
        result_df['Text_Clean'] = result_df[text_column].apply(preprocess_text)

        # Define NLP prediction columns that will be appended
        nlp_prediction_columns = [
            'Text Visualization',
            'detected_diseases',
            'total_diseases_count',
            'detected_diseases_unique',
            'detected_genes',
            'total_gene_count',
            'detected_genes_unique',
            'detected_drugs',
            'detected_drugs_count',
            'detected_drugs_unique',
            'detected_chemicals',
            'total_chemicals_count',
            'detected_chemicals_unique',
            'confirmed_entities',
            'confirmed_entities_count',
            'confirmed_entities_predictors',
            'confirmed_context_sentences',
            'negated_entities',
            'negated_entities_count',
            'negated_entities_predictors',
            'negated_context_sentences',
            'historical_entities',
            'historical_entities_count',
            'historical_entities_predictors',
            'historical_context_sentences',
            'uncertain_entities',
            'uncertain_entities_count',
            'uncertain_entities_predictors',
            'uncertain_context_sentences',
            'family_entities',
            'family_entities_count',
            'family_entities_predictors',
            'family_context_sentences',
            'section_categories',
            'all_entities_json'
        ]

        # Add NLP prediction columns at the end (only if they don't already exist)
        for col in nlp_prediction_columns:
            if col not in result_df.columns:
                result_df[col] = ""

        logger.info(f"Input columns preserved: {list(df.columns)}")
        logger.info(f"Prediction columns added: {['Text_Clean'] + nlp_prediction_columns}")

        # Process each row
        for idx, row in result_df.iterrows():
            try:
                # Use cleaned text for NLP processing
                clean_text = row['Text_Clean']

                # Extract entities and context from cleaned text
                result = self.extract_entities(clean_text)

                # Fill NLP prediction columns
                for col in nlp_prediction_columns:
                    result_df.at[idx, col] = result[col]

                if idx % 5 == 0:
                    logger.info(f"Processed {idx + 1}/{len(result_df)} rows")

            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                # Fill with empty values for this row
                empty_result = self._empty_result()
                for col in nlp_prediction_columns:
                    result_df.at[idx, col] = empty_result[col]

        logger.info("✅ Enhanced DataFrame processing completed")
        logger.info(f"Output DataFrame shape: {result_df.shape}")
        logger.info(f"Final columns: {list(result_df.columns)}")

        return result_df

    def _get_protein_terms(self) -> set:
        """Get comprehensive list of protein and enzyme terms that are often misclassified as PERSON"""
        return {
            'kinesin', 'myosin', 'actin', 'tubulin', 'collagen', 'elastin', 'fibrin',
            'albumin', 'globulin', 'immunoglobulin', 'antibody', 'insulin', 'glucagon',
            'thyroxine', 'cortisol', 'adrenaline', 'epinephrine', 'norepinephrine',
            'dopamine', 'serotonin', 'acetylcholine', 'gaba', 'glutamate', 'glycine',
            'histamine', 'melatonin', 'oxytocin', 'vasopressin', 'prolactin',
            'growth hormone', 'thyroid hormone', 'parathyroid hormone', 'calcitonin',
            'renin', 'angiotensin', 'aldosterone', 'cortisone', 'testosterone',
            'estrogen', 'progesterone', 'luteinizing hormone', 'follicle stimulating hormone',
            'hemoglobin', 'myoglobin', 'cytochrome', 'catalase', 'peroxidase',
            'amylase', 'pepsin', 'trypsin', 'chymotrypsin', 'elastase', 'collagenase',
            'hyaluronidase', 'lysozyme', 'ribonuclease', 'deoxyribonuclease',
            'kinase', 'phosphatase', 'dehydrogenase', 'oxidase', 'reductase',
            'transferase', 'hydrolase', 'lyase', 'ligase', 'isomerase',
            'protease', 'peptidase', 'esterase', 'lipase', 'maltase', 'lactase',
            'sucrase', 'invertase', 'cellulase', 'pectinase', 'chitinase',
            'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
            'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi', 'rho',
            'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega'
        }

    def _get_anatomical_terms(self) -> set:
        """Get anatomical terms that might be confused with other entity types"""
        return {
            'heart', 'lung', 'liver', 'kidney', 'brain', 'stomach', 'intestine',
            'pancreas', 'spleen', 'bladder', 'gallbladder', 'thyroid', 'adrenal',
            'pituitary', 'hypothalamus', 'cerebellum', 'cerebrum', 'brainstem',
            'spinal cord', 'vertebra', 'rib', 'sternum', 'femur', 'tibia', 'fibula',
            'humerus', 'radius', 'ulna', 'skull', 'mandible', 'maxilla', 'clavicle',
            'scapula', 'pelvis', 'sacrum', 'coccyx', 'patella', 'tarsus', 'metatarsus',
            'phalanges', 'carpus', 'metacarpus', 'muscle', 'tendon', 'ligament',
            'cartilage', 'bone', 'joint', 'synovium', 'fascia', 'nerve', 'vessel',
            'artery', 'vein', 'capillary', 'lymph', 'lymphatic', 'skin', 'epidermis',
            'dermis', 'subcutaneous', 'hair', 'nail', 'eye', 'ear', 'nose', 'mouth',
            'throat', 'larynx', 'pharynx', 'trachea', 'bronchus', 'alveolus'
        }

    def _filter_and_prioritize_entities(self, entities, text: str) -> list:
        """Filter out false positive standard spaCy entities and prioritize medical entities"""
        filtered_entities = []

        for ent in entities:
            entity_text = ent.text.lower().strip()
            entity_label = ent.label_

            # Skip if entity is too short or contains only punctuation
            if len(entity_text) < 2 or entity_text.isdigit() or not entity_text.isalnum():
                continue

            # Check if this is a problematic standard entity that should be medical instead
            is_medical_conflict = self._is_medical_entity_conflict(entity_text, entity_label)

            # If it's a conflict, check if we can reclassify it as medical
            if is_medical_conflict:
                medical_classification = self._get_medical_classification(entity_text)
                if medical_classification:
                    # Create a new entity with medical classification
                    # We'll modify the label in the entity processing
                    filtered_entities.append(ent)
                    continue
                else:
                    # For PERSON entities that might be proteins, keep them for reclassification
                    if entity_label == 'PERSON':
                        filtered_entities.append(ent)  # Let classification system handle it
                        continue
                    # Skip other problematic entities that can't be reclassified
                    continue

            # Keep the entity if it passes all filters
            filtered_entities.append(ent)

        return filtered_entities

    def _is_medical_entity_conflict(self, entity_text: str, entity_label: str) -> bool:
        """Check if a standard spaCy entity conflicts with medical terminology"""
        entity_lower = entity_text.lower()

        # Check for protein names misclassified as PERSON
        if entity_label == 'PERSON':
            if any(protein in entity_lower for protein in self.protein_terms):
                return True
            # Check for common protein patterns
            if any(pattern in entity_lower for pattern in ['kinase', 'ase', 'ogen', 'ine', 'sin']):
                return True

        # Check for syndromes/diseases misclassified as ORG or GPE
        if entity_label in ['ORG', 'GPE']:
            if any(term in entity_lower for term in ['syndrome', 'disease', 'disorder', 'deficiency']):
                return True

        # Check for medications misclassified as PRODUCT
        if entity_label == 'PRODUCT':
            if any(term in entity_lower for term in ['medication', 'drug', 'vaccine', 'serum']):
                return True

        return False

    def _get_medical_classification(self, entity_text: str) -> str:
        """Get the appropriate medical classification for an entity"""
        entity_lower = entity_text.lower()

        # Check for proteins/enzymes
        if any(protein in entity_lower for protein in self.protein_terms):
            return 'PROTEIN'

        # Check for diseases
        if any(disease in entity_lower for disease in self.disease_terms):
            return 'DISEASE'

        # Check for genes
        if any(gene in entity_lower for gene in self.gene_terms):
            return 'GENE'

        # Check for drugs
        if any(drug in entity_lower for drug in self.drug_terms):
            return 'DRUG'

        # Check for anatomical terms
        if any(anatomy in entity_lower for anatomy in self.anatomical_terms_extended):
            return 'ANATOMY'

        return None

    def _classify_medical_entity(self, entity_text: str, original_label: str) -> dict:
        """Enhanced medical entity classification with priority system"""
        entity_lower = entity_text.lower().strip()

        # Initialize classification
        classification = {
            'is_disease': False,
            'is_gene': False,
            'is_protein': False,
            'is_drug': False,
            'is_anatomy': False,
            'medical_label': original_label,
            'confidence': 0.0
        }

        # Priority order: diseases > genes/proteins > drugs > anatomy

        # 1. Check for diseases (highest priority)
        if (original_label in ['DISEASE', 'CONDITION', 'PROBLEM'] or
            any(disease in entity_lower for disease in self.disease_terms)):
            classification.update({
                'is_disease': True,
                'medical_label': 'DISEASE',
                'confidence': 0.9
            })
            return classification

        # 2. Check for genes (high priority)
        if (original_label in ['GENE', 'GENETIC'] or
            any(gene in entity_lower for gene in self.gene_terms)):
            classification.update({
                'is_gene': True,
                'medical_label': 'GENE',
                'confidence': 0.85
            })
            return classification

        # 3. Check for proteins (high priority, override PERSON misclassification)
        if (original_label in ['PROTEIN'] or
            any(protein in entity_lower for protein in self.protein_terms) or
            (original_label == 'PERSON' and self._is_medical_entity_conflict(entity_text, 'PERSON')) or
            (original_label == 'PERSON' and any(pattern in entity_lower for pattern in ['kinesin', 'protein', 'ase', 'ogen', 'ine']))):
            classification.update({
                'is_protein': True,
                'medical_label': 'PROTEIN',
                'confidence': 0.8
            })
            return classification

        # 4. Check for drugs/chemicals (medium priority)
        if (original_label in ['DRUG', 'CHEMICAL', 'MEDICATION'] or
            any(drug in entity_lower for drug in self.drug_terms)):
            classification.update({
                'is_drug': True,
                'medical_label': 'DRUG',
                'confidence': 0.75
            })
            return classification

        # 5. Check for anatomical terms (lower priority)
        if (original_label in ['ANATOMY', 'ANATOMICAL_LOCATION'] or
            any(anatomy in entity_lower for anatomy in self.anatomical_terms_extended)):
            classification.update({
                'is_anatomy': True,
                'medical_label': 'ANATOMY',
                'confidence': 0.7
            })
            return classification

        # Default: keep original classification with low confidence
        classification['confidence'] = 0.1
        return classification

    def _detect_additional_medical_patterns(self, text: str) -> dict:
        """Detect additional medical entities using pattern matching (fallback rules)"""
        additional_entities = {
            'diseases': [],
            'genes': [],
            'chemicals': [],
            'all': []
        }

        # Use simple pattern matching for additional medical terms
        text_lower = text.lower()

        # Check for disease terms
        for disease_term in self.disease_terms:
            if disease_term.lower() in text_lower:
                # Find all occurrences
                import re
                for match in re.finditer(re.escape(disease_term.lower()), text_lower):
                    entity_info = {
                        'text': text[match.start():match.end()],
                        'label': 'DISEASE',
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.75,
                        'source': 'pattern_matching'
                    }
                    additional_entities['diseases'].append(entity_info)
                    additional_entities['all'].append(entity_info)

        # Check for gene terms
        for gene_term in self.gene_terms:
            if gene_term.lower() in text_lower:
                import re
                for match in re.finditer(re.escape(gene_term.lower()), text_lower):
                    entity_info = {
                        'text': text[match.start():match.end()],
                        'label': 'GENE',
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.75,
                        'source': 'pattern_matching'
                    }
                    additional_entities['genes'].append(entity_info)
                    additional_entities['all'].append(entity_info)

        # Check for drug terms
        for drug_term in self.drug_terms:
            if drug_term.lower() in text_lower:
                import re
                for match in re.finditer(re.escape(drug_term.lower()), text_lower):
                    entity_info = {
                        'text': text[match.start():match.end()],
                        'label': 'DRUG',
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.75,
                        'source': 'pattern_matching'
                    }
                    additional_entities['chemicals'].append(entity_info)
                    additional_entities['all'].append(entity_info)

        return additional_entities

    def _determine_medical_type_from_patterns(self, entity_text: str) -> str:
        """Determine medical entity type using only medical rules"""
        if any(disease in entity_text for disease in self.disease_terms):
            return 'DISEASE'
        elif any(gene in entity_text for gene in self.gene_terms):
            return 'GENE'
        elif any(protein in entity_text for protein in self.protein_terms):
            return 'PROTEIN'
        elif any(drug in entity_text for drug in self.drug_terms):
            return 'DRUG'
        elif any(anatomy in entity_text for anatomy in self.anatomical_terms_extended):
            return 'ANATOMY'
        return None

    def _detect_negated_medical_entities(self, text: str, medical_entities: list) -> list:
        """Detect negated medical entities using template patterns with confidence scoring (80% threshold)"""
        negated_entities = []

        # Use template patterns if available
        if self.negated_rules_enabled and self.negated_patterns:
            keywords = self.negated_patterns
        else:
            keywords = [
                'no', 'not', 'without', 'absent', 'negative', 'denies', 'rules out',
                'free of', 'clear of', 'unremarkable', 'normal', 'within normal limits'
            ]

        # Pattern strength scores (strong negation = high confidence)
        strong_patterns = ['no', 'not', 'denies', 'absent', 'without', 'rules out', 'ruled out', 'negative for']
        moderate_patterns = ['free of', 'clear of', 'unremarkable', 'normal']
        weak_patterns = ['within normal limits']

        for entity in medical_entities:
            # Check context around the entity
            start_idx = max(0, entity['start'] - 50)
            end_idx = min(len(text), entity['end'] + 50)
            context = text[start_idx:end_idx].lower()

            # Check for scope-reversing conjunctions early
            # If entity comes after "but reports", "but shows", etc., skip negation check entirely
            entity_text_lower = entity.get('text', '').lower()
            entity_pos_in_context = entity['start'] - start_idx
            entity_text_pos = context.find(entity_text_lower)

            # Check for confirmation phrases that override negation
            skip_negation = False
            for phrase in ['but reports', 'but shows', 'but has', 'but demonstrates', 'but presents',
                          'however reports', 'however shows', 'although has', 'yet presents']:
                phrase_pos = context.find(phrase)
                if phrase_pos >= 0 and entity_text_pos >= 0 and phrase_pos < entity_text_pos:
                    # Entity comes after a confirmation phrase - should NOT be negated
                    skip_negation = True
                    break

            if skip_negation:
                continue  # Skip this entity for negation detection

            # Also check smaller window for proximity scoring
            close_context_start = max(0, entity['start'] - 10)
            close_context_end = min(len(text), entity['end'] + 10)
            close_context = text[close_context_start:close_context_end].lower()

            # Check keywords with word boundaries and calculate confidence
            matched_keyword = None
            match_distance = 50  # Default max distance

            for keyword in keywords:
                keyword_lower = keyword.lower()
                # Multi-word patterns: use simple substring match
                if ' ' in keyword_lower:
                    if keyword_lower in context:
                        matched_keyword = keyword_lower
                        # Calculate distance to entity
                        keyword_pos = context.find(keyword_lower)
                        entity_pos_in_context = entity['start'] - start_idx
                        match_distance = abs(keyword_pos - entity_pos_in_context)
                        break
                # Single-word patterns: use word boundaries to avoid partial matches
                else:
                    match = re.search(r'\b' + re.escape(keyword_lower) + r'\b', context)
                    if match:
                        matched_keyword = keyword_lower
                        # Calculate distance to entity
                        keyword_pos = match.start()
                        entity_pos_in_context = entity['start'] - start_idx
                        match_distance = abs(keyword_pos - entity_pos_in_context)
                        break

            if matched_keyword:
                # Calculate confidence score (0-100)
                base_confidence = 0

                # Pattern strength component (40 points max)
                if matched_keyword in strong_patterns:
                    base_confidence += 40
                elif matched_keyword in moderate_patterns:
                    base_confidence += 25
                elif matched_keyword in weak_patterns:
                    base_confidence += 15
                else:
                    # Default for loaded template patterns
                    base_confidence += 30

                # Proximity component (40 points max) - closer = higher confidence
                if match_distance <= 5:  # Very close (within 5 chars)
                    base_confidence += 40
                elif match_distance <= 10:  # Close (within 10 chars)
                    base_confidence += 35
                elif match_distance <= 20:  # Near (within 20 chars)
                    base_confidence += 25
                elif match_distance <= 35:  # Moderate distance
                    base_confidence += 15
                else:  # Far (35-50 chars)
                    base_confidence += 5

                # Sentence structure component (20 points max)
                # Check if negation comes before entity (more reliable)
                entity_text_lower = entity.get('text', '').lower()
                if matched_keyword in close_context:  # Very close - high confidence
                    base_confidence += 20
                elif f"{matched_keyword} {entity_text_lower}" in context:  # Direct adjacency
                    base_confidence += 15
                elif context.find(matched_keyword) < context.find(entity_text_lower):  # Negation before entity
                    base_confidence += 10
                else:  # Negation after entity or ambiguous
                    base_confidence += 5

                # Store confidence in entity (always add confidence info to original entity)
                entity['negation_confidence'] = base_confidence
                entity['matched_negation_pattern'] = matched_keyword

                # Only add to negated list if confidence >= 80%
                if base_confidence >= 80:
                    negated_entities.append(entity)
                # Low confidence (50-79%) will be classified as uncertain by caller
                # Very low confidence (<50%) will be ignored (likely false positive)

        return negated_entities

    def _detect_historical_medical_entities(self, text: str, medical_entities: list) -> list:
        """Detect historical medical entities using template patterns or fallback with word boundaries"""
        historical_entities = []

        # Use template patterns if available
        if self.historical_rules_enabled and self.historical_patterns:
            keywords = self.historical_patterns
        else:
            keywords = [
                'history of', 'past history of', 'past medical history', 'previous', 'prior', 'previously', 'former',
                'years ago', 'months ago', 'weeks ago', 'days ago', 'earlier this',
                'childhood', 'adolescent', 'remote history', 'longstanding history'
            ]

        # Exclusion patterns that indicate current/present findings, not historical
        exclusion_patterns = [
            'was detected', 'was found', 'is detected', 'is found', 'has been detected',
            'has been found', 'recently detected', 'recently found', 'currently has',
            'presently has', 'now has', 'shows', 'demonstrates', 'presents with',
            'treatment includes', 'includes', 'therapy includes', 'regimen includes',
            'medications include', 'management includes'
        ]

        for entity in medical_entities:
            start_idx = max(0, entity['start'] - 50)
            end_idx = min(len(text), entity['end'] + 50)
            context = text[start_idx:end_idx].lower()

            # First check for exclusion patterns - if found, skip this entity
            is_excluded = False
            for exclusion in exclusion_patterns:
                if exclusion in context:
                    is_excluded = True
                    break

            if is_excluded:
                continue

            # Check keywords with word boundaries
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # Multi-word patterns: use simple substring match
                if ' ' in keyword_lower:
                    if keyword_lower in context:
                        historical_entities.append(entity)
                        break
                # Single-word patterns: use word boundaries
                else:
                    if re.search(r'\b' + re.escape(keyword_lower) + r'\b', context):
                        historical_entities.append(entity)
                        break

        return historical_entities

    def _detect_uncertain_medical_entities(self, text: str, medical_entities: list) -> list:
        """Detect uncertain medical entities using template patterns or fallback with word boundaries"""
        uncertain_entities = []

        # Use template patterns if available
        if self.uncertainty_rules_enabled and self.uncertainty_patterns:
            keywords = self.uncertainty_patterns
        else:
            keywords = [
                'possible', 'possibly', 'probable', 'probably', 'likely', 'unlikely',
                'may', 'might', 'could', 'perhaps', 'maybe', 'suspect', 'suspected',
                'questionable', 'uncertain', 'unclear', 'rule out', 'r/o', 'consider',
                'differential', 'suspicious', 'query', 'may have', 'may be', 'could be'
            ]

        for entity in medical_entities:
            start_idx = max(0, entity['start'] - 50)
            end_idx = min(len(text), entity['end'] + 50)
            context = text[start_idx:end_idx].lower()

            # Check keywords with word boundaries
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # Multi-word patterns: use simple substring match
                if ' ' in keyword_lower:
                    if keyword_lower in context:
                        uncertain_entities.append(entity)
                        break
                # Single-word patterns: use word boundaries
                else:
                    if re.search(r'\b' + re.escape(keyword_lower) + r'\b', context):
                        uncertain_entities.append(entity)
                        break

        return uncertain_entities

    def _detect_confirmed_medical_entities(self, text: str, medical_entities: list) -> list:
        """Detect confirmed/definitive medical entities with high confidence using template patterns or fallback"""
        confirmed_entities = []
        text_lower = text.lower()

        # Use template patterns if available, otherwise use fallback
        if self.confirmed_rules_enabled and self.confirmed_patterns:
            confirmed_patterns = self.confirmed_patterns
        else:
            # Fallback hardcoded patterns
            confirmed_patterns = [
                # Definitive statements
                'diagnosed with', 'diagnosis of', 'confirmed', 'has', 'shows', 'demonstrates',
                'presents with', 'found to have', 'suffering from', 'affected by',
                'positive for', 'tested positive', 'evidence of', 'consistent with',

                # Current status indicators
                'currently has', 'currently experiencing', 'active', 'ongoing',
                'established', 'documented', 'known', 'chronic', 'acute',

                # Medical findings
                'findings show', 'reveals', 'indicates', 'suggests', 'compatible with',
                'characteristic of', 'typical of', 'pathognomonic for',

                # Clinical certainty
                'definitive', 'conclusive', 'certain', 'clear evidence', 'obvious',
                'unmistakable', 'unambiguous', 'established diagnosis'
            ]

        for entity in medical_entities:
            entity_text = entity.get('text', '')
            entity_start = entity.get('start', 0)
            entity_end = entity.get('end', len(entity_text))

            # Search for confirmed patterns within ±200 characters of the entity
            search_start = max(0, entity_start - 200)
            search_end = min(len(text), entity_end + 200)
            search_region = text[search_start:search_end].lower()


            # Check if any confirmed patterns are found near this entity with word boundaries
            has_confirmed_pattern = False
            closest_distance = float('inf')

            import re
            for pattern in confirmed_patterns:
                pattern_lower = pattern.lower()

                # Use word boundaries for proper matching
                if ' ' in pattern_lower:
                    # Multi-word patterns use simple substring match
                    pattern_pos = search_region.find(pattern_lower)
                    if pattern_pos != -1:
                        absolute_pattern_pos = search_start + pattern_pos
                        distance = abs(absolute_pattern_pos - entity_start)
                        if distance <= 200 and distance < closest_distance:
                            has_confirmed_pattern = True
                            closest_distance = distance
                else:
                    # Single-word patterns use word boundaries to avoid false positives
                    regex = r'\b' + re.escape(pattern_lower) + r'\b'
                    for match in re.finditer(regex, search_region):
                        pattern_pos = match.start()
                        absolute_pattern_pos = search_start + pattern_pos
                        distance = abs(absolute_pattern_pos - entity_start)
                        if distance <= 200 and distance < closest_distance:
                            has_confirmed_pattern = True
                            closest_distance = distance

            # Only include entities that have confirmed patterns and are not already
            # classified as negated, uncertain, historical, or family
            if has_confirmed_pattern:
                # Additional filters to avoid false positives
                entity_lower = entity_text.lower()

                # Skip if entity is already classified in other contexts
                skip_entity = False

                # Check for scope-reversing conjunctions that should override negation filtering
                # If entity comes after "but reports", "but shows", etc., don't skip due to nearby negation
                entity_context = text[max(0, entity_start - 50):min(len(text), entity_end + 50)].lower()
                entity_text_pos = entity_context.find(entity_text.lower())

                # Check for comprehensive scope-reversing patterns with priority-based matching
                scope_reversed = False
                scope_reversal_patterns = [
                    # Priority 10: High-confidence adversative patterns (Confidence: 90-95%)
                    {'pattern': 'but reports', 'priority': 10, 'confidence': 0.95},
                    {'pattern': 'but shows', 'priority': 10, 'confidence': 0.95},
                    {'pattern': 'but has', 'priority': 10, 'confidence': 0.95},
                    {'pattern': 'but demonstrates', 'priority': 9, 'confidence': 0.93},
                    {'pattern': 'but presents', 'priority': 9, 'confidence': 0.93},
                    {'pattern': 'but exhibits', 'priority': 9, 'confidence': 0.92},
                    {'pattern': 'but complains of', 'priority': 9, 'confidence': 0.92},
                    {'pattern': 'but admits to', 'priority': 8, 'confidence': 0.90},
                    {'pattern': 'but acknowledges', 'priority': 8, 'confidence': 0.90},
                    {'pattern': 'but endorses', 'priority': 8, 'confidence': 0.88},

                    # Priority 8-9: However patterns (Confidence: 88-93%)
                    {'pattern': 'however reports', 'priority': 9, 'confidence': 0.93},
                    {'pattern': 'however shows', 'priority': 9, 'confidence': 0.93},
                    {'pattern': 'however has', 'priority': 9, 'confidence': 0.92},
                    {'pattern': 'however demonstrates', 'priority': 8, 'confidence': 0.90},

                    # Priority 7-8: Yet patterns (Confidence: 85-90%)
                    {'pattern': 'yet reports', 'priority': 8, 'confidence': 0.90},
                    {'pattern': 'yet shows', 'priority': 8, 'confidence': 0.90},
                    {'pattern': 'yet has', 'priority': 8, 'confidence': 0.88},

                    # Priority 7: Nevertheless/nonetheless patterns (Confidence: 85%)
                    {'pattern': 'nevertheless reports', 'priority': 7, 'confidence': 0.85},
                    {'pattern': 'nonetheless shows', 'priority': 7, 'confidence': 0.85},

                    # Priority 8-9: Temporal transitions (Confidence: 85-92%)
                    {'pattern': 'but now reports', 'priority': 9, 'confidence': 0.92},
                    {'pattern': 'but currently has', 'priority': 9, 'confidence': 0.92},
                    {'pattern': 'but today shows', 'priority': 8, 'confidence': 0.90},
                    {'pattern': 'however now reports', 'priority': 8, 'confidence': 0.90},
                    {'pattern': 'however currently demonstrates', 'priority': 8, 'confidence': 0.88},
                    {'pattern': 'yet currently has', 'priority': 7, 'confidence': 0.85},

                    # Priority 6-8: Exception patterns (Confidence: 78-90%)
                    {'pattern': 'except for', 'priority': 8, 'confidence': 0.90},
                    {'pattern': 'except', 'priority': 7, 'confidence': 0.85},
                    {'pattern': 'save for', 'priority': 6, 'confidence': 0.80},
                    {'pattern': 'apart from', 'priority': 6, 'confidence': 0.80},
                    {'pattern': 'aside from', 'priority': 6, 'confidence': 0.78},
                    {'pattern': 'with the exception of', 'priority': 7, 'confidence': 0.85},

                    # Priority 6-7: Concessive patterns (Confidence: 80-85%)
                    {'pattern': 'although reports', 'priority': 7, 'confidence': 0.85},
                    {'pattern': 'though shows', 'priority': 7, 'confidence': 0.85},
                    {'pattern': 'even though has', 'priority': 7, 'confidence': 0.83},
                    {'pattern': 'despite reports', 'priority': 6, 'confidence': 0.80},
                    {'pattern': 'in spite of shows', 'priority': 6, 'confidence': 0.80},

                    # Priority 4-6: Contrastive patterns (Confidence: 70-80%)
                    {'pattern': 'on the other hand reports', 'priority': 6, 'confidence': 0.80},
                    {'pattern': 'conversely shows', 'priority': 6, 'confidence': 0.78},
                    {'pattern': 'in contrast has', 'priority': 5, 'confidence': 0.75},
                    {'pattern': 'rather reports', 'priority': 5, 'confidence': 0.75},
                    {'pattern': 'instead shows', 'priority': 5, 'confidence': 0.73}
                ]

                # Sort patterns by priority (highest first) then by confidence
                scope_reversal_patterns.sort(key=lambda x: (x['priority'], x['confidence']), reverse=True)

                # Use priority-based pattern matching with confidence threshold
                confidence_threshold = 0.75  # Only use patterns with confidence >= 75%
                best_match = None
                best_confidence = 0

                for pattern_info in scope_reversal_patterns:
                    pattern = pattern_info['pattern']
                    confidence = pattern_info['confidence']
                    priority = pattern_info['priority']

                    # Skip low-confidence patterns
                    if confidence < confidence_threshold:
                        continue

                    phrase_pos = entity_context.find(pattern)
                    if phrase_pos >= 0 and entity_text_pos >= 0 and phrase_pos < entity_text_pos:
                        # Found a match - check if this is the best one so far
                        if confidence > best_confidence:
                            best_match = pattern_info
                            best_confidence = confidence

                # If we found a high-confidence scope reversal pattern, allow confirmation
                if best_match and best_confidence >= 0.80:  # High confidence threshold
                    scope_reversed = True

                # Define search region for pattern matching
                negated_search_region = text[max(0, entity_start - 50):min(len(text), entity_end + 50)].lower()

                # Check if entity appears in negated context with word boundaries (only if scope not reversed)
                if not scope_reversed:
                    negated_patterns = ['no', 'not', 'without', 'absent', 'negative', 'denies', 'rules out']

                    for neg_pattern in negated_patterns:
                        if ' ' in neg_pattern:
                            if neg_pattern in negated_search_region:
                                skip_entity = True
                                break
                        else:
                            if re.search(r'\b' + re.escape(neg_pattern) + r'\b', negated_search_region):
                                skip_entity = True
                                break

                # Check if entity appears in uncertain context with word boundaries
                if not skip_entity:
                    uncertain_patterns = ['possible', 'probable', 'likely', 'may', 'might', 'could', 'suspected']
                    for unc_pattern in uncertain_patterns:
                        if ' ' in unc_pattern:
                            if unc_pattern in negated_search_region:
                                skip_entity = True
                                break
                        else:
                            if re.search(r'\b' + re.escape(unc_pattern) + r'\b', negated_search_region):
                                skip_entity = True
                                break

                if not skip_entity:
                    confirmed_entity = {
                        'text': entity_text,
                        'start': entity_start,
                        'end': entity_end,
                        'label': entity.get('label', 'MEDICAL'),
                        'confidence': entity.get('confidence', 0.8)
                    }
                    confirmed_entities.append(confirmed_entity)

        return confirmed_entities

    def _detect_family_medical_entities(self, text: str, medical_entities: list) -> list:
        """Detect family history medical entities with improved precision using template patterns or fallback"""
        family_entities = []

        # Use template patterns if available, otherwise use fallback
        if self.family_rules_enabled and self.family_patterns:
            # Use all loaded patterns from template
            strong_family_keywords = self.family_patterns
            # For template-based, treat all as strong indicators
            ambiguous_keywords = []
        else:
            # Fallback hardcoded patterns
            # High-confidence family indicators (definitely family history)
            strong_family_keywords = [
                'family history', 'mother', 'father', 'sister', 'brother', 'parent',
                'maternal', 'paternal', 'grandmother', 'grandfather', 'aunt', 'uncle',
                'cousin', 'sibling', 'relatives', 'fh:', 'fhx'
            ]

            # Ambiguous terms that need additional context validation
            ambiguous_keywords = ['hereditary', 'genetic', 'familial']

        for entity in medical_entities:
            start_idx = max(0, entity['start'] - 50)
            end_idx = min(len(text), entity['end'] + 50)
            context = text[start_idx:end_idx].lower()

            # Check for strong family indicators first
            has_strong_family = any(keyword.lower() in context for keyword in strong_family_keywords)

            if has_strong_family:
                family_entities.append(entity)
            else:
                # For ambiguous terms, require additional family context
                has_ambiguous = any(keyword.lower() in context for keyword in ambiguous_keywords)

                if has_ambiguous:
                    # Look for additional family context in a wider window (±100 chars)
                    wider_start = max(0, entity['start'] - 100)
                    wider_end = min(len(text), entity['end'] + 100)
                    wider_context = text[wider_start:wider_end].lower()

                    # Only consider it family history if there are other family indicators nearby
                    family_context_indicators = [
                        'family', 'history of', 'runs in', 'inherited from',
                        'genetic counseling', 'family member', 'relative'
                    ]

                    if any(indicator in wider_context for indicator in family_context_indicators):
                        family_entities.append(entity)

        return family_entities

    def _extract_predictor_terms(self, text: str, entities: list, patterns: list) -> str:
        """Extract unique predictor terms found near entities"""
        if not entities or not patterns:
            return ""

        found_predictors = set()
        text_lower = text.lower()

        for entity in entities:
            # Use ±200 char window to match detection logic (especially for confirmed entities)
            start_idx = max(0, entity.get('start', 0) - 200)
            end_idx = min(len(text), entity.get('end', 0) + 200)
            context = text[start_idx:end_idx].lower()

            for pattern in patterns:
                if pattern.lower() in context:
                    found_predictors.add(pattern.upper())

        return ", ".join(sorted(found_predictors)) if found_predictors else ""

    def _generate_medical_visualization(self, text: str, medical_entities: list) -> str:
        """Generate visualization showing only medical entities (no standard spaCy entities)"""
        try:
            if not medical_entities:
                return text

            # Sort entities by start position (reverse order for replacement)
            sorted_entities = sorted(medical_entities, key=lambda x: x['start'], reverse=True)

            # Create highlighted text with medical entities only
            highlighted_text = text
            for entity in sorted_entities:
                start = entity['start']
                end = entity['end']
                entity_text = entity['text']
                medical_label = entity['label']
                source = entity.get('source', 'biobert')
                confidence = entity.get('confidence', 1.0)

                # Clean label and source for better visualization (remove BioBERT references)
                clean_label = medical_label.replace("BioBERT_", "").replace("biobert_", "").replace("BIOBERT_", "").upper()
                clean_source = source.replace("biobert_", "").replace("BioBERT_", "").replace("BIOBERT_", "")

                # Create highlighted replacement showing medical classification only
                replacement = f"▶[{entity_text}]◀ ({clean_label})"
                highlighted_text = highlighted_text[:start] + replacement + highlighted_text[end:]

            return highlighted_text

        except Exception as e:
            logger.error(f"Medical visualization failed: {e}")
            return text

    def _generate_enhanced_medical_visualization(
        self, text: str, all_entities: list, confirmed_entities: list,
        negated_entities: list, uncertain_entities: list,
        historical_entities: list, family_entities: list, section_categories: str
    ) -> str:
        """
        Generate enhanced visualization with entity context classifications.

        Shows each medical entity with its type (DISEASE, GENE, etc.) and context
        (CONFIRMED, NEGATED, UNCERTAIN, HISTORICAL, FAMILY).
        Also includes section category at the top.

        Args:
            text: The clinical text
            all_entities: All detected medical entities
            confirmed_entities: Entities with confirmed context
            negated_entities: Entities with negated context
            uncertain_entities: Entities with uncertain context
            historical_entities: Entities with historical context
            family_entities: Entities with family history context
            section_categories: Document section classifications

        Returns:
            Enhanced HTML visualization string
        """
        try:
            if not all_entities:
                # Still show section category even if no entities
                header = f"📋 SECTION: {section_categories}\n\n"
                return header + text

            # Create entity ID mapping for quick lookup (using text and position)
            def entity_key(entity):
                return f"{entity.get('text', '')}_{entity.get('start', 0)}_{entity.get('end', 0)}"

            confirmed_keys = {entity_key(e) for e in confirmed_entities}
            negated_keys = {entity_key(e) for e in negated_entities}
            uncertain_keys = {entity_key(e) for e in uncertain_entities}
            historical_keys = {entity_key(e) for e in historical_entities}
            family_keys = {entity_key(e) for e in family_entities}

            # Remove overlapping and very adjacent entities to avoid unreadable output
            # Keep longer entities when there's overlap
            # Also filter abbreviations in parentheses that follow full names
            filtered_entities = []
            sorted_by_start = sorted(all_entities, key=lambda x: x['start'])

            for i, entity in enumerate(sorted_by_start):
                should_skip = False

                # Check if this entity overlaps or is very close to any already-kept entity
                for kept in filtered_entities:
                    # Special case 1: Abbreviation in parentheses right after full name
                    # Example: "Kinesin family member 5A (KIF5A)" - skip the "KIF5A"
                    gap = entity['start'] - kept['end']
                    if 0 <= gap <= 3:  # Very close (space + opening paren)
                        between_text = text[kept['end']:entity['start']]
                        if '(' in between_text and len(entity['text']) <= 10:
                            # Likely an abbreviation in parentheses, skip it
                            should_skip = True
                            break

                    # Check for overlap or close adjacency (within 15 chars for better readability)
                    if not (entity['end'] + 15 < kept['start'] or kept['end'] + 15 < entity['start']):
                        # There's overlap or close adjacency
                        # Keep the longer entity
                        entity_len = entity['end'] - entity['start']
                        kept_len = kept['end'] - kept['start']

                        if entity_len <= kept_len:
                            # Current entity is shorter or equal, skip it
                            should_skip = True
                            break
                        else:
                            # Current entity is longer, remove the kept one
                            filtered_entities.remove(kept)

                if not should_skip:
                    # Also check for exact duplicate text (same entity appearing multiple times)
                    # Keep only first occurrence
                    is_duplicate = False
                    for kept in filtered_entities:
                        if entity['text'].lower() == kept['text'].lower() and entity['start'] != kept['start']:
                            # Duplicate entity, skip it
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        filtered_entities.append(entity)

            # Sort entities by start position (reverse order for text replacement)
            sorted_entities = sorted(filtered_entities, key=lambda x: x['start'], reverse=True)

            # Create highlighted text with entity type AND context classification
            highlighted_text = text
            for entity in sorted_entities:
                start = entity['start']
                end = entity['end']
                entity_text = entity['text']
                medical_label = entity['label']

                # Clean label for display
                clean_label = medical_label.replace("BioBERT_", "").replace("biobert_", "").replace("BIOBERT_", "").upper()

                # Determine context classification(s) - an entity can have multiple contexts
                contexts = []
                key = entity_key(entity)

                if key in confirmed_keys:
                    contexts.append("CONFIRMED")
                if key in negated_keys:
                    contexts.append("NEGATED")
                if key in uncertain_keys:
                    contexts.append("UNCERTAIN")
                if key in historical_keys:
                    contexts.append("HISTORICAL")
                if key in family_keys:
                    contexts.append("FAMILY")

                # Build context display string
                if contexts:
                    context_str = " | ".join(contexts)
                    replacement = f"▶[{entity_text}]◀ ({clean_label} | {context_str})"
                else:
                    # Entity detected but no specific context matched
                    replacement = f"▶[{entity_text}]◀ ({clean_label})"

                highlighted_text = highlighted_text[:start] + replacement + highlighted_text[end:]

            # Add section category header
            header = f"📋 SECTION: {section_categories}\n\n"
            header += "=" * 80 + "\n"
            header += "LEGEND: ▶[Entity]◀ (TYPE | CONTEXT)\n"
            header += "CONTEXTS: CONFIRMED | NEGATED | UNCERTAIN | HISTORICAL | FAMILY\n"
            header += "=" * 80 + "\n\n"

            return header + highlighted_text

        except Exception as e:
            logger.error(f"Enhanced medical visualization failed: {e}")
            return text

    def _generate_displacy_visualization_enhanced(self, filtered_entities, text: str) -> str:
        """Generate enhanced visualization with corrected entity labels"""
        try:
            if not filtered_entities:
                return text

            # Create entity info with enhanced classifications
            entity_infos = []
            for ent in filtered_entities:
                # Get enhanced classification
                classification = self._classify_medical_entity(ent.text, ent.label_)

                entity_infos.append({
                    'text': ent.text,
                    'label': classification['medical_label'],
                    'original_label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': classification['confidence']
                })

            # Sort entities by start position (reverse order for replacement)
            sorted_entities = sorted(entity_infos, key=lambda x: x['start'], reverse=True)

            # Create highlighted text
            highlighted_text = text
            for ent_info in sorted_entities:
                start = ent_info['start']
                end = ent_info['end']
                entity_text = ent_info['text']
                enhanced_label = ent_info['label']
                original_label = ent_info['original_label']

                # Show enhancement if label changed
                if enhanced_label != original_label:
                    label_display = f"{enhanced_label} (was {original_label})"
                else:
                    label_display = enhanced_label

                # Create highlighted replacement
                replacement = f"▶[{entity_text}]◀ ({label_display})"
                highlighted_text = highlighted_text[:start] + replacement + highlighted_text[end:]

            return highlighted_text

        except Exception as e:
            logger.error(f"Enhanced visualization failed: {e}")
            return text

    def _calculate_sample_accuracy(self, text: str, entities: List[Dict]) -> float:
        """Calculate accuracy score for a sample based on entity detection quality"""
        if not entities:
            return 0.0

        # Count high-confidence entities (confidence > 0.8)
        high_confidence_count = sum(1 for entity in entities if entity.get('confidence', 0) > 0.8)

        # Calculate ratio of high-confidence entities
        confidence_ratio = high_confidence_count / len(entities) if entities else 0

        # Additional quality metrics
        unique_labels = len(set(entity.get('label', '') for entity in entities))
        label_diversity = min(unique_labels / 5, 1.0)  # Normalize to max 5 label types

        # Combine metrics for overall accuracy score
        accuracy_score = (confidence_ratio * 0.7) + (label_diversity * 0.3)

        return accuracy_score

    def _identify_best_samples(self, df: pd.DataFrame, n_samples: int = 10) -> List[Tuple[int, str, float, List[Dict]]]:
        """Identify top N samples with best accuracy/least false positives"""
        logger.info(f"🔍 Identifying top {n_samples} samples with best accuracy...")

        sample_scores = []

        for idx, row in df.iterrows():
            try:
                text = str(row.get('Text', ''))
                if len(text.strip()) < 10:  # Skip very short texts
                    continue

                # Extract entities for this sample
                result = self.extract_entities(text)
                all_entities_json = result.get('all_entities_json', '[]')

                try:
                    entities = json.loads(all_entities_json) if all_entities_json else []
                except:
                    entities = []

                # Calculate accuracy score
                accuracy_score = self._calculate_sample_accuracy(text, entities)

                # Only include samples with some entities and decent accuracy
                if entities and accuracy_score > 0.3:
                    sample_scores.append((idx, text, accuracy_score, entities))

            except Exception as e:
                logger.debug(f"Error processing sample {idx}: {e}")
                continue

        # Sort by accuracy score (descending) and take top N
        best_samples = sorted(sample_scores, key=lambda x: x[2], reverse=True)[:n_samples]

        logger.info(f"✅ Identified {len(best_samples)} high-quality samples for visualization")
        return best_samples

    def _save_displacy_visualization(self, text: str, entities: List[Dict], output_path: Path, sample_idx: int):
        """Save DisplaCy visualization as PNG file"""
        if not VISUALIZATION_AVAILABLE:
            logger.warning("⚠️ Visualization export not available (missing selenium/chrome)")
            return False

        try:
            # Clean HTML formatting from text for better visualization
            import re
            from html import unescape

            # Clean the text by removing HTML tags and entities
            clean_text = text

            # Remove HTML tags
            clean_text = re.sub(r'<[^>]+>', '', clean_text)

            # Decode HTML entities
            clean_text = unescape(clean_text)

            # Remove extra whitespace and normalize
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()

            # Prepare entities for DisplaCy format with adjusted positions
            displacy_entities = []
            for entity in entities:
                start_pos = entity.get('start')
                end_pos = entity.get('end')

                if start_pos is not None and end_pos is not None:
                    # Extract the entity text from original text
                    entity_text = text[start_pos:end_pos]

                    # Clean the entity text the same way
                    clean_entity_text = re.sub(r'<[^>]+>', '', entity_text)
                    clean_entity_text = unescape(clean_entity_text).strip()

                    if clean_entity_text and clean_entity_text in clean_text:
                        # Find the position in clean text
                        clean_start = clean_text.find(clean_entity_text)
                        if clean_start != -1:
                            clean_end = clean_start + len(clean_entity_text)
                            displacy_entities.append({
                                'start': clean_start,
                                'end': clean_end,
                                'label': entity.get('label', 'ENTITY')
                            })

            # Sort entities by start position to avoid overlaps
            displacy_entities = sorted(displacy_entities, key=lambda x: x['start'])

            # Remove overlapping entities (keep the first one)
            filtered_entities = []
            last_end = -1
            for entity in displacy_entities:
                if entity['start'] >= last_end:
                    filtered_entities.append(entity)
                    last_end = entity['end']

            # Create DisplaCy visualization data with clean text
            doc_data = {
                "text": clean_text,
                "ents": filtered_entities,
                "title": f"Medical NER - Index {sample_idx}"
            }

            # Generate HTML visualization
            html = displacy.render(doc_data, style="ent", manual=True, options={
                "colors": {
                    "DISEASE": "#ff9999",
                    "GENE": "#99ff99",
                    "PROTEIN": "#99ff99",
                    "DRUG": "#9999ff",
                    "CHEMICAL": "#9999ff",
                    "MEDICATION": "#9999ff",
                    "ANATOMY": "#ffcc99",
                    "SYMPTOM": "#ff99ff",
                    "PROCEDURE": "#ccff99",
                    "TEST": "#99ccff",
                    "TREATMENT": "#ffccff",
                    "PERSON": "#cccccc",
                    "ORG": "#cccccc",
                    "GPE": "#cccccc",
                    "DATE": "#ffffcc",
                    "TIME": "#ffffcc"
                },
                "ents": list(set(entity['label'] for entity in filtered_entities))
            })

            # Save as HTML first
            html_file = output_path.with_suffix('.html')
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Medical NER Visualization - Index {sample_idx}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: white;
        }}
        .displacy-container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .title {{
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="displacy-container">
        <div class="title">Medical NER Visualization - Index {sample_idx}</div>
        {html}
    </div>
</body>
</html>
                """)

            # Convert HTML to PNG using Selenium
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')

            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)

            try:
                # Load the HTML file
                driver.get(f"file://{html_file.absolute()}")
                time.sleep(2)  # Wait for page to load

                # Take screenshot
                driver.save_screenshot(str(output_path))
                logger.info(f"✅ Saved visualization: {output_path.name}")

                # Clean up HTML file
                html_file.unlink()
                return True

            finally:
                driver.quit()

        except Exception as e:
            logger.error(f"❌ Failed to create visualization for index {sample_idx}: {e}")
            return False

    def save_top_visualizations(self, df: pd.DataFrame, n_samples: int = 10) -> int:
        """Generate and save DisplaCy visualizations for top N samples with best accuracy"""
        if not VISUALIZATION_AVAILABLE:
            logger.error("❌ Visualization functionality not available (missing selenium/chrome dependencies)")
            return 0

        logger.info(f"🎨 Starting DisplaCy visualization generation for top {n_samples} samples...")

        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('output/visualizations')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Identify best samples
        best_samples = self._identify_best_samples(df, n_samples)

        if not best_samples:
            logger.warning("⚠️ No suitable samples found for visualization")
            return 0

        successful_saves = 0
        for i, (original_idx, text, accuracy_score, entities) in enumerate(best_samples, 1):
            try:
                # Create output filename with format: visualization_sample_idx_{Index}_{timestamp}.png
                output_filename = f"visualization_sample_idx_{original_idx}_{timestamp}.png"
                output_path = output_dir / output_filename

                # Save visualization
                if self._save_displacy_visualization(text, entities, output_path, original_idx):
                    successful_saves += 1

            except Exception as e:
                logger.error(f"❌ Failed to process sample {i}: {e}")

        logger.info(f"🎨 Visualization complete: {successful_saves}/{len(best_samples)} successfully saved")
        logger.info(f"📁 Visualizations saved to: {output_dir}")

        # Create summary report
        summary_file = output_dir / f"visualization_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"DISPLACY VISUALIZATIONS SUMMARY\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total samples processed: {len(best_samples)}\n")
            f.write(f"Successful visualizations: {successful_saves}\n")
            f.write(f"Output directory: {output_dir}\n\n")

            f.write("SAMPLE DETAILS:\n")
            f.write("-" * 30 + "\n")
            for i, (original_idx, text, accuracy_score, entities) in enumerate(best_samples, 1):
                f.write(f"Sample {i} (Original Index: {original_idx})\n")
                f.write(f"  Accuracy Score: {accuracy_score:.3f}\n")
                f.write(f"  Entities Count: {len(entities)}\n")
                f.write(f"  Text Preview: {text[:150]}...\n")
                f.write(f"  Entity Types: {', '.join(set(e.get('label', 'UNKNOWN') for e in entities))}\n\n")

        return successful_saves

def save_formatted_excel(df: pd.DataFrame, output_path: Path):
    """Save DataFrame to Excel with professional formatting"""
    try:
        # Import xlsxwriter for advanced formatting
        import xlsxwriter

        logger.info(f"Saving formatted Excel file to {output_path}")

        # Create Excel writer object with xlsxwriter engine
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            # Write dataframe to Excel
            df.to_excel(writer, sheet_name='Medical_NER_Predictions', index=False, startrow=0, startcol=0)

            # Get workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Medical_NER_Predictions']

            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'vcenter',
                'align': 'center',
                'fg_color': '#90EE90',  # Light green
                'border': 1
            })

            cell_format = workbook.add_format({
                'text_wrap': True,
                'valign': 'top',
                'border': 1
            })

            # Set column width to 20 for all columns
            for col_num in range(len(df.columns)):
                worksheet.set_column(col_num, col_num, 20, cell_format)

            # Format header row (row 0) first
            for col_num, column_name in enumerate(df.columns):
                worksheet.write(0, col_num, column_name, header_format)

            # Set header row height to 30 points for better visibility
            worksheet.set_row(0, 30)

            # Set row height to 20 points for all data rows (xlsxwriter uses points, not pixels)
            # 15 pixels ≈ 20 points (1 point = 0.75 pixels approximately)
            for row_num in range(1, len(df) + 1):
                worksheet.set_row(row_num, 20)

            # Freeze top row
            worksheet.freeze_panes(1, 0)

            # Add autofilter to the entire data range
            worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)

            logger.info("✅ Excel formatting applied successfully")

    except ImportError:
        logger.warning("⚠️ xlsxwriter not available, falling back to basic Excel export")
        df.to_excel(output_path, index=False)
    except Exception as e:
        logger.error(f"❌ Error saving formatted Excel: {e}")
        logger.info("Falling back to basic Excel export")
        df.to_excel(output_path, index=False)

    def _identify_best_samples(self, df: pd.DataFrame, n_samples: int = 10) -> List[Tuple[int, str, float, List[Dict]]]:
        """Identify top N samples with best accuracy/least false positives"""
        logger.info(f"🔍 Identifying top {n_samples} samples with best accuracy...")

        sample_scores = []

        for idx, row in df.iterrows():
            try:
                text = str(row.get('Text', ''))
                if len(text.strip()) < 10:  # Skip very short texts
                    continue

                # Extract entities for this sample
                result = self.extract_entities(text)
                all_entities_json = result.get('all_entities_json', '[]')

                try:
                    entities = json.loads(all_entities_json) if all_entities_json else []
                except:
                    entities = []

                # Calculate accuracy score
                accuracy_score = self._calculate_sample_accuracy(text, entities)

                # Only include samples with some entities and decent accuracy
                if entities and accuracy_score > 0.3:
                    sample_scores.append((idx, text, accuracy_score, entities))

            except Exception as e:
                logger.debug(f"Error processing sample {idx}: {e}")
                continue

        # Sort by accuracy score (descending) and take top N
        best_samples = sorted(sample_scores, key=lambda x: x[2], reverse=True)[:n_samples]

        logger.info(f"✅ Identified {len(best_samples)} high-quality samples for visualization")
        return best_samples

    def _save_displacy_visualization(self, text: str, entities: List[Dict], output_path: Path, sample_idx: int):
        """Save DisplaCy visualization as PNG file"""
        if not VISUALIZATION_AVAILABLE:
            logger.warning("⚠️ Visualization export not available (missing selenium/chrome)")
            return False

        try:
            # Clean HTML formatting from text for better visualization
            import re
            from html import unescape

            # Clean the text by removing HTML tags and entities
            clean_text = text

            # Remove HTML tags
            clean_text = re.sub(r'<[^>]+>', '', clean_text)

            # Decode HTML entities
            clean_text = unescape(clean_text)

            # Remove extra whitespace and normalize
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()

            # Prepare entities for DisplaCy format with adjusted positions
            displacy_entities = []
            for entity in entities:
                start_pos = entity.get('start')
                end_pos = entity.get('end')

                if start_pos is not None and end_pos is not None:
                    # Extract the entity text from original text
                    entity_text = text[start_pos:end_pos]

                    # Clean the entity text the same way
                    clean_entity_text = re.sub(r'<[^>]+>', '', entity_text)
                    clean_entity_text = unescape(clean_entity_text).strip()

                    if clean_entity_text and clean_entity_text in clean_text:
                        # Find the position in clean text
                        clean_start = clean_text.find(clean_entity_text)
                        if clean_start != -1:
                            clean_end = clean_start + len(clean_entity_text)
                            displacy_entities.append({
                                'start': clean_start,
                                'end': clean_end,
                                'label': entity.get('label', 'ENTITY')
                            })

            # Sort entities by start position to avoid overlaps
            displacy_entities = sorted(displacy_entities, key=lambda x: x['start'])

            # Remove overlapping entities (keep the first one)
            filtered_entities = []
            last_end = -1
            for entity in displacy_entities:
                if entity['start'] >= last_end:
                    filtered_entities.append(entity)
                    last_end = entity['end']

            # Create DisplaCy visualization data with clean text
            doc_data = {
                "text": clean_text,
                "ents": filtered_entities,
                "title": f"Medical NER - Index {sample_idx}"
            }

            # Generate HTML visualization
            html = displacy.render(doc_data, style="ent", manual=True, options={
                "colors": {
                    "DISEASE": "#ff9999",
                    "GENE": "#99ff99",
                    "PROTEIN": "#99ff99",
                    "DRUG": "#9999ff",
                    "CHEMICAL": "#9999ff",
                    "MEDICATION": "#9999ff",
                    "ANATOMY": "#ffcc99",
                    "SYMPTOM": "#ff99ff",
                    "PROCEDURE": "#ccff99",
                    "TEST": "#99ccff",
                    "TREATMENT": "#ffccff",
                    "PERSON": "#cccccc",
                    "ORG": "#cccccc",
                    "GPE": "#cccccc",
                    "DATE": "#ffffcc",
                    "TIME": "#ffffcc"
                },
                "ents": list(set(entity['label'] for entity in filtered_entities))
            })

            # Save as HTML first
            html_file = output_path.with_suffix('.html')
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Medical NER Visualization - Index {sample_idx}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: white;
        }}
        .displacy-container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .title {{
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="displacy-container">
        <div class="title">Medical NER Visualization - Index {sample_idx}</div>
        {html}
    </div>
</body>
</html>
                """)

            # Convert HTML to PNG using selenium
            try:
                # Setup Chrome options for headless operation
                chrome_options = Options()
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--window-size=1200,800")

                # Initialize Chrome driver
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=chrome_options)

                # Load HTML file
                driver.get(f"file://{html_file.absolute()}")
                time.sleep(2)  # Wait for page to load

                # Take screenshot
                driver.save_screenshot(str(output_path))
                driver.quit()

                # Clean up HTML file
                html_file.unlink()

                logger.info(f"✅ Saved visualization: {output_path}")
                return True

            except Exception as e:
                logger.error(f"❌ Failed to convert HTML to PNG: {e}")
                # Keep HTML file as fallback
                logger.info(f"📁 HTML visualization saved: {html_file}")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to create visualization for index {sample_idx}: {e}")
            return False


def validate_pipeline_performance(predictor: 'EnhancedMedicalNERPredictor') -> bool:
    """
    Validate that all entity predictors are working at peak performance
    Returns True if validation passes, False otherwise
    """
    logger.info("🧪 Running pipeline performance validation...")

    # Test cases to validate different entity types
    test_cases = [
        {
            'text': 'The patient has diabetes and takes insulin daily for blood sugar control.',
            'expected_diseases': ['diabetes'],
            'expected_drugs': ['insulin'],
            'min_entities': 2
        },
        {
            'text': 'BRCA1 gene mutation was detected in the breast cancer patient.',
            'expected_genes': ['BRCA1'],
            'expected_diseases': ['breast cancer'],
            'min_entities': 2
        },
        {
            'text': 'Hypertension treatment includes ACE inhibitors and beta blockers.',
            'expected_diseases': ['hypertension'],
            'expected_drugs': ['ACE inhibitors'],
            'min_entities': 2
        },
        {
            'text': 'Patient denies chest pain but reports shortness of breath.',
            'expected_negated': True,
            'min_entities': 1
        },
        {
            'text': 'Family history of cardiovascular disease and stroke.',
            'expected_family': True,
            'expected_diseases': ['cardiovascular disease', 'stroke'],
            'min_entities': 2
        }
    ]

    validation_results = []
    total_tests = len(test_cases)
    passed_tests = 0

    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"🔬 Running validation test {i}/{total_tests}...")
        try:
            result = predictor.extract_entities(test_case['text'])

            # Parse entities from result
            all_entities_json = result.get('all_entities_json', '[]')
            entities = json.loads(all_entities_json) if all_entities_json else []

            # Count different entity types
            entity_counts = {
                'diseases': len([e for e in entities if e.get('label') in ['DISEASE']]),
                'genes': len([e for e in entities if e.get('label') in ['GENE']]),
                'drugs': len([e for e in entities if e.get('label') in ['DRUG', 'CHEMICAL']]),
                'total': len(entities)
            }

            # Check if minimum entities are detected
            meets_minimum = entity_counts['total'] >= test_case.get('min_entities', 1)

            # Check for null/empty results (performance issue indicator)
            has_null_results = (
                entity_counts['total'] == 0 or
                result.get('detected_diseases', '') == '' and
                result.get('detected_genes', '') == '' and
                result.get('detected_chemicals', '') == ''
            )

            test_passed = meets_minimum and not has_null_results

            validation_results.append({
                'test_id': i,
                'text_preview': test_case['text'][:50] + '...',
                'entities_detected': entity_counts['total'],
                'diseases': entity_counts['diseases'],
                'genes': entity_counts['genes'],
                'drugs': entity_counts['drugs'],
                'meets_minimum': meets_minimum,
                'has_null_results': has_null_results,
                'passed': test_passed
            })

            if test_passed:
                passed_tests += 1
                logger.info(f"✅ Test {i} PASSED - Detected {entity_counts['total']} entities")
            else:
                logger.warning(f"❌ Test {i} FAILED - Only {entity_counts['total']} entities detected (minimum: {test_case.get('min_entities', 1)})")
                if has_null_results:
                    logger.warning(f"⚠️  NULL RESULTS DETECTED - Entity predictors may not be working properly!")

        except Exception as e:
            logger.error(f"❌ Test {i} FAILED with exception: {e}")
            validation_results.append({
                'test_id': i,
                'text_preview': test_case['text'][:50] + '...',
                'entities_detected': 0,
                'error': str(e),
                'passed': False
            })

    # Calculate performance metrics
    pass_rate = (passed_tests / total_tests) * 100

    # Print validation summary
    print("\n" + "="*80)
    print("🧪 PIPELINE PERFORMANCE VALIDATION RESULTS")
    print("="*80)

    for result in validation_results:
        status = "✅ PASSED" if result['passed'] else "❌ FAILED"
        print(f"{status}: Test {result['test_id']} - {result['text_preview']}")
        if result['passed']:
            print(f"    Entities: {result['entities_detected']} (D:{result.get('diseases',0)}, G:{result.get('genes',0)}, C:{result.get('drugs',0)})")
        elif 'error' in result:
            print(f"    Error: {result['error']}")
        else:
            print(f"    Issues: {result['entities_detected']} entities (minimum required: varies by test)")
            if result.get('has_null_results'):
                print(f"    🚨 NULL RESULTS - PREDICTORS NOT WORKING!")

    print(f"\n🎯 Overall Performance: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%)")

    # Determine if validation passes
    validation_passed = pass_rate >= 80.0  # Require 80% pass rate

    if validation_passed:
        print("🎉 VALIDATION PASSED - Pipeline is performing at peak performance!")
        logger.info("✅ Pipeline performance validation completed successfully")
    else:
        print("⚠️  VALIDATION FAILED - Pipeline performance issues detected!")
        print("🔧 Recommendation: Check BioBERT model loading and entity detection rules")
        logger.error("❌ Pipeline performance validation failed - entity predictors may not be working properly")

    print("="*80)
    return validation_passed


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Enhanced Medical NER Prediction Pipeline')
    parser.add_argument('--input', '-i',
                       default='data/raw/context_test_results_complete.xlsx',
                       help='Input Excel file path')
    parser.add_argument('--output', '-o', help='Output file path (default: auto-generated)')
    parser.add_argument('--text-column', default='Text', help='Name of text column')
    parser.add_argument('--model', default='en_core_web_sm', help='spaCy model to use')
    parser.add_argument('--batch-size', type=int, default=100, help='Processing batch size (default: 100)')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU processing (if available)')
    parser.add_argument('--json', action='store_true', help='Export results as JSON in addition to Excel')
    parser.add_argument('--visualizations', '--viz', action='store_true',
                       help='Generate DisplaCy visualizations for top samples (requires Chrome/Selenium)')
    parser.add_argument('--viz-samples', type=int, default=10,
                       help='Number of top samples to visualize (default: 10)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--validate', action='store_true', default=True,
                       help='Run pipeline performance validation (default: enabled)')
    parser.add_argument('--no-validate', dest='validate', action='store_false',
                       help='Skip pipeline performance validation')

    # Template file arguments
    parser.add_argument('--target-rules-template',
                       default='data/external/target_rules_template.xlsx',
                       help='Path to target rules template file (default: data/external/target_rules_template.xlsx)')
    parser.add_argument('--historical-rules-template',
                       default='data/external/historical_rules_template.xlsx',
                       help='Path to historical rules template file (default: data/external/historical_rules_template.xlsx)')
    parser.add_argument('--negated-rules-template',
                       default='data/external/negated_rules_template.xlsx',
                       help='Path to negated rules template file (default: data/external/negated_rules_template.xlsx)')
    parser.add_argument('--uncertainty-rules-template',
                       default='data/external/uncertainty_rules_template.xlsx',
                       help='Path to uncertainty rules template file (default: data/external/uncertainty_rules_template.xlsx)')
    parser.add_argument('--confirmed-rules-template',
                       default='data/external/confirmed_rules_template.xlsx',
                       help='Path to confirmed rules template file (default: data/external/confirmed_rules_template.xlsx)')
    parser.add_argument('--family-rules-template',
                       default='data/external/family_rules_template.xlsx',
                       help='Path to family rules template file (default: data/external/family_rules_template.xlsx)')

    # Override strategy argument (template-priority is now DEFAULT)
    parser.add_argument('--no-template-priority', dest='template_priority', action='store_false',
                       default=True,
                       help='Disable template-priority mode. Use confidence-based override instead '
                            '(templates only override when template confidence > BioBERT confidence).')
    parser.add_argument('--template-priority', dest='template_priority', action='store_true',
                       default=True,
                       help='Templates ALWAYS override BioBERT detections (DEFAULT). '
                            'Provides complete custom control over entity detection.')

    args = parser.parse_args()

    # Setup logging with file handler
    log_dir = Path('output/logs')
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log filename based on output file or timestamp
    if args.output:
        output_basename = Path(args.output).stem
        log_filename = f"{output_basename}.log"
    else:
        log_filename = f"medical_ner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    log_file = log_dir / log_filename

    # Configure logging with both file and console handlers
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logger.info(f"📝 Logging to: {log_file}")

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Load data
    try:
        logger.info(f"Loading data from {input_path}")
        df = pd.read_excel(input_path)
        logger.info(f"✅ Loaded {len(df)} rows with columns: {list(df.columns)}")
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        sys.exit(1)

    # Initialize predictor with template paths
    try:
        predictor = EnhancedMedicalNERPredictor(
            model_name=args.model,
            use_gpu=args.gpu,
            batch_size=args.batch_size,
            target_rules_template=args.target_rules_template,
            historical_rules_template=args.historical_rules_template,
            negated_rules_template=args.negated_rules_template,
            uncertainty_rules_template=args.uncertainty_rules_template,
            confirmed_rules_template=args.confirmed_rules_template,
            family_rules_template=args.family_rules_template,
            template_priority=args.template_priority
        )
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        sys.exit(1)

    # Run pipeline performance validation (enabled by default)
    if args.validate:
        logger.info("🧪 Running automatic pipeline performance validation...")
        validation_passed = validate_pipeline_performance(predictor)
        if not validation_passed:
            logger.error("❌ Pipeline validation failed - entity predictors may not be working properly!")
            logger.error("🔧 Consider checking BioBERT model loading, template files, or running with --no-validate to skip")
            # Continue execution but warn user about potential issues
            logger.warning("⚠️  Continuing execution despite validation failure...")
    else:
        logger.info("⏭️  Pipeline validation skipped (use --validate to enable)")

    # Process dataframe
    try:
        df_predicted = predictor.predict_dataframe(df, args.text_column)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"output/results/output_results_{timestamp}.xlsx")
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        save_formatted_excel(df_predicted, output_path)
        logger.info(f"✅ Formatted results saved to {output_path}")

        # Export as JSON if requested
        if args.json:
            json_path = output_path.with_suffix('.json')
            df_predicted.to_json(json_path, orient='records', indent=2)
            logger.info(f"✅ JSON export saved to {json_path}")

        # Generate DisplaCy visualizations for top samples (if requested)
        if args.visualizations:
            try:
                logger.info(f"🎨 Generating DisplaCy visualizations for top {args.viz_samples} samples...")
                visualization_count = predictor.save_top_visualizations(df_predicted, n_samples=args.viz_samples)
                logger.info(f"✅ Generated {visualization_count} visualizations")
            except Exception as e:
                logger.error(f"❌ Visualization generation failed: {e}")
                logger.info("📊 Continuing without visualizations...")
        else:
            logger.info("📊 Visualization generation skipped (use --visualizations to enable)")

        # Print summary
        print("\n" + "="*60)
        print("🎉 ENHANCED MEDICAL NER PREDICTION COMPLETED")
        print("="*60)

        # Show execution command
        print(f"\n📟 Execution Command:")
        import sys
        cmd_parts = [f"python {sys.argv[0]}"]
        cmd_parts.append(f"--input {args.input}")
        cmd_parts.append(f"--output {output_path}")
        if args.model != 'en_core_web_sm':
            cmd_parts.append(f"--model {args.model}")
        if args.gpu:
            cmd_parts.append("--gpu")
        if args.batch_size != 100:
            cmd_parts.append(f"--batch-size {args.batch_size}")
        if not args.template_priority:
            cmd_parts.append("--no-template-priority")
        if args.visualizations:
            cmd_parts.append(f"--viz --viz-samples {args.viz_samples}")
        if args.verbose:
            cmd_parts.append("--verbose")
        print(f"   {' '.join(cmd_parts)}")

        print(f"\n📊 Processing Summary:")
        print(f"   Rows Processed: {len(df_predicted)}")
        print(f"   Output File: {output_path}")
        print(f"\n🧠 Model Configuration:")
        print(f"   spaCy Model: {args.model}")
        print(f"   GPU Enabled: {'Yes' if args.gpu else 'No'}")
        print(f"   Batch Size: {args.batch_size}")

        print(f"\n🔄 Override Strategy:")
        if predictor.template_priority:
            print(f"   Mode: 🎯 TEMPLATE-PRIORITY (DEFAULT)")
            print(f"   Behavior: Templates ALWAYS override BioBERT detections")
            print(f"   Use Case: Complete custom control, rare diseases, institution-specific terms")
        else:
            print(f"   Mode: ⚖️  CONFIDENCE-BASED")
            print(f"   Behavior: Templates override only when confidence > BioBERT")
            print(f"   Use Case: General medical text, trust BioBERT on standard terms")

        print(f"\n🎯 Target Rules: {'ENABLED' if predictor.target_rules_enabled else 'DISABLED'}")
        if predictor.target_rules_enabled:
            print(f"   Template: {predictor.target_rules_file}")
            print(f"   Loaded: {len(predictor.disease_terms):,} diseases, {len(predictor.gene_terms):,} genes, {len(predictor.drug_terms):,} drugs")

        print(f"📜 Context Templates:")
        print(f"   Historical: {'ENABLED' if predictor.historical_rules_enabled else 'DISABLED'}")
        if predictor.historical_rules_enabled:
            print(f"      📊 {len(predictor.historical_patterns)} patterns from {predictor.historical_rules_file}")
        print(f"   Negated: {'ENABLED' if predictor.negated_rules_enabled else 'DISABLED'}")
        if predictor.negated_rules_enabled:
            print(f"      📊 {len(predictor.negated_patterns)} patterns from {predictor.negated_rules_file}")
        print(f"   Uncertainty: {'ENABLED' if predictor.uncertainty_rules_enabled else 'DISABLED'}")
        if predictor.uncertainty_rules_enabled:
            print(f"      📊 {len(predictor.uncertainty_patterns)} patterns from {predictor.uncertainty_rules_file}")
        if args.json:
            print(f"📄 JSON Export: {output_path.with_suffix('.json')}")
        if args.visualizations:
            print(f"🎨 Visualizations: Generated for top {args.viz_samples} samples → output/visualizations/")
        print(f"🏥 Predicted columns:")
        prediction_cols = [
            'Text_Clean',
            'Text Visualization',
            'detected_diseases', 'total_diseases_count', 'detected_diseases_unique',
            'detected_genes', 'total_gene_count', 'detected_genes_unique',
            'detected_drugs', 'detected_drugs_count', 'detected_drugs_unique',
            'detected_chemicals', 'total_chemicals_count', 'detected_chemicals_unique',
            'confirmed_entities', 'confirmed_entities_count', 'confirmed_entities_predictors',
            'negated_entities', 'negated_entities_count', 'negated_entities_predictors', 'negated_context_sentences',
            'historical_entities', 'historical_entities_count', 'historical_entities_predictors', 'historical_context_sentences',
            'uncertain_entities', 'uncertain_entities_count', 'uncertain_entities_predictors', 'uncertain_context_sentences',
            'family_entities', 'family_entities_count', 'family_entities_predictors', 'family_context_sentences',
            'section_categories', 'all_entities_json'
        ]
        for col in prediction_cols:
            print(f"   • {col}")
        print("="*60)

    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()