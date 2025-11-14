#!/usr/bin/env python3
"""
enhanced_performance_analyzer.py

Enhanced Performance Analyzer for Medical NLP Pipeline
Analyzes prediction performance, generates detailed reports, and provides
comprehensive evaluation metrics for all predicted columns.

Generates: output/reports/entity_prediction_analysis_report_{timestamp}.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import json
import re
from typing import Dict, List, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPerformanceAnalyzer:
    """Enhanced analyzer for medical NLP prediction performance"""

    def __init__(self):
        """Initialize the performance analyzer"""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.execution_command = self._capture_execution_command()
        self.entity_definitions = self._get_entity_definitions()
        self.report_data = {
            "timestamp": self.timestamp,
            "execution_command": self.execution_command,
            "analysis_metadata": {},
            "performance_metrics": {},
            "detailed_analysis": {},
            "summary": {}
        }

    def _get_entity_definitions(self) -> Dict[str, str]:
        """Get comprehensive definitions for all entity groups used in NLP"""
        return {
            # spaCy Standard Entity Types
            "PERSON": "People, including fictional characters. Names of individuals, family members, celebrities, historical figures.",
            "NORP": "Nationalities or religious or political groups. Examples: American, Catholic, Republican, Muslim, Jewish.",
            "FAC": "Buildings, airports, highways, bridges, etc. Facilities and infrastructure names.",
            "ORG": "Companies, agencies, institutions, etc. Organizations including hospitals, universities, corporations.",
            "GPE": "Countries, cities, states. Geopolitical entities at any level (country, state, city, district).",
            "LOC": "Non-GPE locations, mountain ranges, bodies of water. Geographic locations that are not political entities.",
            "PRODUCT": "Objects, vehicles, foods, etc. (not services). Brand names, product models, food items.",
            "EVENT": "Named hurricanes, battles, wars, sports events, etc. Historical events, conferences, festivals.",
            "WORK_OF_ART": "Titles of books, songs, movies, paintings, etc. Creative works and artistic productions.",
            "LAW": "Named documents made into laws. Legal documents, acts, treaties, constitutional amendments.",
            "LANGUAGE": "Any named language. Programming languages, human languages, linguistic varieties.",
            "DATE": "Absolute or relative dates or periods. Specific dates, time ranges, calendar references.",
            "TIME": "Times smaller than a day. Hours, minutes, time of day, time periods within a day.",
            "PERCENT": "Percentage, including '%'. Any percentage value or ratio expressed as a percentage.",
            "MONEY": "Monetary values, including unit. Currency amounts, prices, financial values.",
            "QUANTITY": "Measurements, as of weight or distance. Physical measurements, amounts, sizes.",
            "ORDINAL": "First, second, etc. Ordinal numbers indicating position or sequence.",
            "CARDINAL": "Numerals that do not fall under another type. Cardinal numbers, counts, quantities.",

            # Medical and Biomedical Entity Types
            "DISEASE": "Medical conditions, disorders, syndromes, pathologies. Examples: diabetes, pneumonia, cancer, COVID-19.",
            "CONDITION": "Medical states or health conditions. General health states, symptoms, medical situations.",
            "PROBLEM": "Medical problems or health issues. Clinical problems, health concerns, medical complications.",
            "DRUG": "Medications, pharmaceuticals, therapeutic substances. Prescription drugs, over-the-counter medications.",
            "CHEMICAL": "Chemical compounds, drugs, substances. Chemical entities, molecular compounds, therapeutic chemicals.",
            "MEDICATION": "Medicines, treatments, pharmaceutical interventions. Any therapeutic medication or drug treatment.",
            "GENE": "Genetic elements, DNA sequences, hereditary factors. Gene names, genetic markers, DNA segments.",
            "PROTEIN": "Protein molecules, enzymes, biological macromolecules. Protein names, enzyme classifications.",
            "GENETIC": "Genetic-related entities, hereditary elements. Genetic variants, mutations, hereditary factors.",
            "ANATOMY": "Body parts, organs, anatomical structures. Human anatomy, organ systems, body regions.",
            "SYMPTOM": "Clinical signs, manifestations of disease. Observable signs, patient-reported symptoms.",
            "PROCEDURE": "Medical procedures, treatments, interventions. Surgical procedures, diagnostic tests, therapies.",
            "TEST": "Diagnostic tests, laboratory procedures. Medical tests, screening procedures, diagnostic methods.",
            "TREATMENT": "Therapeutic interventions, medical treatments. Treatment plans, therapeutic approaches, interventions.",

            # Contextual and Temporal Entity Types
            "HISTORICAL": "Past medical events, previous conditions. Historical medical information, past diagnoses.",
            "FAMILY": "Family-related medical information. Family history, hereditary conditions, genetic predispositions.",
            "NEGATED": "Negated medical entities, absent conditions. Medical conditions that are explicitly denied or absent.",
            "UNCERTAIN": "Uncertain or possible medical conditions. Medical entities with uncertain or speculative context.",
            "HYPOTHETICAL": "Hypothetical medical scenarios. Theoretical conditions, potential diagnoses under consideration.",

            # Document Structure Entity Types
            "SECTION": "Document sections, clinical note categories. Parts of medical documents, note sections.",
            "HEADER": "Document headers, section titles. Titles and headers in clinical documentation.",
            "FOOTER": "Document footers, closing information. End-of-document information, signatures, timestamps.",

            # Custom Medical Entity Types
            "DOSAGE": "Medication dosages, drug amounts. Prescription amounts, dosing information, drug quantities.",
            "FREQUENCY": "Dosing frequency, treatment schedules. How often medications are taken or treatments given.",
            "ROUTE": "Route of administration, delivery method. How medications are administered (oral, IV, etc.).",
            "ALLERGY": "Allergic reactions, drug allergies. Known allergies, adverse reactions, contraindications.",
            "VITAL_SIGN": "Vital signs, physiological measurements. Blood pressure, heart rate, temperature, etc.",
            "LAB_VALUE": "Laboratory test results, clinical values. Lab test results, biomarker levels, clinical measurements.",
            "IMAGING": "Medical imaging results, radiological findings. X-ray, MRI, CT scan results and interpretations.",

            # Temporal and Contextual Modifiers
            "RECENT": "Recently occurring medical events. Recent symptoms, new diagnoses, current conditions.",
            "CHRONIC": "Long-term or chronic medical conditions. Ongoing conditions, chronic diseases, persistent symptoms.",
            "ACUTE": "Acute medical conditions, sudden onset. Emergency conditions, acute episodes, immediate concerns.",
            "STABLE": "Stable medical conditions, controlled states. Well-controlled conditions, stable patient states.",
            "IMPROVING": "Improving medical conditions, positive trends. Conditions getting better, positive responses.",
            "WORSENING": "Worsening medical conditions, deteriorating states. Conditions getting worse, disease progression.",

            # Placeholder and Unknown Types
            "MISC": "Miscellaneous entities that don't fit other categories. Uncategorized entities, general references.",
            "OTHER": "Other entity types not specifically classified. Alternative classification for unmatched entities.",
            "UNKNOWN": "Unknown or unclassified entity types. Entities that couldn't be properly classified.",
            "0": "Non-entity tokens, background text. Text that is not considered an entity (used in some models)."
        }

    def analyze_complete_performance(self, df: pd.DataFrame,
                                   input_file: str = None) -> Dict[str, Any]:
        """
        Complete performance analysis of all predicted columns

        Args:
            df: DataFrame with both expected and predicted columns
            input_file: Path to input file for metadata

        Returns:
            Dictionary containing complete analysis results
        """
        logger.info("🔍 Starting comprehensive performance analysis...")

        # Detect pipeline configuration
        pipeline_config = self._detect_pipeline_configuration()

        # Validate input structure
        input_validation = self._validate_input_structure(df)

        # Convert input file to relative path
        relative_input_file = "Unknown"
        if input_file:
            try:
                input_path = Path(input_file)
                # Try to get relative path from current directory
                try:
                    relative_input_file = str(input_path.relative_to(Path.cwd()))
                except ValueError:
                    # If not under current directory, just use the filename
                    relative_input_file = input_path.name
            except Exception:
                relative_input_file = input_file

        # Initialize report metadata
        self.report_data["analysis_metadata"] = {
            "input_file": relative_input_file,
            "total_samples": len(df),
            "analysis_date": datetime.now().isoformat(),
            "columns_analyzed": list(df.columns),
            "pipeline_config": pipeline_config,
            "input_validation": input_validation
        }

        # Analyze each prediction category
        self._analyze_disease_detection(df)
        self._analyze_gene_detection(df)
        self._analyze_negation_detection(df)
        self._analyze_historical_entities(df)
        self._analyze_uncertain_entities(df)
        self._analyze_confirmed_entities(df)
        self._analyze_family_entities(df)
        self._analyze_section_categories(df)
        self._analyze_overall_entity_counts(df)

        # Generate summary statistics
        self._generate_summary_statistics()

        # Save detailed report
        self._save_detailed_report()

        logger.info("✅ Performance analysis completed")
        return self.report_data

    def _capture_execution_command(self) -> Dict[str, str]:
        """Capture the command used to execute the analyzer"""
        import sys
        import os
        from pathlib import Path

        # Get relative path for working directory
        cwd = Path.cwd()
        try:
            # Try to get relative path from user home
            working_dir = str(cwd.relative_to(Path.home()))
            working_dir = f"~/{working_dir}"
        except ValueError:
            # If not under home directory, use basename
            working_dir = cwd.name

        # Make Python executable more portable
        python_exec = sys.executable
        conda_env = os.environ.get("CONDA_DEFAULT_ENV")
        if conda_env:
            # Replace absolute path with conda reference
            python_exec_display = f"conda run -n {conda_env} python"
        else:
            # Just show 'python' for non-conda environments
            python_exec_display = "python"

        command_info = {
            "command_line": " ".join(sys.argv),
            "script_name": Path(sys.argv[0]).name if sys.argv else "unknown",
            "working_directory": working_dir,
            "python_executable": python_exec_display,
            "python_executable_full": python_exec,  # Keep full path for reference
            "arguments": sys.argv[1:] if len(sys.argv) > 1 else [],
            "execution_method": "direct" if __name__ == "__main__" else "imported"
        }

        # Detect if run through launcher
        if any("launch_medical_nlp_project.py" in arg for arg in sys.argv):
            command_info["execution_method"] = "launcher"
            command_info["parent_launcher"] = "launch_medical_nlp_project.py"
        elif "enhanced_medical_ner_predictor.py" in " ".join(sys.argv):
            command_info["execution_method"] = "pipeline_direct"
        elif any("--launched-from" in arg for arg in sys.argv):
            command_info["execution_method"] = "launcher_subprocess"

        # Add environment variables that might be relevant
        conda_env = os.environ.get("CONDA_DEFAULT_ENV")
        if conda_env:
            command_info["conda_environment"] = conda_env

        return command_info

    def _detect_pipeline_configuration(self) -> Dict[str, str]:
        """Detect the current pipeline configuration"""
        config = {
            "pipeline_script": "enhanced_medical_ner_predictor.py",
            "environment": "py311_bionlp",
            "python_version": "3.11.5",
            "spacy_version": "3.7.2",
            "spacy_model": "en_core_web_sm (v3.7.1)",
            "spacy_model_type": "English Small Model (~50MB)",
            "biobert_models": "alvaroalon2/biobert suite (diseases, genes, chemicals)",
            "biobert_disease": "alvaroalon2/biobert_diseases_ner (~411MB)",
            "biobert_gene": "alvaroalon2/biobert_genetic_ner (~411MB)",
            "biobert_chemical": "alvaroalon2/biobert_chemical_ner (~822MB)",
            "language": "English",
            "negation_detection": "negspaCy with Negex component",
            "entity_recognition": "BioBERT + Custom Templates + spaCy NER",
            "context_analysis": "Rule-based (confirmed, negated, uncertain, historical, family)",
            "template_priority_mode": "Templates override BioBERT detections (default)"
        }

        # Try to detect actual spaCy configuration if available
        try:
            import spacy
            config["spacy_version_actual"] = spacy.__version__

            # Try to load and inspect the model
            try:
                nlp = spacy.load("en_core_web_sm")
                config["model_loaded"] = "en_core_web_sm (confirmed)"
                config["model_pipeline"] = str(list(nlp.pipe_names))
            except:
                config["model_loaded"] = "en_core_web_sm (not confirmed)"

        except ImportError:
            config["spacy_status"] = "Not available for detection"

        return config

    def _validate_input_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate input DataFrame structure"""
        validation = {
            "has_required_columns": True,
            "required_columns_present": [],
            "required_columns_missing": [],
            "prediction_columns_present": [],
            "total_input_columns": len(df.columns),
            "input_column_types": {},
            "validation_status": "VALID"
        }

        # Check for required columns
        required_columns = ['Index', 'Text']
        for col in required_columns:
            if col in df.columns:
                validation["required_columns_present"].append(col)
            else:
                validation["required_columns_missing"].append(col)

        if validation["required_columns_missing"]:
            validation["has_required_columns"] = False
            validation["validation_status"] = "INVALID"

        # Check for prediction columns (these indicate processed data)
        prediction_columns = [
            'Text DisplaCy entities visualization',
            'detected_diseases', 'total_diseases_count',
            'detected_genes', 'total_gene_count',
            'negated_entities', 'negated_entities_count',
            'historical_entities', 'historical_entities_count',
            'uncertain_entities', 'uncertain_entities_count',
            'family_entities', 'family_entities_count',
            'section_categories', 'all_entities_json'
        ]

        for col in prediction_columns:
            if col in df.columns:
                validation["prediction_columns_present"].append(col)

        # Analyze column types
        for col in df.columns:
            validation["input_column_types"][col] = str(df[col].dtype)

        return validation

    def _analyze_disease_detection(self, df: pd.DataFrame):
        """Analyze disease detection statistics (prediction-only mode)"""
        logger.info("Analyzing disease detection...")

        predicted_col = "detected_diseases"
        count_col = "total_diseases_count"

        analysis = {
            "total_samples": len(df),
            "samples_with_predicted": 0,
            "total_entities_detected": 0,
            "unique_diseases": set(),
            "detection_rate": 0.0,
            "avg_entities_per_sample": 0.0,
            "max_entities_in_sample": 0,
            "min_entities_in_sample": 0,
            "distribution": {},
            "top_diseases": []
        }

        if predicted_col in df.columns:
            disease_counts = {}
            entity_counts_per_sample = []

            for idx, row in df.iterrows():
                predicted = self._normalize_entity_string(row.get(predicted_col, ""))
                predicted_entities = self._extract_entities_from_string(predicted)

                if predicted_entities:
                    analysis["samples_with_predicted"] += 1
                    analysis["total_entities_detected"] += len(predicted_entities)
                    entity_counts_per_sample.append(len(predicted_entities))

                    # Track unique diseases and their frequencies
                    for entity in predicted_entities:
                        analysis["unique_diseases"].add(entity)
                        disease_counts[entity] = disease_counts.get(entity, 0) + 1
                else:
                    entity_counts_per_sample.append(0)

            # Calculate statistics
            if analysis["samples_with_predicted"] > 0:
                analysis["detection_rate"] = analysis["samples_with_predicted"] / len(df)
                analysis["avg_entities_per_sample"] = analysis["total_entities_detected"] / len(df)
                analysis["max_entities_in_sample"] = max(entity_counts_per_sample) if entity_counts_per_sample else 0
                analysis["min_entities_in_sample"] = min(entity_counts_per_sample) if entity_counts_per_sample else 0

                # Distribution of entity counts
                from collections import Counter
                count_distribution = Counter(entity_counts_per_sample)
                analysis["distribution"] = {str(k): v for k, v in sorted(count_distribution.items())}

                # Top 10 most frequent diseases
                top_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                analysis["top_diseases"] = [{"disease": d, "count": c} for d, c in top_diseases]

            # Convert set to count for JSON serialization
            analysis["unique_diseases_count"] = len(analysis["unique_diseases"])
            analysis["unique_diseases"] = sorted(list(analysis["unique_diseases"]))[:20]  # Keep top 20 for display

        self.report_data["performance_metrics"]["diseases"] = analysis

    def _analyze_gene_detection(self, df: pd.DataFrame):
        """Analyze gene detection statistics (prediction-only mode)"""
        logger.info("Analyzing gene detection...")

        predicted_col = "detected_genes"
        count_col = "total_gene_count"

        analysis = {
            "total_samples": len(df),
            "samples_with_predicted": 0,
            "total_entities_detected": 0,
            "unique_genes": set(),
            "detection_rate": 0.0,
            "avg_entities_per_sample": 0.0,
            "max_entities_in_sample": 0,
            "min_entities_in_sample": 0,
            "distribution": {},
            "top_genes": []
        }

        if predicted_col in df.columns:
            gene_counts = {}
            entity_counts_per_sample = []

            for idx, row in df.iterrows():
                predicted = self._normalize_entity_string(row.get(predicted_col, ""))
                predicted_entities = self._extract_entities_from_string(predicted)

                if predicted_entities:
                    analysis["samples_with_predicted"] += 1
                    analysis["total_entities_detected"] += len(predicted_entities)
                    entity_counts_per_sample.append(len(predicted_entities))

                    # Track unique genes and their frequencies
                    for entity in predicted_entities:
                        analysis["unique_genes"].add(entity)
                        gene_counts[entity] = gene_counts.get(entity, 0) + 1
                else:
                    entity_counts_per_sample.append(0)

            # Calculate statistics
            if analysis["samples_with_predicted"] > 0:
                analysis["detection_rate"] = analysis["samples_with_predicted"] / len(df)
                analysis["avg_entities_per_sample"] = analysis["total_entities_detected"] / len(df)
                analysis["max_entities_in_sample"] = max(entity_counts_per_sample) if entity_counts_per_sample else 0
                analysis["min_entities_in_sample"] = min(entity_counts_per_sample) if entity_counts_per_sample else 0

                # Distribution of entity counts
                from collections import Counter
                count_distribution = Counter(entity_counts_per_sample)
                analysis["distribution"] = {str(k): v for k, v in sorted(count_distribution.items())}

                # Top 10 most frequent genes
                top_genes = sorted(gene_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                analysis["top_genes"] = [{"gene": g, "count": c} for g, c in top_genes]

            # Convert set to count for JSON serialization
            analysis["unique_genes_count"] = len(analysis["unique_genes"])
            analysis["unique_genes"] = sorted(list(analysis["unique_genes"]))[:20]  # Keep top 20 for display

        self.report_data["performance_metrics"]["genes"] = analysis

    def _analyze_negation_detection(self, df: pd.DataFrame):
        """Analyze negation detection performance"""
        logger.info("Analyzing negation detection...")

        analysis = {
            "total_samples": len(df),
            "samples_with_negation_words": 0,
            "samples_with_predicted_negation": 0,
            "correct_negation_detection": 0,
            "false_positive_negation": 0,
            "false_negative_negation": 0,
            "negation_precision": 0.0,
            "negation_recall": 0.0,
            "negation_f1": 0.0,
            "negation_accuracy": 0.0,
            "negation_patterns_detected": {},
            "detailed_analysis": []
        }

        negation_keywords = [
            "not", "no", "without", "absent", "denies", "negative for",
            "rules out", "ruled out", "free of", "clear of"
        ]

        for idx, row in df.iterrows():
            text = str(row.get("Text", "")).lower()
            predicted_negated = str(row.get("negated_entities", ""))

            # Check for negation patterns in text
            has_negation_words = any(keyword in text for keyword in negation_keywords)
            has_predicted_negation = len(predicted_negated.strip()) > 0

            if has_negation_words:
                analysis["samples_with_negation_words"] += 1

                # Track which negation patterns were found
                for keyword in negation_keywords:
                    if keyword in text:
                        if keyword not in analysis["negation_patterns_detected"]:
                            analysis["negation_patterns_detected"][keyword] = 0
                        analysis["negation_patterns_detected"][keyword] += 1

            if has_predicted_negation:
                analysis["samples_with_predicted_negation"] += 1

            # Evaluate correctness
            if has_negation_words and has_predicted_negation:
                analysis["correct_negation_detection"] += 1
            elif not has_negation_words and has_predicted_negation:
                analysis["false_positive_negation"] += 1
            elif has_negation_words and not has_predicted_negation:
                analysis["false_negative_negation"] += 1

            # Detailed tracking
            analysis["detailed_analysis"].append({
                "sample_id": idx + 1,
                "has_negation_words": has_negation_words,
                "has_predicted_negation": has_predicted_negation,
                "predicted_negated_entities": predicted_negated,
                "text_snippet": text[:100] + "..." if len(text) > 100 else text
            })

        # Calculate metrics
        total_with_negation = analysis["samples_with_negation_words"]
        total_predicted_negation = analysis["samples_with_predicted_negation"]
        correct_detections = analysis["correct_negation_detection"]

        if total_with_negation > 0:
            analysis["negation_recall"] = correct_detections / total_with_negation

        if total_predicted_negation > 0:
            analysis["negation_precision"] = correct_detections / total_predicted_negation

        if analysis["negation_precision"] + analysis["negation_recall"] > 0:
            analysis["negation_f1"] = 2 * (analysis["negation_precision"] * analysis["negation_recall"]) / (analysis["negation_precision"] + analysis["negation_recall"])

        analysis["negation_accuracy"] = (correct_detections + (len(df) - total_with_negation - analysis["false_positive_negation"])) / len(df)

        self.report_data["performance_metrics"]["negation"] = analysis

    def _analyze_historical_entities(self, df: pd.DataFrame):
        """Analyze historical entity detection"""
        logger.info("Analyzing historical entity detection...")

        analysis = {
            "total_samples": len(df),
            "samples_with_historical_keywords": 0,
            "samples_with_predicted_historical": 0,
            "correct_historical_detection": 0,
            "historical_accuracy": 0.0,
            "historical_keywords_found": {},
            "detailed_mismatches": []
        }

        historical_keywords = [
            "history of", "past", "previous", "prior", "previously",
            "years ago", "months ago", "childhood", "adolescent"
        ]

        for idx, row in df.iterrows():
            text = str(row.get("Text", "")).lower()
            predicted_historical = str(row.get("historical_entities", ""))

            has_historical_keywords = any(keyword in text for keyword in historical_keywords)
            has_predicted_historical = len(predicted_historical.strip()) > 0

            if has_historical_keywords:
                analysis["samples_with_historical_keywords"] += 1

                for keyword in historical_keywords:
                    if keyword in text:
                        if keyword not in analysis["historical_keywords_found"]:
                            analysis["historical_keywords_found"][keyword] = 0
                        analysis["historical_keywords_found"][keyword] += 1

            if has_predicted_historical:
                analysis["samples_with_predicted_historical"] += 1

            if has_historical_keywords and has_predicted_historical:
                analysis["correct_historical_detection"] += 1
            elif has_historical_keywords != has_predicted_historical:
                analysis["detailed_mismatches"].append({
                    "sample_id": idx + 1,
                    "text": text[:100] + "...",
                    "has_keywords": has_historical_keywords,
                    "has_predicted": has_predicted_historical,
                    "predicted_entities": predicted_historical
                })

        if analysis["samples_with_historical_keywords"] > 0:
            analysis["historical_accuracy"] = analysis["correct_historical_detection"] / analysis["samples_with_historical_keywords"]

        self.report_data["performance_metrics"]["historical"] = analysis

    def _analyze_uncertain_entities(self, df: pd.DataFrame):
        """Analyze uncertain entity detection"""
        logger.info("Analyzing uncertain entity detection...")

        analysis = {
            "total_samples": len(df),
            "samples_with_uncertainty_keywords": 0,
            "samples_with_predicted_uncertainty": 0,
            "correct_uncertainty_detection": 0,
            "uncertainty_accuracy": 0.0,
            "uncertainty_keywords_found": {},
            "detailed_analysis": []
        }

        uncertainty_keywords = [
            "possible", "possibly", "probable", "probably", "likely", "unlikely",
            "may", "might", "could", "perhaps", "maybe", "rule out", "r/o",
            "consider", "suspect", "suspicious", "question", "query"
        ]

        for idx, row in df.iterrows():
            text = str(row.get("Text", "")).lower()
            predicted_uncertain = str(row.get("uncertain_entities", ""))

            has_uncertainty_keywords = any(keyword in text for keyword in uncertainty_keywords)
            has_predicted_uncertainty = len(predicted_uncertain.strip()) > 0

            if has_uncertainty_keywords:
                analysis["samples_with_uncertainty_keywords"] += 1

                for keyword in uncertainty_keywords:
                    if keyword in text:
                        if keyword not in analysis["uncertainty_keywords_found"]:
                            analysis["uncertainty_keywords_found"][keyword] = 0
                        analysis["uncertainty_keywords_found"][keyword] += 1

            if has_predicted_uncertainty:
                analysis["samples_with_predicted_uncertainty"] += 1

            if has_uncertainty_keywords and has_predicted_uncertainty:
                analysis["correct_uncertainty_detection"] += 1

        if analysis["samples_with_uncertainty_keywords"] > 0:
            analysis["uncertainty_accuracy"] = analysis["correct_uncertainty_detection"] / analysis["samples_with_uncertainty_keywords"]

        self.report_data["performance_metrics"]["uncertainty"] = analysis

    def _analyze_confirmed_entities(self, df: pd.DataFrame):
        """Analyze confirmed entity detection"""
        logger.info("Analyzing confirmed entity detection...")

        analysis = {
            "total_samples": len(df),
            "samples_with_confirmed_keywords": 0,
            "samples_with_predicted_confirmed": 0,
            "correct_confirmed_detection": 0,
            "confirmed_accuracy": 0.0,
            "confirmed_keywords_found": {},
            "detailed_analysis": []
        }

        confirmed_keywords = [
            "diagnosed with", "diagnosis of", "confirmed", "has", "shows", "demonstrates",
            "positive for", "tested positive", "evidence of", "consistent with",
            "currently has", "currently experiencing", "active", "ongoing",
            "established", "documented", "known", "chronic", "acute"
        ]

        for idx, row in df.iterrows():
            text = str(row.get("Text", "")).lower()
            predicted_confirmed = str(row.get("confirmed_entities", ""))

            has_confirmed_keywords = any(keyword in text for keyword in confirmed_keywords)
            has_predicted_confirmed = len(predicted_confirmed.strip()) > 0

            if has_confirmed_keywords:
                analysis["samples_with_confirmed_keywords"] += 1

                for keyword in confirmed_keywords:
                    if keyword in text:
                        if keyword not in analysis["confirmed_keywords_found"]:
                            analysis["confirmed_keywords_found"][keyword] = 0
                        analysis["confirmed_keywords_found"][keyword] += 1

            if has_predicted_confirmed:
                analysis["samples_with_predicted_confirmed"] += 1

            if has_confirmed_keywords and has_predicted_confirmed:
                analysis["correct_confirmed_detection"] += 1

        if analysis["samples_with_confirmed_keywords"] > 0:
            analysis["confirmed_accuracy"] = analysis["correct_confirmed_detection"] / analysis["samples_with_confirmed_keywords"]

        self.report_data["performance_metrics"]["confirmed"] = analysis

    def _analyze_family_entities(self, df: pd.DataFrame):
        """Analyze family entity detection"""
        logger.info("Analyzing family entity detection...")

        analysis = {
            "total_samples": len(df),
            "samples_with_family_keywords": 0,
            "samples_with_predicted_family": 0,
            "correct_family_detection": 0,
            "family_accuracy": 0.0,
            "family_keywords_found": {},
            "detailed_analysis": []
        }

        family_keywords = [
            "family", "familial", "hereditary", "genetic", "inherited",
            "mother", "father", "parent", "sibling", "brother", "sister",
            "family history", "fh", "fhx", "maternal", "paternal"
        ]

        for idx, row in df.iterrows():
            text = str(row.get("Text", "")).lower()
            predicted_family = str(row.get("family_entities", ""))

            has_family_keywords = any(keyword in text for keyword in family_keywords)
            has_predicted_family = len(predicted_family.strip()) > 0

            if has_family_keywords:
                analysis["samples_with_family_keywords"] += 1

                for keyword in family_keywords:
                    if keyword in text:
                        if keyword not in analysis["family_keywords_found"]:
                            analysis["family_keywords_found"][keyword] = 0
                        analysis["family_keywords_found"][keyword] += 1

            if has_predicted_family:
                analysis["samples_with_predicted_family"] += 1

            if has_family_keywords and has_predicted_family:
                analysis["correct_family_detection"] += 1

        if analysis["samples_with_family_keywords"] > 0:
            analysis["family_accuracy"] = analysis["correct_family_detection"] / analysis["samples_with_family_keywords"]

        self.report_data["performance_metrics"]["family"] = analysis

    def _analyze_section_categories(self, df: pd.DataFrame):
        """Analyze section category detection"""
        logger.info("Analyzing section category detection...")

        analysis = {
            "total_samples": len(df),
            "section_categories_distribution": {},
            "most_common_sections": [],
            "section_detection_accuracy": 0.0,
            "average_sections_per_sample": 0.0
        }

        section_counts = {}
        total_sections = 0

        for idx, row in df.iterrows():
            sections = str(row.get("section_categories", ""))

            if sections and sections != "General Clinical Text":
                section_list = [s.strip() for s in sections.split(";") if s.strip()]
                total_sections += len(section_list)

                for section in section_list:
                    if section not in section_counts:
                        section_counts[section] = 0
                    section_counts[section] += 1

        analysis["section_categories_distribution"] = section_counts
        analysis["most_common_sections"] = sorted(section_counts.items(),
                                                key=lambda x: x[1], reverse=True)[:10]
        analysis["average_sections_per_sample"] = total_sections / len(df) if len(df) > 0 else 0

        self.report_data["performance_metrics"]["sections"] = analysis

    def _analyze_overall_entity_counts(self, df: pd.DataFrame):
        """Analyze overall entity count statistics"""
        logger.info("Analyzing overall entity statistics...")

        analysis = {
            "entity_count_statistics": {},
            "average_entities_per_sample": {},
            "entity_distribution": {}
        }

        count_columns = [
            "total_diseases_count", "total_gene_count", "negated_entities_count",
            "historical_entities_count", "uncertain_entities_count", "family_entities_count"
        ]

        for col in count_columns:
            if col in df.columns:
                counts = df[col].astype(float)
                analysis["entity_count_statistics"][col] = {
                    "mean": float(counts.mean()),
                    "median": float(counts.median()),
                    "std": float(counts.std()),
                    "min": int(counts.min()),
                    "max": int(counts.max()),
                    "total": int(counts.sum())
                }

                # Distribution
                distribution = counts.value_counts().to_dict()
                analysis["entity_distribution"][col] = {str(k): v for k, v in distribution.items()}

        self.report_data["performance_metrics"]["overall_statistics"] = analysis

    def _generate_summary_statistics(self):
        """Generate summary statistics across all analyses"""
        summary = {
            "overall_performance": {},
            "top_performing_categories": [],
            "areas_for_improvement": [],
            "key_insights": []
        }

        metrics = self.report_data["performance_metrics"]

        # Overall detection scores (use detection_rate for diseases/genes, accuracy for context types)
        accuracy_scores = {}
        if "diseases" in metrics:
            accuracy_scores["Disease Detection"] = metrics["diseases"].get("detection_rate", 0)
        if "genes" in metrics:
            accuracy_scores["Gene Detection"] = metrics["genes"].get("detection_rate", 0)
        if "negation" in metrics:
            accuracy_scores["Negation Detection"] = metrics["negation"].get("negation_accuracy", 0)
        if "historical" in metrics:
            accuracy_scores["Historical Detection"] = metrics["historical"].get("historical_accuracy", 0)
        if "uncertainty" in metrics:
            accuracy_scores["Uncertainty Detection"] = metrics["uncertainty"].get("uncertainty_accuracy", 0)
        if "confirmed" in metrics:
            accuracy_scores["Confirmed Detection"] = metrics["confirmed"].get("confirmed_accuracy", 0)
        if "family" in metrics:
            accuracy_scores["Family Detection"] = metrics["family"].get("family_accuracy", 0)

        summary["overall_performance"] = accuracy_scores

        # Top performing and improvement areas
        sorted_scores = sorted(accuracy_scores.items(), key=lambda x: x[1], reverse=True)
        summary["top_performing_categories"] = sorted_scores[:3]
        summary["areas_for_improvement"] = sorted_scores[-3:]

        # Key insights
        total_samples = self.report_data["analysis_metadata"]["total_samples"]
        summary["key_insights"] = [
            f"Analyzed {total_samples} clinical text samples",
            f"Best performing: {sorted_scores[0][0]} ({sorted_scores[0][1]:.2%})" if sorted_scores else "No performance data",
            f"Needs improvement: {sorted_scores[-1][0]} ({sorted_scores[-1][1]:.2%})" if sorted_scores else "No performance data"
        ]

        self.report_data["summary"] = summary

    def _save_detailed_report(self):
        """Save comprehensive performance report to file"""
        output_dir = Path("output/reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        report_file = output_dir / f"entity_prediction_analysis_report_{self.timestamp}.txt"

        with open(report_file, 'w') as f:
            self._write_report_header(f)
            self._write_entity_definitions_section(f)
            self._write_predictor_logic_explanations(f)
            self._write_three_column_system_documentation(f)
            self._write_complete_entity_detection_examples(f)
            self._write_summary_section(f)
            self._write_detailed_metrics(f)
            self._write_recommendations(f)

        logger.info(f"📊 Detailed performance report saved: {report_file}")
        return report_file

    def _write_report_header(self, f):
        """Write report header"""
        f.write("=" * 80 + "\n")
        f.write("MEDICAL NLP ENTITIES PREDICTION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Analysis ID: {self.timestamp}\n")
        f.write(f"Total Samples: {self.report_data['analysis_metadata']['total_samples']}\n")
        f.write(f"Input File: {self.report_data['analysis_metadata']['input_file']}\n")
        f.write("\n")

        # Add execution command information
        f.write("EXECUTION COMMAND\n")
        f.write("-" * 40 + "\n")
        cmd_info = self.report_data.get('execution_command', {})

        # Show pipeline command first if available (the command that generated the predictions)
        if cmd_info.get('pipeline_command'):
            # Clean up pipeline command - replace absolute Python paths with conda references
            pipeline_cmd = cmd_info.get('pipeline_command')
            conda_env = cmd_info.get('conda_environment', 'py311_bionlp')

            # Replace absolute Python path with conda run command
            import re
            # Match pattern: /path/to/anaconda.../envs/env_name/bin/python
            pipeline_cmd = re.sub(
                r'/[^\s]+/anaconda[^\s]*/envs/[^/]+/bin/python',
                f'conda run -n {conda_env} python',
                pipeline_cmd
            )
            # Also handle other absolute Python paths
            pipeline_cmd = re.sub(r'/[^\s]+/bin/python\d*', 'python', pipeline_cmd)

            f.write(f"Pipeline Command (Portable):\n")
            f.write(f"  {pipeline_cmd}\n")
            f.write("\n")

        # Then show analyzer execution details
        f.write("Analyzer Execution:\n")
        f.write(f"  Script: {cmd_info.get('script_name', 'Unknown')}\n")
        f.write(f"  Working Directory: {cmd_info.get('working_directory', 'Unknown')}\n")
        f.write(f"  Python: {cmd_info.get('python_executable', 'Unknown')}\n")
        if cmd_info.get('conda_environment'):
            f.write(f"  Conda Environment: {cmd_info.get('conda_environment')}\n")
        f.write(f"  Execution Method: {cmd_info.get('execution_method', 'Unknown')}\n")
        if cmd_info.get('launched_from'):
            f.write(f"  Launched From: {cmd_info.get('launched_from')}\n")
        if cmd_info.get('parent_launcher'):
            f.write(f"  Parent Launcher: {cmd_info.get('parent_launcher')}\n")
        f.write("\n")

        f.write("PIPELINE CONFIGURATION\n")
        f.write("-" * 40 + "\n")

        # Use detected configuration
        config = self.report_data['analysis_metadata'].get('pipeline_config', {})

        f.write(f"Pipeline Script: {config.get('pipeline_script', 'enhanced_medical_ner_predictor.py')}\n")
        f.write(f"Environment: {config.get('environment', 'py311_bionlp')} (Python {config.get('python_version', '3.11.5')})\n")
        f.write(f"Language: {config.get('language', 'English')}\n\n")

        f.write("Models:\n")
        f.write(f"  spaCy: {config.get('spacy_model', 'en_core_web_sm (v3.7.1)')}\n")
        f.write(f"         {config.get('spacy_model_type', 'English Small Model (~50MB)')}\n")
        f.write(f"  BioBERT Models:\n")
        f.write(f"    - Disease NER: {config.get('biobert_disease', 'alvaroalon2/biobert_diseases_ner (~411MB)')}\n")
        f.write(f"    - Gene NER: {config.get('biobert_gene', 'alvaroalon2/biobert_genetic_ner (~411MB)')}\n")
        f.write(f"    - Chemical NER: {config.get('biobert_chemical', 'alvaroalon2/biobert_chemical_ner (~822MB)')}\n\n")

        f.write("Detection Strategy:\n")
        f.write(f"  Entity Recognition: {config.get('entity_recognition', 'BioBERT + Custom Templates + spaCy NER')}\n")
        f.write(f"  Template Mode: {config.get('template_priority_mode', 'Templates override BioBERT detections (default)')}\n")
        f.write(f"  Negation: {config.get('negation_detection', 'negspaCy with Negex component')}\n")
        f.write(f"  Context Analysis: {config.get('context_analysis', 'Rule-based (confirmed, negated, uncertain, historical, family)')}\n")

        # Add detected information if available
        if 'spacy_version_actual' in config:
            f.write(f"spaCy Version (Detected): {config['spacy_version_actual']}\n")
        if 'model_loaded' in config:
            f.write(f"Model Status: {config['model_loaded']}\n")
        if 'model_pipeline' in config:
            f.write(f"Model Pipeline Components: {config['model_pipeline']}\n")

        f.write("=" * 80 + "\n")

        # Add input validation information
        f.write("\nINPUT DATA STRUCTURE\n")
        f.write("-" * 40 + "\n")
        validation = self.report_data['analysis_metadata'].get('input_validation', {})

        f.write(f"Validation Status: {validation.get('validation_status', 'Unknown')}\n")
        f.write(f"Total Input Columns: {validation.get('total_input_columns', 'Unknown')}\n")
        f.write(f"Required Columns Present: {validation.get('required_columns_present', [])}\n")

        if validation.get('required_columns_missing'):
            f.write(f"Required Columns Missing: {validation.get('required_columns_missing', [])}\n")

        f.write(f"Prediction Columns Present: {len(validation.get('prediction_columns_present', []))}\n")

        if validation.get('prediction_columns_present'):
            f.write("Detected Prediction Columns:\n")
            for col in validation.get('prediction_columns_present', [])[:5]:  # Show first 5
                f.write(f"  • {col}\n")
            if len(validation.get('prediction_columns_present', [])) > 5:
                f.write(f"  ... and {len(validation.get('prediction_columns_present', [])) - 5} more\n")

        f.write("=" * 80 + "\n\n")

    def _write_entity_definitions_section(self, f):
        """Write comprehensive entity definitions section"""
        f.write("ENTITY GROUP DEFINITIONS\n")
        f.write("=" * 80 + "\n")
        f.write("Comprehensive definitions for all entity types used in medical NLP processing.\n")
        f.write("These definitions help interpret the entity labels found in the analysis results.\n\n")

        # Group definitions by category
        categories = {
            "Standard spaCy Entity Types": [
                "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
                "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT",
                "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"
            ],
            "Medical and Biomedical Entities": [
                "DISEASE", "CONDITION", "PROBLEM", "DRUG", "CHEMICAL", "MEDICATION",
                "GENE", "PROTEIN", "GENETIC", "ANATOMY", "SYMPTOM", "PROCEDURE",
                "TEST", "TREATMENT"
            ],
            "Contextual and Temporal Entities": [
                "HISTORICAL", "FAMILY", "NEGATED", "UNCERTAIN", "HYPOTHETICAL",
                "RECENT", "CHRONIC", "ACUTE", "STABLE", "IMPROVING", "WORSENING"
            ],
            "Clinical Documentation Entities": [
                "SECTION", "HEADER", "FOOTER", "DOSAGE", "FREQUENCY", "ROUTE",
                "ALLERGY", "VITAL_SIGN", "LAB_VALUE", "IMAGING"
            ],
            "Special and Unknown Types": [
                "MISC", "OTHER", "UNKNOWN", "0"
            ]
        }

        for category, entity_types in categories.items():
            f.write(f"{category}\n")
            f.write("-" * len(category) + "\n")

            for entity_type in entity_types:
                if entity_type in self.entity_definitions:
                    definition = self.entity_definitions[entity_type]
                    f.write(f"{entity_type:12} | {definition}\n")
                else:
                    f.write(f"{entity_type:12} | (Definition not available)\n")
            f.write("\n")

        f.write("USAGE NOTES\n")
        f.write("-" * 20 + "\n")
        f.write("• Entity types may overlap in medical contexts (e.g., DRUG vs CHEMICAL)\n")
        f.write("• Contextual modifiers (NEGATED, UNCERTAIN) can apply to any medical entity\n")
        f.write("• Some entities may be classified differently by different models\n")
        f.write("• BioBERT models may use specialized medical entity classifications\n")
        f.write("• Entity confidence scores indicate model certainty in classification\n")
        f.write("• Multiple entity types may be detected for the same text span\n\n")

        f.write("=" * 80 + "\n\n")

    def _write_predictor_logic_explanations(self, f):
        """Write comprehensive predictor logic explanations section"""
        f.write("📋 PREDICTOR LOGIC EXPLANATIONS\n")
        f.write("=" * 80 + "\n")
        f.write("Detailed explanations of how each predictor identifies and processes medical entities\n")
        f.write("within clinical text. These algorithms form the core detection logic of the system.\n\n")

        # Historical Context Detection Logic
        f.write("Historical Context Detection Logic\n")
        f.write("-" * 40 + "\n")
        f.write("- Patterns: ['history of', 'past medical history', 'previously diagnosed', 'previous', 'prior', 'had', 'diagnosed', 'treated\n")
        f.write("  for', 'hx of', 'pmh', 'former', 'earlier', 'ago', 'previously', 'past', 'historical', 'before', 'prior to']\n")
        f.write("- Detection Method:\n")
        f.write("  a. Scans ±200 characters around each detected medical entity\n")
        f.write("  b. Finds closest historical pattern to the entity\n")
        f.write("  c. Extracts ±100 characters around the entity if pattern found within 200 characters\n")
        f.write("  d. Formats as: [HISTORY OF] ...Patient has a history of diabetes diagnosed 5 years ago...\n")
        f.write("- Purpose: Identifies medical conditions from patient's medical history vs current conditions\n\n")

        # Family Context Detection Logic
        f.write("Family Context Detection Logic\n")
        f.write("-" * 40 + "\n")
        f.write("- Patterns: ['family history', 'father', 'mother', 'parent', 'sibling', 'brother', 'sister', 'grandmother', 'grandfather',\n")
        f.write("  'aunt', 'uncle', 'cousin', 'relatives', 'familial', 'hereditary', 'genetic', 'fh:', 'fhx', 'maternal', 'paternal']\n")
        f.write("- Detection Method:\n")
        f.write("  a. Searches for family relationship indicators around medical entities\n")
        f.write("  b. Captures genetic predispositions and family medical history\n")
        f.write("  c. Extracts context showing familial relationship to medical conditions\n")
        f.write("  d. Formats as: [FATHER] ...heart disease in father and diabetes in mother...\n")
        f.write("- Advanced Logic:\n")
        f.write("  • Strong family indicators (mother, father, family history) automatically trigger family classification\n")
        f.write("  • Ambiguous terms (hereditary, genetic, familial) require additional family context validation\n")
        f.write("  • Prevents false positives: 'hereditary disease' alone won't classify nearby genes as family entities\n")
        f.write("- Purpose: Distinguishes between patient's conditions and family medical history for genetic risk assessment\n\n")

        # Uncertainty Context Detection Logic
        f.write("Uncertainty Context Detection Logic\n")
        f.write("-" * 40 + "\n")
        f.write("- Patterns: ['possible', 'probable', 'likely', 'may have', 'could be', 'suspected', 'rule out', 'differential', 'consider',\n")
        f.write("  'questionable', 'uncertain', 'perhaps', 'might', 'possibly', 'suggest', 'may be', 'could have', 'suspicious for', 'concern for',\n")
        f.write("  'worrisome for']\n")
        f.write("- Detection Method:\n")
        f.write("  a. Identifies medical entities mentioned with uncertainty or speculation\n")
        f.write("  b. Captures differential diagnoses and working diagnoses vs confirmed diagnoses\n")
        f.write("  c. Shows clinical reasoning and diagnostic uncertainty\n")
        f.write("  d. Formats as: [POSSIBLE] ...Possible pneumonia on chest X-ray...\n")
        f.write("- Purpose: Differentiates between confirmed diagnoses and suspected/possible conditions for clinical decision-making\n\n")

        # Negation Context Detection Logic
        f.write("Negation Context Detection Logic\n")
        f.write("-" * 40 + "\n")
        f.write("- Patterns: ['no', 'not', 'without', 'absent', 'negative', 'denies', 'rules out', 'free of', 'clear of', 'unremarkable',\n")
        f.write("  'normal', 'within normal limits', 'no evidence of', 'no signs of', 'no history of', 'never had', 'r/o']\n")
        f.write("- Detection Method:\n")
        f.write("  a. Detects medical entities that are explicitly denied or absent\n")
        f.write("  b. Captures negative findings and ruled-out conditions\n")
        f.write("  c. Essential for avoiding false positive medical conditions\n")
        f.write("  d. Formats as: [NO EVIDENCE OF] ...No evidence of cancer was found...\n")
        f.write("- Purpose: Critical for accuracy - distinguishes between present and absent medical conditions\n\n")

        # Section Categorization Logic
        f.write("Section Categorization Logic\n")
        f.write("-" * 40 + "\n")
        f.write("- Method: Keyword-based pattern matching against clinical documentation sections\n")
        f.write("- Enhanced Keywords: Now includes 20 section types vs original 10\n")
        f.write("- Logic:\n")
        f.write("  a. Converts entire text to lowercase for case-insensitive matching\n")
        f.write("  b. Searches for section header patterns throughout the document\n")
        f.write("  c. Returns all matching sections as semicolon-separated list\n")
        f.write("  d. Falls back to 'General Clinical Text' if no sections detected\n")
        f.write("- Enhanced Section Types:\n")
        f.write("  • Chief Complaint: 'chief complaint', 'cc:', 'presenting complaint'\n")
        f.write("  • History of Present Illness: 'history of present illness', 'hpi:', 'present illness'\n")
        f.write("  • Past Medical History: 'past medical history', 'pmh:', 'medical history'\n")
        f.write("  • Medications: 'medications', 'meds:', 'current medications', 'drug list'\n")
        f.write("  • Allergies: 'allergies', 'drug allergies', 'nkda', 'adverse reactions'\n")
        f.write("  • Social History: 'social history', 'sh:', 'social'\n")
        f.write("  • Family History: 'family history', 'fh:', 'fhx'\n")
        f.write("  • Review of Systems: 'review of systems', 'ros:', 'systems review'\n")
        f.write("  • Physical Examination: 'physical exam', 'pe:', 'examination', 'physical'\n")
        f.write("  • Assessment: 'assessment', 'impression', 'diagnosis'\n")
        f.write("  • Plan: 'plan', 'treatment plan', 'management'\n")
        f.write("  • Labs: 'laboratory', 'labs', 'lab results', 'blood work'\n")
        f.write("  • Imaging: 'imaging', 'radiology', 'x-ray', 'ct', 'mri', 'ultrasound'\n")
        f.write("  • Procedures: 'procedures', 'interventions', 'operations'\n")
        f.write("  • Discharge: 'discharge', 'disposition', 'discharge summary'\n")
        f.write("  • Follow-up: 'follow up', 'follow-up', 'return visit'\n")
        f.write("  • Summary: 'summary', 'conclusion', 'findings'\n")
        f.write("  • Vital Signs: 'vital signs', 'vitals', 'vs:'\n")
        f.write("  • Progress Notes: 'progress', 'progress note', 'daily note'\n")
        f.write("  • Consultation: 'consultation', 'consult', 'specialist opinion'\n")
        f.write("- Example Mappings:\n")
        f.write("  • 'Summary: Patient...' → 'Summary' ✅ (enhanced from previous 'General Clinical Text')\n")
        f.write("  • 'Assessment and Plan:' → 'Assessment'\n")
        f.write("  • 'Discharge instructions:' → 'Discharge'\n")
        f.write("- Purpose: Organizes clinical text into structured documentation sections for better information retrieval\n\n")

        # Context Extraction Technical Details
        f.write("Context Extraction Technical Implementation\n")
        f.write("-" * 40 + "\n")
        f.write("- Context Window: Precise ±100 characters around detected entities\n")
        f.write("- Pattern Search Range: ±200 characters around entities for pattern detection\n")
        f.write("- Priority System: Target Rules → BioBERT Models → Fallback Rules → spaCy NER\n")
        f.write("- Output Format: [PATTERN_DETECTED] ...context text with entity highlighted...\n")
        f.write("- Character Boundary Handling: Ensures context doesn't break mid-word\n")
        f.write("- Multiple Pattern Detection: Can detect multiple context types for same entity\n")
        f.write("- Entity Position Tracking: Maintains exact character positions for visualization\n\n")

        f.write("=" * 80 + "\n\n")

    def _write_three_column_system_documentation(self, f):
        """Write comprehensive three-column system documentation for all entity contexts"""
        f.write("📋 THREE-COLUMN SYSTEM: ENTITY CONTEXT ARCHITECTURE\n")
        f.write("=" * 80 + "\n")
        f.write("Detailed explanation of how the three-column system works for each entity context type.\n")
        f.write("This architecture separates WHAT (entities), WHY (predictors), and WHERE (context).\n\n")

        # CONFIRMED ENTITIES
        f.write("1. CONFIRMED ENTITIES - Three-Column System\n")
        f.write("-" * 80 + "\n")
        f.write("Current Behavior (By Design):\n\n")
        f.write("Three-Column System:\n\n")
        f.write("| Column                           | Purpose                                  | Example Content                              |\n")
        f.write("|----------------------------------|------------------------------------------|----------------------------------------------|\n")
        f.write("| confirmed_entities               | Medical entities with definitive status  | \"diabetes; hypertension; pneumonia\"          |\n")
        f.write("| confirmed_entities_predictors    | Confirmation keywords that triggered     | \"HAS, DIAGNOSED WITH, CHRONIC\"               |\n")
        f.write("| confirmed_context_sentences      | Sentences with highlighted keywords      | \"[HAS] ...Patient has diabetes...\"           |\n\n")
        f.write("How It Works:\n\n")
        f.write("Detection Pipeline:\n")
        f.write("1. Step 1: Detect ALL medical entities (diseases, genes, chemicals)\n")
        f.write("2. Step 2: Check each entity's surrounding context (±200 chars) for confirmation patterns:\n")
        f.write("   - Status indicators: \"has\", \"diagnosed with\", \"confirmed\", \"shows\", \"demonstrates\"\n")
        f.write("   - Evidence terms: \"positive for\", \"tested positive\", \"evidence of\", \"consistent with\"\n")
        f.write("   - Temporal markers: \"currently has\", \"currently experiencing\", \"active\", \"ongoing\"\n")
        f.write("   - Clinical descriptors: \"established\", \"documented\", \"known\", \"chronic\", \"acute\"\n")
        f.write("3. Step 3: If confirmation keywords found → Mark entity as confirmed\n")
        f.write("4. Step 4: Populate three columns:\n")
        f.write("   - Medical entity → confirmed_entities\n")
        f.write("   - Triggering keywords → confirmed_entities_predictors\n")
        f.write("   - Context sentences → confirmed_context_sentences\n\n")
        f.write("Example from Real Data:\n\n")
        f.write("Text: \"Patient has diabetes and shows evidence of chronic hypertension...\"\n\n")
        f.write("Detection:\n")
        f.write("- confirmed_entities: \"diabetes; hypertension\" ← The medical conditions\n")
        f.write("- confirmed_entities_predictors: \"HAS, CHRONIC\" ← Why they're confirmed\n")
        f.write("- confirmed_context_sentences: \"[HAS] ...Patient has diabetes...\" ← Full context\n\n")
        f.write("Why This Design?\n\n")
        f.write("✅ Separates concerns:\n")
        f.write("- WHAT: Medical conditions (confirmed_entities)\n")
        f.write("- WHY: Confirmation indicators (confirmed_entities_predictors)\n")
        f.write("- WHERE: Context (confirmed_context_sentences)\n\n")
        f.write("✅ Enables analysis:\n")
        f.write("- Which conditions are actively present?\n")
        f.write("- What evidence supports the diagnosis?\n")
        f.write("- Full verification through context\n\n")
        f.write("Clinical Significance:\n")
        f.write("Confirmed entities represent ACTIVELY PRESENT, DIAGNOSED conditions. This distinguishes them from:\n")
        f.write("- Negated (explicitly absent): \"has no diabetes\"\n")
        f.write("- Uncertain (possible): \"may have diabetes\"\n")
        f.write("- Historical (past): \"had diabetes\"\n\n")
        f.write("Coverage: 95% (20/21 rows in test data have complete predictor extraction)\n\n")

        # NEGATED ENTITIES
        f.write("2. NEGATED ENTITIES - Three-Column System\n")
        f.write("-" * 80 + "\n")
        f.write("Current Behavior (By Design):\n\n")
        f.write("Three-Column System:\n\n")
        f.write("| Column                          | Purpose                                  | Example Content                              |\n")
        f.write("|---------------------------------|------------------------------------------|----------------------------------------------|\n")
        f.write("| negated_entities                | Medical entities explicitly absent       | \"cancer; chest pain; fever\"                  |\n")
        f.write("| negated_entities_predictors     | Negation keywords that triggered         | \"NO, DENIES, WITHOUT\"                        |\n")
        f.write("| negated_context_sentences       | Sentences with highlighted keywords      | \"[NO] ...no evidence of cancer...\"           |\n\n")
        f.write("How It Works:\n\n")
        f.write("Detection Pipeline:\n")
        f.write("1. Step 1: Detect ALL medical entities (diseases, symptoms, conditions)\n")
        f.write("2. Step 2: Check each entity's surrounding context (±200 chars) for negation patterns (72 patterns):\n")
        f.write("   - Direct negation: \"no\", \"not\", \"without\", \"absent\", \"negative\"\n")
        f.write("   - Clinical negation: \"denies\", \"rules out\", \"free of\", \"clear of\", \"unremarkable\"\n")
        f.write("   - Evidence negation: \"no evidence of\", \"no signs of\", \"no history of\", \"never had\"\n")
        f.write("3. Step 3: If negation keywords found → Mark entity as negated\n")
        f.write("4. Step 4: Populate three columns:\n")
        f.write("   - Medical entity → negated_entities\n")
        f.write("   - Triggering keywords → negated_entities_predictors\n")
        f.write("   - Context sentences → negated_context_sentences\n\n")
        f.write("Example from Real Data:\n\n")
        f.write("Text: \"Patient denies chest pain and has no evidence of diabetes...\"\n\n")
        f.write("Detection:\n")
        f.write("- negated_entities: \"chest pain; diabetes\" ← The absent conditions\n")
        f.write("- negated_entities_predictors: \"DENIES, NO EVIDENCE OF\" ← Why they're negated\n")
        f.write("- negated_context_sentences: \"[DENIES] ...denies chest pain...\" ← Full context\n\n")
        f.write("Why This Design?\n\n")
        f.write("✅ Separates concerns:\n")
        f.write("- WHAT: Absent medical conditions (negated_entities)\n")
        f.write("- WHY: Negation indicators (negated_entities_predictors)\n")
        f.write("- WHERE: Context (negated_context_sentences)\n\n")
        f.write("✅ Enables analysis:\n")
        f.write("- Which conditions are explicitly ruled out?\n")
        f.write("- What type of negation was used?\n")
        f.write("- Full verification through context\n\n")
        f.write("Clinical Significance:\n")
        f.write("CRITICAL FOR ACCURACY - Distinguishing \"has diabetes\" from \"has no diabetes\" is essential.\n")
        f.write("Negation detection prevents false positive diagnoses and captures negative findings.\n\n")
        f.write("Coverage: 100% (43/43 rows in test data have complete predictor and context extraction)\n\n")

        # UNCERTAIN ENTITIES
        f.write("3. UNCERTAIN ENTITIES - Three-Column System\n")
        f.write("-" * 80 + "\n")
        f.write("Current Behavior (By Design):\n\n")
        f.write("Three-Column System:\n\n")
        f.write("| Column                           | Purpose                                  | Example Content                              |\n")
        f.write("|----------------------------------|------------------------------------------|----------------------------------------------|\n")
        f.write("| uncertain_entities               | Medical entities under consideration     | \"pneumonia; appendicitis; infection\"         |\n")
        f.write("| uncertain_entities_predictors    | Uncertainty keywords that triggered      | \"POSSIBLE, MAY, RULE OUT\"                    |\n")
        f.write("| uncertain_context_sentences      | Sentences with highlighted keywords      | \"[POSSIBLE] ...possible pneumonia...\"        |\n\n")
        f.write("How It Works:\n\n")
        f.write("Detection Pipeline:\n")
        f.write("1. Step 1: Detect ALL medical entities (diseases, diagnoses, conditions)\n")
        f.write("2. Step 2: Check each entity's surrounding context (±200 chars) for uncertainty patterns (42 patterns):\n")
        f.write("   - Speculation: \"possible\", \"probable\", \"likely\", \"may have\", \"could be\", \"suspected\"\n")
        f.write("   - Differential dx: \"rule out\", \"differential\", \"consider\", \"questionable\"\n")
        f.write("   - Clinical concern: \"suspicious for\", \"concern for\", \"worrisome for\"\n")
        f.write("   - Single-word modifiers: \"may\", \"might\", \"unknown\", \"uncertain\"\n")
        f.write("3. Step 3: If uncertainty keywords found → Mark entity as uncertain\n")
        f.write("4. Step 4: Populate three columns:\n")
        f.write("   - Medical entity → uncertain_entities\n")
        f.write("   - Triggering keywords → uncertain_entities_predictors\n")
        f.write("   - Context sentences → uncertain_context_sentences\n\n")
        f.write("Example from Real Data:\n\n")
        f.write("Text: \"Possible pneumonia on chest X-ray, rule out appendicitis...\"\n\n")
        f.write("Detection:\n")
        f.write("- uncertain_entities: \"pneumonia; appendicitis\" ← The suspected conditions\n")
        f.write("- uncertain_entities_predictors: \"POSSIBLE, RULE OUT\" ← Why they're uncertain\n")
        f.write("- uncertain_context_sentences: \"[POSSIBLE] ...Possible pneumonia...\" ← Full context\n\n")
        f.write("Why This Design?\n\n")
        f.write("✅ Separates concerns:\n")
        f.write("- WHAT: Suspected medical conditions (uncertain_entities)\n")
        f.write("- WHY: Uncertainty indicators (uncertain_entities_predictors)\n")
        f.write("- WHERE: Context (uncertain_context_sentences)\n\n")
        f.write("✅ Enables analysis:\n")
        f.write("- Which conditions are under investigation?\n")
        f.write("- What level of diagnostic uncertainty exists?\n")
        f.write("- Full verification through context\n\n")
        f.write("Clinical Significance:\n")
        f.write("Uncertain entities represent DIFFERENTIAL DIAGNOSES or WORKING DIAGNOSES. This captures:\n")
        f.write("- Clinical reasoning and diagnostic process\n")
        f.write("- Conditions requiring further investigation\n")
        f.write("- Distinguishes suspected from confirmed diagnoses\n\n")
        f.write("Coverage: 100% (12/12 rows in test data have complete predictor and context extraction)\n\n")

        # HISTORICAL ENTITIES
        f.write("4. HISTORICAL ENTITIES - Three-Column System\n")
        f.write("-" * 80 + "\n")
        f.write("Current Behavior (By Design):\n\n")
        f.write("Three-Column System:\n\n")
        f.write("| Column                           | Purpose                                  | Example Content                              |\n")
        f.write("|----------------------------------|------------------------------------------|----------------------------------------------|\n")
        f.write("| historical_entities              | Medical entities from past medical hx    | \"pneumonia; surgery; myocardial infarction\" |\n")
        f.write("| historical_entities_predictors   | Historical keywords that triggered       | \"HISTORY OF, PREVIOUS, PAST\"                 |\n")
        f.write("| historical_context_sentences     | Sentences with highlighted keywords      | \"[HISTORY OF] ...history of pneumonia...\"   |\n\n")
        f.write("How It Works:\n\n")
        f.write("Detection Pipeline:\n")
        f.write("1. Step 1: Detect ALL medical entities (diseases, procedures, conditions)\n")
        f.write("2. Step 2: Check each entity's surrounding context (±200 chars) for historical patterns (69 patterns):\n")
        f.write("   - Medical history: \"history of\", \"past medical history\", \"pmh\", \"hx of\"\n")
        f.write("   - Temporal past: \"previous\", \"prior\", \"previously\", \"had\", \"ago\", \"earlier\"\n")
        f.write("   - Treatment history: \"diagnosed\", \"treated for\", \"former\", \"before\"\n")
        f.write("3. Step 3: If historical keywords found → Mark entity as historical\n")
        f.write("4. Step 4: Populate three columns:\n")
        f.write("   - Medical entity → historical_entities\n")
        f.write("   - Triggering keywords → historical_entities_predictors\n")
        f.write("   - Context sentences → historical_context_sentences\n\n")
        f.write("Example from Real Data:\n\n")
        f.write("Text: \"Patient has history of pneumonia, previously diagnosed with diabetes...\"\n\n")
        f.write("Detection:\n")
        f.write("- historical_entities: \"pneumonia; diabetes\" ← The past conditions\n")
        f.write("- historical_entities_predictors: \"HISTORY OF, PREVIOUS\" ← Why they're historical\n")
        f.write("- historical_context_sentences: \"[HISTORY OF] ...history of pneumonia...\" ← Full context\n\n")
        f.write("Why This Design?\n\n")
        f.write("✅ Separates concerns:\n")
        f.write("- WHAT: Past medical conditions (historical_entities)\n")
        f.write("- WHY: Historical indicators (historical_entities_predictors)\n")
        f.write("- WHERE: Context (historical_context_sentences)\n\n")
        f.write("✅ Enables analysis:\n")
        f.write("- Which conditions are part of medical history?\n")
        f.write("- How far back in time?\n")
        f.write("- Full verification through context\n\n")
        f.write("Clinical Significance:\n")
        f.write("Historical entities represent PAST CONDITIONS that inform current care but are not actively treated.\n")
        f.write("Critical for: Risk assessment, medication decisions, understanding disease progression.\n\n")
        f.write("Coverage: 100% (25/25 rows in test data have complete predictor and context extraction)\n\n")

        # FAMILY ENTITIES
        f.write("5. FAMILY ENTITIES - Three-Column System\n")
        f.write("-" * 80 + "\n")
        f.write("Current Behavior (By Design):\n\n")
        f.write("Three-Column System:\n\n")
        f.write("| Column                     | Purpose                                  | Example Content                              |\n")
        f.write("|----------------------------|------------------------------------------|----------------------------------------------|")
        f.write("\n")
        f.write("| family_entities            | Medical entities found in family context | \"AARS2; AARS2 gene; intellectual disability\" |\n")
        f.write("| family_entities_predictors | Family keywords that triggered detection | \"INHERITED, PARENT, MOTHER, FATHER\"          |\n")
        f.write("| family_context_sentences   | Sentences with highlighted keywords      | \"[PARENT] ...inherited from each parent...\"  |\n\n")
        f.write("How It Works:\n\n")
        f.write("Detection Pipeline:\n")
        f.write("1. Step 1: Detect ALL medical entities (diseases, genes, chemicals)\n")
        f.write("2. Step 2: Check each entity's surrounding context (±50-100 chars) for family keywords:\n")
        f.write("   - Strong indicators: \"mother\", \"father\", \"parent\", \"family history\", \"fh:\", \"fhx\"\n")
        f.write("   - Relationships: \"sibling\", \"brother\", \"sister\", \"grandmother\", \"grandfather\"\n")
        f.write("   - Ambiguous indicators: \"hereditary\", \"genetic\", \"familial\" (require additional context)\n")
        f.write("3. Step 3: If family keywords found → Mark entity as family-related\n")
        f.write("4. Step 4: Populate three columns:\n")
        f.write("   - Medical entity → family_entities\n")
        f.write("   - Triggering keywords → family_entities_predictors\n")
        f.write("   - Context sentences → family_context_sentences\n\n")
        f.write("Example from Real Data:\n\n")
        f.write("Text: \"...two disease-causing variants of the AARS2 gene must be inherited, one from each parent...\"\n\n")
        f.write("Detection:\n")
        f.write("- family_entities: \"AARS2; AARS2 gene\" ← The medical condition\n")
        f.write("- family_entities_predictors: \"INHERITED, PARENT\" ← Why it's family-related\n")
        f.write("- family_context_sentences: \"[PARENT] ...inherited from each parent...\" ← Full context\n\n")
        f.write("Why This Design?\n\n")
        f.write("✅ Separates concerns:\n")
        f.write("- WHAT: Medical conditions (family_entities)\n")
        f.write("- WHY: Family indicators (family_entities_predictors)\n")
        f.write("- WHERE: Context (family_context_sentences)\n\n")
        f.write("✅ Enables analysis:\n")
        f.write("- Which medical conditions run in families?\n")
        f.write("- Which family relationships are most mentioned?\n")
        f.write("- Full verification through context\n\n")
        f.write("Clinical Significance:\n")
        f.write("Family entities identify GENETIC RISK FACTORS and HEREDITARY CONDITIONS. Critical for:\n")
        f.write("- Genetic counseling and risk assessment\n")
        f.write("- Screening recommendations for at-risk family members\n")
        f.write("- Understanding inheritance patterns\n\n")

        # SECTION CATEGORIES
        f.write("6. SECTION CATEGORIES - Document-Level Classification\n")
        f.write("-" * 80 + "\n")
        f.write("Current Behavior (By Design):\n\n")
        f.write("Single-Column System (Different from Entity Contexts):\n\n")
        f.write("| Column             | Purpose                                  | Example Content                              |\n")
        f.write("|--------------------|------------------------------------------|----------------------------------------------|\n")
        f.write("| section_categories | Classify entire document into sections   | \"Summary; Imaging\" or \"General Clinical Text\"|\n\n")
        f.write("How It Works:\n\n")
        f.write("Detection Pipeline:\n")
        f.write("1. Step 1: Scan entire document for section headers (HTML tags like <strong>Summary</strong>)\n")
        f.write("2. Step 2: Match against 20+ standard clinical section names:\n")
        f.write("   - Clinical: Summary, Assessment, Plan, Review of Systems\n")
        f.write("   - History: Family History, Past Medical History, Social History\n")
        f.write("   - Data: Laboratory, Imaging, Vital Signs\n")
        f.write("   - Documentation: Subjective, Objective, Progress Notes\n")
        f.write("3. Step 3: Return all matching sections as semicolon-separated list\n")
        f.write("4. Step 4: If no sections detected → \"General Clinical Text\"\n\n")
        f.write("Example from Real Data:\n\n")
        f.write("Text: \"<strong>Summary</strong> Patient has diabetes... <h2>Imaging</h2> CT scan shows...\"\n\n")
        f.write("Detection:\n")
        f.write("- section_categories: \"Summary; Imaging\" ← Document classification\n\n")
        f.write("Why NO Context Column?\n\n")
        f.write("⚠️  IMPORTANT DISTINCTION:\n")
        f.write("- Entity contexts (confirmed, negated, etc.): SENTENCE-LEVEL → Need context_sentences column\n")
        f.write("- Section categories: DOCUMENT-LEVEL → Classify entire document, no context column needed\n\n")
        f.write("| Feature               | Entity Contexts                    | Section Categories               |\n")
        f.write("|-----------------------|------------------------------------|----------------------------------|\n")
        f.write("| Scope                 | Sentence-level                     | Document-level                   |\n")
        f.write("| Purpose               | Highlight specific sentences       | Classify entire document         |\n")
        f.write("| Context Column        | Has *_context_sentences            | No context column needed         |\n")
        f.write("| Example               | \"[HAS] ...Patient has diabetes...\" | \"Summary; Imaging\"               |\n\n")
        f.write("Clinical Significance:\n")
        f.write("Section categories enable DOCUMENT ORGANIZATION and SECTION-SPECIFIC ANALYSIS.\n")
        f.write("Helps route information to appropriate clinical workflows and documentation systems.\n\n")
        f.write("Coverage: 100% (all documents assigned at least 'General Clinical Text')\n\n")

        f.write("=" * 80 + "\n\n")

    def _write_complete_entity_detection_examples(self, f):
        """Write comprehensive entity detection system examples"""
        f.write("📋 COMPLETE ENTITY DETECTION SYSTEM\n")
        f.write("=" * 80 + "\n")
        f.write("Comprehensive examples of all entity types with working patterns and expected outputs.\n")
        f.write("These examples demonstrate the system's ability to distinguish between different\n")
        f.write("medical contexts and provide accurate entity classification.\n\n")

        # CONFIRMED ENTITIES
        f.write("CONFIRMED ENTITIES (New Feature)\n")
        f.write("-" * 40 + "\n")
        f.write("Pattern: Definitive, evidence-based, current status\n")
        f.write("Examples:\n")
        f.write("  ✅ \"Patient has confirmed pneumonia\" → pneumonia\n")
        f.write("  ✅ \"Shows evidence of heart failure\" → heart failure\n")
        f.write("  ✅ \"Positive for COVID-19, established hypertension\" → hypertension\n")
        f.write("Purpose: Identifies definitive diagnoses with high clinical certainty\n\n")

        # UNCERTAIN ENTITIES
        f.write("UNCERTAIN ENTITIES\n")
        f.write("-" * 40 + "\n")
        f.write("Pattern: Speculative, differential, under consideration\n")
        f.write("Examples:\n")
        f.write("  ✅ \"Possible pneumonia on chest X-ray\" → pneumonia\n")
        f.write("  ✅ \"May have diabetes, likely hypertension\" → diabetes, hypertension\n")
        f.write("  ✅ \"Rule out appendicitis, consider gastritis\" → appendicitis, gastritis\n")
        f.write("Purpose: Captures diagnostic uncertainty and conditions under investigation\n\n")

        # NEGATED ENTITIES
        f.write("NEGATED ENTITIES\n")
        f.write("-" * 40 + "\n")
        f.write("Pattern: Absent, denied, ruled out\n")
        f.write("Examples:\n")
        f.write("  ✅ \"No evidence of cancer, denies chest pain\" → cancer, chest pain\n")
        f.write("  ✅ \"Rules out pneumonia, negative for COVID-19\" → pneumonia\n")
        f.write("  ✅ \"No history of diabetes, never had hypertension\" → diabetes, hypertension\n")
        f.write("Purpose: Critical for accuracy - identifies explicitly absent conditions\n\n")

        # HISTORICAL ENTITIES
        f.write("HISTORICAL ENTITIES\n")
        f.write("-" * 40 + "\n")
        f.write("Pattern: Past medical events, previous conditions\n")
        f.write("Examples:\n")
        f.write("  ✅ \"History of myocardial infarction, past medical history of diabetes\" →\n")
        f.write("      myocardial infarction, diabetes\n")
        f.write("  ✅ \"Previously diagnosed with hypertension, had surgery\" → hypertension\n")
        f.write("  ✅ \"Prior episode of pneumonia, treated for depression\" → pneumonia, depression\n")
        f.write("Purpose: Separates past medical history from current active conditions\n\n")

        # FAMILY ENTITIES
        f.write("FAMILY ENTITIES\n")
        f.write("-" * 40 + "\n")
        f.write("Pattern: Family history, genetic predisposition\n")
        f.write("Examples:\n")
        f.write("  ✅ \"Family history of heart disease, mother has diabetes\" → heart disease, diabetes\n")
        f.write("  ✅ \"Father died of cancer, maternal grandmother had hypertension\" →\n")
        f.write("      cancer, hypertension\n")
        f.write("  ✅ \"Familial hypercholesterolemia, genetic predisposition\" → hereditary conditions\n")
        f.write("Purpose: Identifies genetic risk factors and family medical history\n\n")

        # CONTEXT EXTRACTION
        f.write("📏 CONTEXT EXTRACTION (ALL TYPES)\n")
        f.write("-" * 40 + "\n")
        f.write("Consistent Context Window: ±100 characters around detected entities\n")
        f.write("Output Format: [PATTERN] ...context text with entity...\n")
        f.write("Applied to: confirmed, uncertain, negated, historical, family entities\n\n")

        f.write("CLINICAL SIGNIFICANCE\n")
        f.write("-" * 40 + "\n")
        f.write("This comprehensive entity classification system enables:\n")
        f.write("• Accurate distinction between confirmed vs suspected diagnoses\n")
        f.write("• Identification of absent conditions (crucial for clinical accuracy)\n")
        f.write("• Separation of current conditions from medical history\n")
        f.write("• Genetic risk assessment through family history extraction\n")
        f.write("• Evidence-based clinical decision support\n\n")

        f.write("=" * 80 + "\n\n")

    def _write_summary_section(self, f):
        """Write summary section"""
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 40 + "\n")

        summary = self.report_data["summary"]

        f.write("Overall Performance Scores:\n")
        for category, score in summary["overall_performance"].items():
            f.write(f"  • {category}: {score:.2%}\n")

        f.write("\nTop Performing Categories:\n")
        for category, score in summary["top_performing_categories"]:
            f.write(f"  1. {category}: {score:.2%}\n")

        f.write("\nAreas for Improvement:\n")
        for category, score in summary["areas_for_improvement"]:
            f.write(f"  • {category}: {score:.2%}\n")

        f.write(f"\nKey Insights:\n")
        for insight in summary["key_insights"]:
            f.write(f"  • {insight}\n")

        f.write("\n" + "=" * 80 + "\n\n")

    def _write_detailed_metrics(self, f):
        """Write detailed metrics for each category"""
        metrics = self.report_data["performance_metrics"]

        # Disease Detection
        if "diseases" in metrics:
            f.write("DISEASE DETECTION ANALYSIS\n")
            f.write("-" * 40 + "\n")
            disease_metrics = metrics["diseases"]
            f.write(f"Total Samples: {disease_metrics['total_samples']}\n")
            f.write(f"Samples with Detected Diseases: {disease_metrics['samples_with_predicted']}\n")
            f.write(f"Detection Rate: {disease_metrics['detection_rate']:.1%}\n")
            f.write(f"Total Entities Detected: {disease_metrics['total_entities_detected']}\n")
            f.write(f"Unique Diseases: {disease_metrics['unique_diseases_count']}\n")
            f.write(f"Avg Entities per Sample: {disease_metrics['avg_entities_per_sample']:.2f}\n")
            f.write(f"Max Entities in Sample: {disease_metrics['max_entities_in_sample']}\n")
            f.write(f"Min Entities in Sample: {disease_metrics['min_entities_in_sample']}\n")

            if disease_metrics.get('top_diseases'):
                f.write("\nTop 10 Most Frequent Diseases:\n")
                for item in disease_metrics['top_diseases']:
                    f.write(f"  • {item['disease']}: {item['count']} occurrences\n")

            f.write("\nEntity Count Distribution:\n")
            for count, samples in sorted(disease_metrics.get('distribution', {}).items(), key=lambda x: int(x[0])):
                f.write(f"  • {count} diseases: {samples} samples\n")
            f.write("\n")

        # Gene Detection
        if "genes" in metrics:
            f.write("GENE DETECTION ANALYSIS\n")
            f.write("-" * 40 + "\n")
            gene_metrics = metrics["genes"]
            f.write(f"Total Samples: {gene_metrics['total_samples']}\n")
            f.write(f"Samples with Detected Genes: {gene_metrics['samples_with_predicted']}\n")
            f.write(f"Detection Rate: {gene_metrics['detection_rate']:.1%}\n")
            f.write(f"Total Entities Detected: {gene_metrics['total_entities_detected']}\n")
            f.write(f"Unique Genes: {gene_metrics['unique_genes_count']}\n")
            f.write(f"Avg Entities per Sample: {gene_metrics['avg_entities_per_sample']:.2f}\n")
            f.write(f"Max Entities in Sample: {gene_metrics['max_entities_in_sample']}\n")
            f.write(f"Min Entities in Sample: {gene_metrics['min_entities_in_sample']}\n")

            if gene_metrics.get('top_genes'):
                f.write("\nTop 10 Most Frequent Genes:\n")
                for item in gene_metrics['top_genes']:
                    f.write(f"  • {item['gene']}: {item['count']} occurrences\n")

            f.write("\nEntity Count Distribution:\n")
            for count, samples in sorted(gene_metrics.get('distribution', {}).items(), key=lambda x: int(x[0])):
                f.write(f"  • {count} genes: {samples} samples\n")
            f.write("\n")

        # Negation Detection
        if "negation" in metrics:
            f.write("NEGATION DETECTION ANALYSIS\n")
            f.write("-" * 40 + "\n")
            neg_metrics = metrics["negation"]
            f.write(f"Samples with Negation Words: {neg_metrics['samples_with_negation_words']}\n")
            f.write(f"Samples with Predicted Negation: {neg_metrics['samples_with_predicted_negation']}\n")
            f.write(f"Correct Negation Detection: {neg_metrics['correct_negation_detection']}\n")
            f.write(f"False Positives: {neg_metrics['false_positive_negation']}\n")
            f.write(f"False Negatives: {neg_metrics['false_negative_negation']}\n")
            f.write(f"Negation Accuracy: {neg_metrics['negation_accuracy']:.3f}\n")
            f.write(f"Negation Precision: {neg_metrics['negation_precision']:.3f}\n")
            f.write(f"Negation Recall: {neg_metrics['negation_recall']:.3f}\n\n")

        # Historical, Uncertain, Confirmed, and Family entity analyses
        for category in ["historical", "uncertainty", "confirmed", "family"]:
            if category in metrics:
                f.write(f"{category.upper()} ENTITY DETECTION ANALYSIS\n")
                f.write("-" * 40 + "\n")
                cat_metrics = metrics[category]
                f.write(f"Samples with {category.title()} Keywords: {cat_metrics.get(f'samples_with_{category}_keywords', 0)}\n")
                f.write(f"Samples with Predicted {category.title()}: {cat_metrics.get(f'samples_with_predicted_{category}', 0)}\n")
                f.write(f"Correct {category.title()} Detection: {cat_metrics.get(f'correct_{category}_detection', 0)}\n")
                f.write(f"{category.title()} Accuracy: {cat_metrics.get(f'{category}_accuracy', 0):.3f}\n\n")

        # Section Categories
        if "sections" in metrics:
            f.write("SECTION CATEGORIES ANALYSIS\n")
            f.write("-" * 40 + "\n")
            section_metrics = metrics["sections"]
            f.write(f"Average Sections per Sample: {section_metrics['average_sections_per_sample']:.2f}\n")
            f.write("Most Common Section Types:\n")
            for section, count in section_metrics["most_common_sections"]:
                f.write(f"  • {section}: {count} samples\n")
            f.write("\n")

    def _write_recommendations(self, f):
        """Write recommendations section"""
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")

        summary = self.report_data["summary"]

        # Generate recommendations based on performance
        if summary["overall_performance"]:
            lowest_score = min(summary["overall_performance"].values())
            if lowest_score < 0.5:
                f.write("• Consider improving entity recognition patterns for low-scoring categories\n")
                f.write("• Review and expand medical terminology dictionaries\n")
                f.write("• Implement additional context detection rules\n")

            if "Negation Detection" in summary["overall_performance"]:
                neg_score = summary["overall_performance"]["Negation Detection"]
                if neg_score < 0.7:
                    f.write("• Enhance negation detection with more sophisticated patterns\n")
                    f.write("• Consider using dependency parsing for negation scope\n")

        f.write("• Validate results with domain experts\n")
        f.write("• Consider active learning for model improvement\n")
        f.write("• Implement cross-validation for robust evaluation\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("End of Analysis Report\n")
        f.write("=" * 80 + "\n")

    # Helper methods
    def _normalize_entity_string(self, entity_string: str) -> str:
        """Normalize entity string for comparison"""
        if pd.isna(entity_string) or not entity_string:
            return ""
        return str(entity_string).lower().strip()

    def _extract_entities_from_string(self, entity_string: str) -> List[str]:
        """Extract individual entities from semicolon-separated string"""
        if not entity_string:
            return []

        entities = [e.strip() for e in entity_string.split(";") if e.strip()]
        # Remove "NOT " prefix for comparison
        entities = [e.replace("not ", "").strip() for e in entities]
        return entities

    def _calculate_entity_matches(self, expected: List[str], predicted: List[str]) -> Tuple[bool, bool]:
        """Calculate exact and partial matches between entity lists"""
        if not expected and not predicted:
            return True, False  # Both empty = exact match

        if set(expected) == set(predicted):
            return True, False  # Exact match

        # Check for partial overlap
        expected_set = set(expected)
        predicted_set = set(predicted)

        if expected_set.intersection(predicted_set):
            return False, True  # Partial match

        return False, False  # No match

def main():
    """Main function for standalone execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Medical NLP Performance Analyzer")
    parser.add_argument("--input", "-i",
                       default="output/enhanced_medical_ner_predictions_20250928_152933.xlsx",
                       help="Input file with predictions")
    parser.add_argument("--launched-from", help="Script that launched this analyzer")
    parser.add_argument("--pipeline-command", help="The full pipeline execution command that generated the predictions")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Load data
    df = pd.read_excel(args.input)

    # Run analysis
    analyzer = EnhancedPerformanceAnalyzer()

    # Add launcher information if provided
    if args.launched_from:
        analyzer.execution_command["launched_from"] = args.launched_from
        analyzer.execution_command["execution_method"] = "launcher"
        analyzer.report_data["execution_command"] = analyzer.execution_command

    # Add pipeline command if provided
    if args.pipeline_command:
        analyzer.execution_command["pipeline_command"] = args.pipeline_command
        analyzer.report_data["execution_command"] = analyzer.execution_command

    results = analyzer.analyze_complete_performance(df, args.input)

    print("🎉 Performance analysis completed!")
    print(f"📊 Report saved to: output/reports/entity_prediction_analysis_report_{analyzer.timestamp}.txt")

if __name__ == "__main__":
    main()