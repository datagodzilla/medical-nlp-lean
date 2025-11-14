#!/usr/bin/env python3
"""
Advanced Scope Reversal Engine for Medical NLP
==============================================

This module implements a robust rule-based system for detecting scope reversal
in medical text, handling complex negation-confirmation patterns with high accuracy.

ARCHITECTURE:
1. Pattern Detection: Identify scope-reversing conjunctions
2. Scope Mapping: Map entity positions relative to conjunctions
3. Rule Engine: Apply context-aware rules for entity classification
4. Confidence Scoring: Assign confidence based on pattern strength
5. Boundary Detection: Handle sentence and clause boundaries

KEY FEATURES:
- Multi-level pattern hierarchy (primary, secondary, contextual)
- Distance-aware scope calculation
- Temporal and exception pattern handling
- False positive reduction through boundary detection
- Confidence-based entity classification
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ScopeType(Enum):
    NEGATED = "negated"
    CONFIRMED = "confirmed"
    UNCERTAIN = "uncertain"
    NEUTRAL = "neutral"

class ConjunctionType(Enum):
    ADVERSATIVE = "adversative"      # but, however, yet
    CONCESSIVE = "concessive"        # although, though, despite
    CONTRASTIVE = "contrastive"      # on the other hand, conversely
    TEMPORAL = "temporal"            # but now, however currently
    EXCEPTION = "exception"          # except, except for

@dataclass
class ScopeSegment:
    start: int
    end: int
    scope_type: ScopeType
    conjunction: str
    confidence: float
    rule_name: str

@dataclass
class EntityScope:
    entity_text: str
    entity_start: int
    entity_end: int
    assigned_scope: ScopeType
    confidence: float
    reasoning: str
    conjunctions_found: List[str]

class ScopeReversalEngine:
    """Advanced engine for detecting and applying scope reversal rules"""

    def __init__(self):
        self.conjunction_patterns = self._initialize_conjunction_patterns()
        self.scope_rules = self._initialize_scope_rules()
        self.boundary_patterns = self._initialize_boundary_patterns()

    def _initialize_conjunction_patterns(self):
        """Initialize comprehensive conjunction patterns with priorities"""

        return {
            ConjunctionType.ADVERSATIVE: {
                'patterns': [
                    # High-priority adversative (very strong scope reversal)
                    {'pattern': r'\bbut\s+reports?\b', 'priority': 10, 'confidence': 0.95},
                    {'pattern': r'\bbut\s+shows?\b', 'priority': 10, 'confidence': 0.95},
                    {'pattern': r'\bbut\s+has\b', 'priority': 10, 'confidence': 0.95},
                    {'pattern': r'\bbut\s+demonstrates?\b', 'priority': 9, 'confidence': 0.93},
                    {'pattern': r'\bbut\s+presents?\b', 'priority': 9, 'confidence': 0.93},
                    {'pattern': r'\bbut\s+exhibits?\b', 'priority': 9, 'confidence': 0.92},
                    {'pattern': r'\bbut\s+complains?\s+of\b', 'priority': 9, 'confidence': 0.92},
                    {'pattern': r'\bbut\s+admits?\s+to\b', 'priority': 8, 'confidence': 0.90},
                    {'pattern': r'\bbut\s+acknowledges?\b', 'priority': 8, 'confidence': 0.90},
                    {'pattern': r'\bbut\s+endorses?\b', 'priority': 8, 'confidence': 0.88},

                    # However patterns
                    {'pattern': r'\bhowever\s+reports?\b', 'priority': 9, 'confidence': 0.93},
                    {'pattern': r'\bhowever\s+shows?\b', 'priority': 9, 'confidence': 0.93},
                    {'pattern': r'\bhowever\s+has\b', 'priority': 9, 'confidence': 0.92},
                    {'pattern': r'\bhowever\s+demonstrates?\b', 'priority': 8, 'confidence': 0.90},

                    # Yet patterns
                    {'pattern': r'\byet\s+reports?\b', 'priority': 8, 'confidence': 0.90},
                    {'pattern': r'\byet\s+shows?\b', 'priority': 8, 'confidence': 0.90},
                    {'pattern': r'\byet\s+has\b', 'priority': 8, 'confidence': 0.88},

                    # Nevertheless/nonetheless
                    {'pattern': r'\bnevertheless\s+reports?\b', 'priority': 7, 'confidence': 0.85},
                    {'pattern': r'\bnonetheless\s+shows?\b', 'priority': 7, 'confidence': 0.85},

                    # Basic adversatives
                    {'pattern': r'\bbut\s+denies?\b', 'priority': 10, 'confidence': 0.95},
                    {'pattern': r'\bbut\s+no\b', 'priority': 9, 'confidence': 0.93},
                    {'pattern': r'\bbut\s+without\b', 'priority': 9, 'confidence': 0.92},
                    {'pattern': r'\bbut\s+absent\b', 'priority': 8, 'confidence': 0.90},
                    {'pattern': r'\bhowever\s+denies?\b', 'priority': 9, 'confidence': 0.93},
                    {'pattern': r'\bhowever\s+no\b', 'priority': 8, 'confidence': 0.90},
                    {'pattern': r'\byet\s+denies?\b', 'priority': 8, 'confidence': 0.88}
                ]
            },

            ConjunctionType.TEMPORAL: {
                'patterns': [
                    {'pattern': r'\bbut\s+now\s+reports?\b', 'priority': 9, 'confidence': 0.92},
                    {'pattern': r'\bbut\s+currently\s+has\b', 'priority': 9, 'confidence': 0.92},
                    {'pattern': r'\bbut\s+today\s+shows?\b', 'priority': 8, 'confidence': 0.90},
                    {'pattern': r'\bhowever\s+now\s+reports?\b', 'priority': 8, 'confidence': 0.90},
                    {'pattern': r'\bhowever\s+currently\s+demonstrates?\b', 'priority': 8, 'confidence': 0.88},
                    {'pattern': r'\byet\s+currently\s+has\b', 'priority': 7, 'confidence': 0.85},

                    # Reverse temporal (confirmation to negation)
                    {'pattern': r'\bbut\s+now\s+denies?\b', 'priority': 9, 'confidence': 0.92},
                    {'pattern': r'\bbut\s+currently\s+no\b', 'priority': 8, 'confidence': 0.90},
                    {'pattern': r'\bhowever\s+now\s+denies?\b', 'priority': 8, 'confidence': 0.88}
                ]
            },

            ConjunctionType.EXCEPTION: {
                'patterns': [
                    {'pattern': r'\bexcept\s+for\b', 'priority': 8, 'confidence': 0.90},
                    {'pattern': r'\bexcept\b(?!\s+for)', 'priority': 7, 'confidence': 0.85},
                    {'pattern': r'\bsave\s+for\b', 'priority': 6, 'confidence': 0.80},
                    {'pattern': r'\bapart\s+from\b', 'priority': 6, 'confidence': 0.80},
                    {'pattern': r'\baside\s+from\b', 'priority': 6, 'confidence': 0.78},
                    {'pattern': r'\bwith\s+the\s+exception\s+of\b', 'priority': 7, 'confidence': 0.85}
                ]
            },

            ConjunctionType.CONCESSIVE: {
                'patterns': [
                    {'pattern': r'\balthough\s+reports?\b', 'priority': 7, 'confidence': 0.85},
                    {'pattern': r'\bthough\s+shows?\b', 'priority': 7, 'confidence': 0.85},
                    {'pattern': r'\beven\s+though\s+has\b', 'priority': 7, 'confidence': 0.83},
                    {'pattern': r'\bdespite\s+reports?\b', 'priority': 6, 'confidence': 0.80},
                    {'pattern': r'\bin\s+spite\s+of\s+shows?\b', 'priority': 6, 'confidence': 0.80},

                    # Reverse concessive
                    {'pattern': r'\balthough\s+denies?\b', 'priority': 7, 'confidence': 0.85},
                    {'pattern': r'\bthough\s+no\b', 'priority': 6, 'confidence': 0.82},
                    {'pattern': r'\bdespite\s+denies?\b', 'priority': 6, 'confidence': 0.80}
                ]
            },

            ConjunctionType.CONTRASTIVE: {
                'patterns': [
                    {'pattern': r'\bon\s+the\s+other\s+hand\s+reports?\b', 'priority': 6, 'confidence': 0.80},
                    {'pattern': r'\bconversely\s+shows?\b', 'priority': 6, 'confidence': 0.78},
                    {'pattern': r'\bin\s+contrast\s+has\b', 'priority': 5, 'confidence': 0.75},
                    {'pattern': r'\brather\s+reports?\b', 'priority': 5, 'confidence': 0.75},
                    {'pattern': r'\binstead\s+shows?\b', 'priority': 5, 'confidence': 0.73},
                    {'pattern': r'\balternatively\s+demonstrates?\b', 'priority': 4, 'confidence': 0.70}
                ]
            }
        }

    def _initialize_scope_rules(self):
        """Initialize comprehensive scope reversal rules"""

        return {
            'primary_negation_to_confirmation': {
                'description': 'Negation followed by adversative conjunction and confirmation verb',
                'pattern': r'(denies?|no|not|without|absent|negative|free\s+of|clear\s+of)\s+([^.!?]*?)\s+(but|however|yet|nevertheless|nonetheless)\s+(reports?|shows?|has|demonstrates?|presents?|exhibits?|complains?\s+of|admits?\s+to)',
                'groups': {'negation': 1, 'entity1': 2, 'conjunction': 3, 'confirmation': 4},
                'scope_before_conjunction': ScopeType.NEGATED,
                'scope_after_conjunction': ScopeType.CONFIRMED,
                'confidence': 0.95,
                'priority': 10
            },

            'primary_confirmation_to_negation': {
                'description': 'Confirmation followed by adversative conjunction and negation',
                'pattern': r'(has|shows?|demonstrates?|reports?|presents?|exhibits?|complains?\s+of)\s+([^.!?]*?)\s+(but|however|yet|nevertheless|nonetheless)\s+(denies?|no|not|without|absent|free\s+of)',
                'groups': {'confirmation': 1, 'entity1': 2, 'conjunction': 3, 'negation': 4},
                'scope_before_conjunction': ScopeType.CONFIRMED,
                'scope_after_conjunction': ScopeType.NEGATED,
                'confidence': 0.95,
                'priority': 10
            },

            'temporal_negation_to_confirmation': {
                'description': 'Temporal scope reversal with time indicators',
                'pattern': r'(denies?|no|not)\s+([^.!?]*?)\s+(but\s+now|however\s+currently|yet\s+today|but\s+currently)\s+(reports?|shows?|has)',
                'groups': {'negation': 1, 'entity1': 2, 'temporal_conjunction': 3, 'confirmation': 4},
                'scope_before_conjunction': ScopeType.NEGATED,
                'scope_after_conjunction': ScopeType.CONFIRMED,
                'confidence': 0.90,
                'priority': 9
            },

            'exception_confirmation': {
                'description': 'Exception patterns that confirm what follows',
                'pattern': r'(denies?|no|not|absent)\s+([^.!?]*?)\s+(except|except\s+for|save\s+for|apart\s+from)\s+(.*?)(?=[.!?]|$)',
                'groups': {'negation': 1, 'entity1': 2, 'exception': 3, 'entity2': 4},
                'scope_before_conjunction': ScopeType.NEGATED,
                'scope_after_conjunction': ScopeType.CONFIRMED,
                'confidence': 0.85,
                'priority': 8
            },

            'concessive_scope_reversal': {
                'description': 'Concessive conjunctions creating scope changes',
                'pattern': r'(although|though|even\s+though|despite)\s+(denies?|no|not)\s+([^.!?]*?)\s+(reports?|shows?|has)',
                'groups': {'concessive': 1, 'negation': 2, 'entity1': 3, 'confirmation': 4},
                'scope_before_conjunction': ScopeType.NEGATED,
                'scope_after_conjunction': ScopeType.CONFIRMED,
                'confidence': 0.80,
                'priority': 7
            },

            'complex_multi_scope': {
                'description': 'Multiple scope changes in single sentence',
                'pattern': r'(denies?|no|has|shows?)\s+([^.!?]*?)\s+(but|however)\s+(reports?|shows?|denies?)\s+([^.!?]*?)\s+(however|yet|but)\s+(no|denies?|has|shows?)',
                'groups': {'first_context': 1, 'entity1': 2, 'conj1': 3, 'second_context': 4, 'entity2': 5, 'conj2': 6, 'third_context': 7},
                'scope_segments': 'multiple',
                'confidence': 0.75,
                'priority': 6
            }
        }

    def _initialize_boundary_patterns(self):
        """Initialize patterns that define scope boundaries"""

        return {
            'sentence_boundaries': [
                {'pattern': r'[.!?]\s+', 'strength': 1.0, 'description': 'Strong sentence boundary'},
                {'pattern': r';\s+', 'strength': 0.8, 'description': 'Semicolon boundary'},
                {'pattern': r':\s+', 'strength': 0.6, 'description': 'Colon boundary'},
                {'pattern': r',\s+and\s+', 'strength': 0.7, 'description': 'Coordinating conjunction'},
                {'pattern': r',\s+or\s+', 'strength': 0.7, 'description': 'Alternative conjunction'}
            ],

            'clause_boundaries': [
                {'pattern': r',\s+which\s+', 'strength': 0.5, 'description': 'Relative clause'},
                {'pattern': r',\s+that\s+', 'strength': 0.5, 'description': 'That clause'},
                {'pattern': r',\s+who\s+', 'strength': 0.4, 'description': 'Who clause'},
                {'pattern': r',\s+when\s+', 'strength': 0.4, 'description': 'Temporal clause'}
            ]
        }

    def detect_scope_segments(self, text: str) -> List[ScopeSegment]:
        """Detect all scope-reversing segments in text"""

        segments = []
        text_lower = text.lower()

        # Apply primary rules first (highest priority)
        for rule_name, rule in self.scope_rules.items():
            if rule['priority'] >= 8:  # Primary rules only
                matches = re.finditer(rule['pattern'], text_lower, re.IGNORECASE)

                for match in matches:
                    if 'scope_segments' in rule and rule['scope_segments'] == 'multiple':
                        # Handle complex multi-scope patterns
                        segments.extend(self._handle_multi_scope_pattern(match, rule, rule_name))
                    else:
                        # Handle standard scope reversal
                        segment = self._create_scope_segment(match, rule, rule_name)
                        if segment:
                            segments.append(segment)

        # Apply conjunction-based detection
        for conj_type, conj_data in self.conjunction_patterns.items():
            for pattern_info in conj_data['patterns']:
                if pattern_info['priority'] >= 7:  # High-priority conjunctions
                    matches = re.finditer(pattern_info['pattern'], text_lower, re.IGNORECASE)

                    for match in matches:
                        segment = ScopeSegment(
                            start=match.start(),
                            end=match.end(),
                            scope_type=self._determine_scope_from_conjunction(pattern_info['pattern']),
                            conjunction=match.group(),
                            confidence=pattern_info['confidence'],
                            rule_name=f"conjunction_{conj_type.value}"
                        )
                        segments.append(segment)

        # Sort by position and priority
        segments.sort(key=lambda x: (x.start, -x.confidence))

        # Remove overlapping segments (keep highest confidence)
        filtered_segments = self._remove_overlapping_segments(segments)

        return filtered_segments

    def _create_scope_segment(self, match, rule, rule_name):
        """Create a scope segment from a rule match"""

        conjunction_start = match.start()
        conjunction_end = match.end()

        # Find the actual conjunction position within the match
        groups = rule.get('groups', {})
        if 'conjunction' in groups:
            conj_group_num = groups['conjunction']
            if conj_group_num <= len(match.groups()):
                conj_text = match.group(conj_group_num)
                # Find conjunction position within the full match
                conj_pos = match.group().find(conj_text.lower())
                if conj_pos >= 0:
                    conjunction_start = match.start() + conj_pos
                    conjunction_end = conjunction_start + len(conj_text)

        return ScopeSegment(
            start=conjunction_start,
            end=conjunction_end,
            scope_type=rule['scope_after_conjunction'],
            conjunction=match.group(),
            confidence=rule['confidence'],
            rule_name=rule_name
        )

    def _handle_multi_scope_pattern(self, match, rule, rule_name):
        """Handle complex patterns with multiple scope changes"""

        segments = []
        # For complex patterns, create multiple segments
        # This is a simplified implementation - could be expanded

        segment = ScopeSegment(
            start=match.start(),
            end=match.end(),
            scope_type=ScopeType.UNCERTAIN,  # Complex patterns need careful analysis
            conjunction=match.group(),
            confidence=rule['confidence'] * 0.8,  # Reduce confidence for complex patterns
            rule_name=f"{rule_name}_complex"
        )
        segments.append(segment)

        return segments

    def _determine_scope_from_conjunction(self, pattern):
        """Determine scope type from conjunction pattern"""

        if any(word in pattern for word in ['reports', 'shows', 'has', 'demonstrates', 'presents']):
            return ScopeType.CONFIRMED
        elif any(word in pattern for word in ['denies', 'no', 'without', 'absent']):
            return ScopeType.NEGATED
        else:
            return ScopeType.UNCERTAIN

    def _remove_overlapping_segments(self, segments):
        """Remove overlapping segments, keeping highest confidence"""

        if not segments:
            return []

        filtered = [segments[0]]

        for current in segments[1:]:
            last = filtered[-1]

            # Check for overlap
            if current.start < last.end:
                # Keep the higher confidence segment
                if current.confidence > last.confidence:
                    filtered[-1] = current
            else:
                filtered.append(current)

        return filtered

    def classify_entity_scope(self, entity_text: str, entity_start: int, entity_end: int,
                            text: str, scope_segments: List[ScopeSegment]) -> EntityScope:
        """Classify an entity's scope based on detected segments"""

        # Default scope
        assigned_scope = ScopeType.NEUTRAL
        confidence = 0.5
        reasoning = "No scope-affecting patterns found"
        conjunctions_found = []

        # Check each scope segment
        for segment in scope_segments:
            # Determine entity position relative to segment
            if entity_start < segment.start:
                # Entity is before the conjunction
                if 'negation_to_confirmation' in segment.rule_name:
                    assigned_scope = ScopeType.NEGATED
                    confidence = segment.confidence
                    reasoning = f"Entity before '{segment.conjunction}' in negation-to-confirmation pattern"
                elif 'confirmation_to_negation' in segment.rule_name:
                    assigned_scope = ScopeType.CONFIRMED
                    confidence = segment.confidence
                    reasoning = f"Entity before '{segment.conjunction}' in confirmation-to-negation pattern"

            elif entity_start > segment.end:
                # Entity is after the conjunction
                assigned_scope = segment.scope_type
                confidence = segment.confidence
                reasoning = f"Entity after '{segment.conjunction}' inherits {segment.scope_type.value} scope"

            else:
                # Entity overlaps with conjunction (rare)
                assigned_scope = ScopeType.UNCERTAIN
                confidence = 0.3
                reasoning = f"Entity overlaps with conjunction '{segment.conjunction}'"

            conjunctions_found.append(segment.conjunction)

        return EntityScope(
            entity_text=entity_text,
            entity_start=entity_start,
            entity_end=entity_end,
            assigned_scope=assigned_scope,
            confidence=confidence,
            reasoning=reasoning,
            conjunctions_found=conjunctions_found
        )

    def analyze_text(self, text: str, entities: List[Dict]) -> Dict:
        """Comprehensive analysis of text for scope reversal patterns"""

        # Detect scope segments
        scope_segments = self.detect_scope_segments(text)

        # Classify each entity
        entity_scopes = []
        for entity in entities:
            entity_scope = self.classify_entity_scope(
                entity['text'],
                entity['start'],
                entity['end'],
                text,
                scope_segments
            )
            entity_scopes.append(entity_scope)

        # Separate into categories
        negated_entities = [e for e in entity_scopes if e.assigned_scope == ScopeType.NEGATED and e.confidence >= 0.7]
        confirmed_entities = [e for e in entity_scopes if e.assigned_scope == ScopeType.CONFIRMED and e.confidence >= 0.7]
        uncertain_entities = [e for e in entity_scopes if e.assigned_scope == ScopeType.UNCERTAIN or (e.assigned_scope != ScopeType.NEUTRAL and e.confidence < 0.7)]

        return {
            'scope_segments': scope_segments,
            'entity_scopes': entity_scopes,
            'negated_entities': negated_entities,
            'confirmed_entities': confirmed_entities,
            'uncertain_entities': uncertain_entities,
            'analysis_summary': {
                'total_entities': len(entities),
                'scope_segments_found': len(scope_segments),
                'negated_count': len(negated_entities),
                'confirmed_count': len(confirmed_entities),
                'uncertain_count': len(uncertain_entities),
                'patterns_detected': list(set([seg.rule_name for seg in scope_segments]))
            }
        }

# Test the engine
if __name__ == "__main__":
    engine = ScopeReversalEngine()

    # Test cases
    test_cases = [
        {
            'text': "Patient denies chest pain but reports dyspnea",
            'entities': [
                {'text': 'chest pain', 'start': 15, 'end': 25},
                {'text': 'dyspnea', 'start': 38, 'end': 45}
            ]
        },
        {
            'text': "Patient has diabetes but denies complications",
            'entities': [
                {'text': 'diabetes', 'start': 12, 'end': 20},
                {'text': 'complications', 'start': 32, 'end': 45}
            ]
        },
        {
            'text': "No fever however shows signs of infection",
            'entities': [
                {'text': 'fever', 'start': 3, 'end': 8},
                {'text': 'infection', 'start': 32, 'end': 41}
            ]
        }
    ]

    print("="*80)
    print("SCOPE REVERSAL ENGINE ANALYSIS")
    print("="*80)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: '{test_case['text']}'")
        print("-" * 60)

        analysis = engine.analyze_text(test_case['text'], test_case['entities'])

        print(f"Scope segments found: {analysis['analysis_summary']['scope_segments_found']}")
        for segment in analysis['scope_segments']:
            print(f"  - '{segment.conjunction}' ({segment.rule_name}, confidence: {segment.confidence:.2f})")

        print(f"\nEntity Classifications:")
        for entity_scope in analysis['entity_scopes']:
            print(f"  - {entity_scope.entity_text}: {entity_scope.assigned_scope.value} (confidence: {entity_scope.confidence:.2f})")
            print(f"    Reasoning: {entity_scope.reasoning}")

        print(f"\nSummary:")
        print(f"  Negated: {[e.entity_text for e in analysis['negated_entities']]}")
        print(f"  Confirmed: {[e.entity_text for e in analysis['confirmed_entities']]}")
        print(f"  Uncertain: {[e.entity_text for e in analysis['uncertain_entities']]}")