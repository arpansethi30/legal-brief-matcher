# File: utils/data_pipeline.py

import json
import numpy as np
from collections import defaultdict
import re

class LegalDataProcessor:
    """Optimized data processing pipeline for legal briefs"""
    
    def __init__(self):
        self.cleanup_patterns = [
            (r'\s+', ' '),  # Normalize whitespace
            (r'\n+', '\n'),  # Normalize newlines
            (r'\s+\n', '\n'),  # Remove trailing whitespace
            (r'\n\s+', '\n'),  # Remove leading whitespace
        ]
        
        # Citation patterns for extraction
        self.citation_patterns = {
            'case_standard': r'[A-Za-z\s\.\,\']+\sv\.\s[A-Za-z\s\.\,\']+,\s\d+\s[A-Za-z\.]+\s\d+\s*\(\w+\.?\s*\d{4}\)',
            'case_short': r'[A-Za-z\s\.\,\']+,\s\d+\s[A-Za-z\.]+\sat\s\d+',
            'statute': r'\d+\s[A-Za-z\.]+\s§\s*\d+[a-z0-9\-]*',
            'regulation': r'\d+\s[A-Za-z\.]+\s§\s*\d+\.\d+[a-z0-9\-]*',
            'constitution': r'U\.S\.\s+Const\.\s+[aA]mend\.\s+[IVX]+,\s+§\s*\d+'
        }
        
        # Legal term categories for analysis
        self.legal_term_categories = {
            'procedural': [
                'motion', 'dismiss', 'summary judgment', 'pleading', 'standing',
                'jurisdiction', 'venue', 'appeal', 'complaint', 'discovery'
            ],
            'substantive': [
                'negligence', 'breach', 'contract', 'damages', 'liability',
                'property', 'trespass', 'title', 'ownership', 'infringement'
            ],
            'remedial': [
                'injunction', 'damages', 'relief', 'specific performance', 'restitution',
                'remedy', 'award', 'compensation', 'enjoin', 'restraining order'
            ],
            'constitutional': [
                'amendment', 'constitution', 'constitutional', 'rights', 'due process',
                'equal protection', 'first amendment', 'fourth amendment', 'fifth amendment'
            ]
        }
    
    def load_brief_pair(self, file_path):
        """Load brief pair from file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle list vs single pair
            if isinstance(data, list):
                return data
            else:
                return [data]
            
        except Exception as e:
            print(f"Error loading brief pair: {e}")
            return None
    
    def preprocess_brief_pair(self, brief_pair):
        """Preprocess brief pair data"""
        processed = {
            'moving_brief': self._preprocess_brief(brief_pair['moving_brief']),
            'response_brief': self._preprocess_brief(brief_pair['response_brief']),
            'split': brief_pair.get('split', 'test')
        }
        
        # Add true links if available
        if 'true_links' in brief_pair:
            processed['true_links'] = brief_pair['true_links']
        
        # Add metadata analysis
        processed['metadata'] = self._analyze_brief_pair_metadata(processed)
        
        return processed
    
    def _preprocess_brief(self, brief):
        """Preprocess a single brief"""
        processed_brief = {
            'brief_id': brief['brief_id'],
            'brief_arguments': []
        }
        
        # Process each argument
        for arg in brief['brief_arguments']:
            # Extract all features
            citations = self._extract_citations(arg['content'])
            legal_terms = self._extract_legal_terms(arg['content'])
            heading_type = self._categorize_heading(arg['heading'])
            
            processed_arg = {
                'heading': self._cleanup_text(arg['heading']),
                'content': self._cleanup_text(arg['content']),
                'heading_normalized': self._normalize_heading(arg['heading']),
                'heading_type': heading_type,
                'citations': citations,
                'legal_terms': legal_terms,
                'length': len(arg['content']),
                'complexity': self._calculate_complexity(arg['content'])
            }
            processed_brief['brief_arguments'].append(processed_arg)
        
        return processed_brief
    
    def _analyze_brief_pair_metadata(self, processed_pair):
        """Analyze metadata for the brief pair"""
        moving_brief = processed_pair['moving_brief']
        response_brief = processed_pair['response_brief']
        
        # Calculate statistics
        moving_arg_count = len(moving_brief['brief_arguments'])
        response_arg_count = len(response_brief['brief_arguments'])
        
        # Calculate total citations by type
        moving_citations = self._combine_citation_counts(moving_brief['brief_arguments'])
        response_citations = self._combine_citation_counts(response_brief['brief_arguments'])
        
        # Calculate shared citations
        shared_citations = self._find_shared_citations(moving_brief['brief_arguments'], response_brief['brief_arguments'])
        
        # Calculate heading pattern matches
        heading_matches = self._find_heading_pattern_matches(moving_brief['brief_arguments'], response_brief['brief_arguments'])
        
        return {
            'moving_arg_count': moving_arg_count,
            'response_arg_count': response_arg_count,
            'moving_citations': moving_citations,
            'response_citations': response_citations,
            'shared_citations': shared_citations,
            'heading_matches': heading_matches
        }
    
    def _combine_citation_counts(self, arguments):
        """Combine citation counts across all arguments"""
        counts = defaultdict(int)
        for arg in arguments:
            for citation_type, citations in arg['citations'].items():
                counts[citation_type] += len(citations)
        return dict(counts)
    
    def _find_shared_citations(self, moving_arguments, response_arguments):
        """Find citations shared between moving and response briefs"""
        # Extract all citations from both briefs
        moving_all_citations = []
        for arg in moving_arguments:
            for citation_list in arg['citations'].values():
                moving_all_citations.extend(citation_list)
        
        response_all_citations = []
        for arg in response_arguments:
            for citation_list in arg['citations'].values():
                response_all_citations.extend(citation_list)
        
        # Find intersection
        shared = set(moving_all_citations).intersection(set(response_all_citations))
        return list(shared)
    
    def _find_heading_pattern_matches(self, moving_arguments, response_arguments):
        """Find heading pattern matches between briefs"""
        # Extract heading types
        moving_types = [arg['heading_type'] for arg in moving_arguments]
        response_types = [arg['heading_type'] for arg in response_arguments]
        
        # Find matches
        matches = []
        for i, m_type in enumerate(moving_types):
            for j, r_type in enumerate(response_types):
                if self._is_matching_heading_type(m_type, r_type):
                    matches.append({
                        'moving_index': i,
                        'response_index': j,
                        'moving_heading': moving_arguments[i]['heading'],
                        'response_heading': response_arguments[j]['heading'],
                    })
        
        return matches
    
    def _is_matching_heading_type(self, type1, type2):
        """Check if heading types match"""
        # Direct match
        if type1 == type2:
            return True
        
        # Match roman to arabic (I. to 1.)
        roman_to_arabic = {
            'roman_i': 'arabic_1',
            'roman_ii': 'arabic_2',
            'roman_iii': 'arabic_3',
            'roman_iv': 'arabic_4',
            'roman_v': 'arabic_5'
        }
        
        # Match roman to alphabetic (I. to A.)
        roman_to_alpha = {
            'roman_i': 'alpha_a',
            'roman_ii': 'alpha_b',
            'roman_iii': 'alpha_c',
            'roman_iv': 'alpha_d',
            'roman_v': 'alpha_e'
        }
        
        # Check for number/letter equivalence
        if type1 in roman_to_arabic and roman_to_arabic[type1] == type2:
            return True
        
        if type1 in roman_to_alpha and roman_to_alpha[type1] == type2:
            return True
        
        # Check for thematic matches (both about irreparable harm, etc.)
        thematic_patterns = [
            ('irreparable_harm', 'harm'),
            ('public_interest', 'public'),
            ('likelihood_success', 'success'),
            ('balance_equities', 'balance'),
            ('first_amendment', 'amendment'),
            ('due_process', 'process')
        ]
        
        for pattern1, pattern2 in thematic_patterns:
            if (pattern1 in type1 and pattern2 in type2) or (pattern1 in type2 and pattern2 in type1):
                return True
        
        return False
    
    def _cleanup_text(self, text):
        """Clean up text by applying patterns"""
        cleaned = text
        for pattern, replacement in self.cleanup_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        return cleaned.strip()
    
    def _normalize_heading(self, heading):
        """Normalize heading for better matching"""
        # Convert to lowercase
        lower = heading.lower()
        
        # Remove punctuation and extra spaces
        normalized = re.sub(r'[^\w\s]', ' ', lower)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove numbering
        normalized = re.sub(r'^[ivxIVX\d]+\.?\s*', '', normalized)
        
        return normalized
    
    def _categorize_heading(self, heading):
        """Categorize heading by type and pattern"""
        heading_lower = heading.lower()
        
        # Check for roman numerals
        roman_match = re.match(r'^([ivxIVX]+)\.', heading)
        if roman_match:
            numeral = roman_match.group(1).lower()
            return f"roman_{numeral}"
        
        # Check for arabic numerals
        arabic_match = re.match(r'^(\d+)\.', heading)
        if arabic_match:
            number = arabic_match.group(1)
            return f"arabic_{number}"
        
        # Check for alphabetic headings
        alpha_match = re.match(r'^([A-Za-z])\.', heading)
        if alpha_match:
            letter = alpha_match.group(1).lower()
            return f"alpha_{letter}"
        
        # Check for common legal patterns
        patterns = [
            ('irreparable', 'irreparable_harm'),
            ('harm', 'harm'),
            ('public interest', 'public_interest'),
            ('balance', 'balance_equities'),
            ('likelihood of success', 'likelihood_success'),
            ('merits', 'merits'),
            ('first amendment', 'first_amendment'),
            ('due process', 'due_process'),
            ('equal protection', 'equal_protection'),
            ('standard of review', 'standard_review'),
            ('background', 'background'),
            ('facts', 'facts'),
            ('conclusion', 'conclusion'),
            ('argument', 'argument'),
            ('standing', 'standing'),
            ('jurisdiction', 'jurisdiction')
        ]
        
        for keyword, category in patterns:
            if keyword in heading_lower:
                return category
        
        # Default to unknown
        return "unknown"
    
    def _extract_citations(self, text):
        """Extract citations from text by type"""
        citations = {}
        
        # Extract each citation type
        for citation_type, pattern in self.citation_patterns.items():
            found = re.findall(pattern, text)
            citations[citation_type] = [cite.strip() for cite in found]
        
        return citations
    
    def _extract_legal_terms(self, text):
        """Extract and categorize legal terms in text"""
        text_lower = text.lower()
        terms = {}
        
        # Extract terms by category
        for category, term_list in self.legal_term_categories.items():
            found = [term for term in term_list if term in text_lower]
            if found:
                terms[category] = found
        
        return terms
    
    def _calculate_complexity(self, text):
        """Calculate legal complexity score based on citations, terms, and sentence structure"""
        # Basic metrics
        word_count = len(text.split())
        sentence_count = len(re.split(r'[.!?]', text))
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # Count citations
        citation_count = sum(len(re.findall(pattern, text)) for pattern in self.citation_patterns.values())
        
        # Legal term density
        legal_term_count = 0
        for term_list in self.legal_term_categories.values():
            legal_term_count += sum(1 for term in term_list if term in text.lower())
        
        legal_term_density = legal_term_count / max(1, word_count)
        
        # Calculate complexity score (0-100)
        complexity = min(100, (
            (avg_sentence_length * 1.5) +  # Longer sentences are more complex
            (citation_count * 5) +  # Citations add complexity
            (legal_term_density * 100)  # Legal term density
        ))
        
        return round(complexity, 1)
    
    def format_results_for_export(self, matches):
        """Format results for JSON export in Bloomberg challenge format"""
        export_data = []
        
        # Format as [moving_heading, response_heading] pairs
        for match in matches:
            export_data.append([match['moving_heading'], match['response_heading']])
        
        return export_data
    
    def evaluate_matches(self, predicted_links, true_links):
        """Evaluate matches against ground truth"""
        if not true_links:
            return {'message': 'No ground truth available for evaluation'}
        
        # Convert to sets for comparison
        predicted_set = set(tuple(link) for link in predicted_links)
        true_set = set(tuple(link) for link in true_links)
        
        # Calculate metrics
        true_positives = predicted_set.intersection(true_set)
        false_positives = predicted_set - true_set
        false_negatives = true_set - predicted_set
        
        precision = len(true_positives) / max(1, len(predicted_set))
        recall = len(true_positives) / max(1, len(true_set))
        f1 = 2 * precision * recall / max(1e-6, precision + recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': list(true_positives),
            'false_positives': list(false_positives),
            'false_negatives': list(false_negatives)
        }