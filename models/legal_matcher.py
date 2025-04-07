# File: models/legal_matcher.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from collections import defaultdict
import re
from models.legal_transformer import LegalBERT
from models.llama_analyzer import LlamaLegalAnalyzer

class LegalArgumentMatcher:
    """Specialized legal argument matching system"""
    
    def __init__(self):
        self.legal_bert = LegalBERT()
        self.section_patterns = self._load_section_patterns()
        # Only initialize LlamaLegalAnalyzer if needed, to save resources
        self.llama_analyzer = None
        self.use_llama = False
    
    def _init_llama_analyzer(self):
        """Lazy initialization of Llama analyzer only when needed"""
        if not self.llama_analyzer:
            from models.llama_analyzer import LlamaLegalAnalyzer
            self.llama_analyzer = LlamaLegalAnalyzer()
            self.use_llama = self.llama_analyzer.is_available
    
    def _load_section_patterns(self):
        """Load patterns for matching brief sections"""
        return {
            'injunction_factors': [
                (r'likelihood\s+of\s+success', r'(not\s+)?likely\s+to\s+succeed'),
                (r'irreparable\s+harm', r'(no\s+)?irreparable\s+harm'),
                (r'balance\s+of\s+(the\s+)?(harms|equities|hardships)', r'balance\s+of\s+(the\s+)?(equities|hardships|harms)(\s+not)?'),
                (r'public\s+interest', r'public\s+interest')
            ],
            'constitutional_factors': [
                (r'first\s+amendment', r'first\s+amendment'),
                (r'fourth\s+amendment', r'fourth\s+amendment'),
                (r'due\s+process', r'due\s+process'),
                (r'equal\s+protection', r'equal\s+protection')
            ],
            'contract_factors': [
                (r'breach', r'(no\s+)?breach'),
                (r'consideration', r'(lack\s+of\s+)?consideration'),
                (r'damages', r'(no\s+)?damages')
            ],
            'heading_patterns': [
                (r'^([IVX]+)\.', r'^(\d+)\.'),  # Roman to Arabic
                (r'^([IVX]+)\.', r'^([A-Z])\.'),  # Roman to Alphabetic
                (r'([IVX]+)\.', r'\1')  # Exact Roman match
            ]
        }
    
    def extract_heading_type(self, heading):
        """Extract heading type and number"""
        heading_lower = heading.lower()
        
        # Check for injunction factors
        for pattern_name, patterns in self.section_patterns.items():
            if pattern_name.endswith('_factors'):
                for i, (affirm_pattern, counter_pattern) in enumerate(patterns):
                    if re.search(affirm_pattern, heading_lower):
                        return f"{pattern_name}_{i}", 'affirm'
                    if re.search(counter_pattern, heading_lower):
                        return f"{pattern_name}_{i}", 'counter'
        
        # Extract numbering scheme
        roman_match = re.match(r'^([IVX]+)\.', heading)
        if roman_match:
            return 'roman', roman_match.group(1)
        
        arabic_match = re.match(r'^(\d+)\.', heading)
        if arabic_match:
            return 'arabic', arabic_match.group(1)
        
        alpha_match = re.match(r'^([A-Z])\.', heading)
        if alpha_match:
            return 'alpha', alpha_match.group(1)
        
        return 'unknown', None
    
    def is_counter_argument(self, moving_heading, response_heading):
        """Check if response heading is a counter to moving heading"""
        moving_type, moving_value = self.extract_heading_type(moving_heading)
        response_type, response_value = self.extract_heading_type(response_heading)
        
        # Check for direct injunction factor matches
        if moving_type.startswith('injunction_factors_') and response_type.startswith('injunction_factors_'):
            factor_num = moving_type.split('_')[-1]
            response_factor_num = response_type.split('_')[-1]
            
            if factor_num == response_factor_num:
                return True
        
        # Check for constitutional factor matches
        if moving_type.startswith('constitutional_factors_') and response_type.startswith('constitutional_factors_'):
            factor_num = moving_type.split('_')[-1]
            response_factor_num = response_type.split('_')[-1]
            
            if factor_num == response_factor_num:
                return True
        
        # Check for contract factor matches
        if moving_type.startswith('contract_factors_') and response_type.startswith('contract_factors_'):
            factor_num = moving_type.split('_')[-1]
            response_factor_num = response_type.split('_')[-1]
            
            if factor_num == response_factor_num:
                return True
        
        # Check for numbering scheme matches
        roman_to_arabic = {'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5', 'VI': '6', 'VII': '7', 'VIII': '8', 'IX': '9', 'X': '10'}
        roman_to_alpha = {'I': 'A', 'II': 'B', 'III': 'C', 'IV': 'D', 'V': 'E', 'VI': 'F', 'VII': 'G', 'VIII': 'H', 'IX': 'I', 'X': 'J'}
        
        if moving_type == 'roman' and response_type == 'arabic':
            if moving_value in roman_to_arabic and roman_to_arabic[moving_value] == response_value:
                return True
        
        if moving_type == 'roman' and response_type == 'alpha':
            if moving_value in roman_to_alpha and roman_to_alpha[moving_value] == response_value:
                return True
        
        return False
    
    def analyze_brief_pair(self, brief_pair):
        """Analyze a brief pair and generate matches with optimized performance"""
        moving_arguments = brief_pair["moving_brief"]["brief_arguments"]
        response_arguments = brief_pair["response_brief"]["brief_arguments"]
        
        # Limit the number of arguments to analyze to prevent excessive CPU usage
        max_args = 15
        moving_arguments = moving_arguments[:max_args] if len(moving_arguments) > max_args else moving_arguments
        response_arguments = response_arguments[:max_args] if len(response_arguments) > max_args else response_arguments
        
        # Generate all pairwise comparisons
        comparisons = []
        # Use progress tracking to allow for interruption
        total_comparisons = len(moving_arguments) * len(response_arguments)
        print(f"Processing {total_comparisons} argument pairs...")
        
        for i, moving_arg in enumerate(moving_arguments):
            for j, response_arg in enumerate(response_arguments):
                # First check structural matches (cheap operation)
                heading_match = self.is_counter_argument(moving_arg["heading"], response_arg["heading"])
                
                # Then get detailed comparison
                comparison = self.legal_bert.compare_legal_arguments(moving_arg, response_arg)
                
                # Calculate enhanced confidence with detailed factors
                enhanced_confidence, confidence_factors = self.calculate_enhanced_confidence(
                    comparison, 
                    heading_match=heading_match,
                    moving_arg=moving_arg,
                    response_arg=response_arg
                )
                
                # Only do expensive Llama analysis for high-confidence matches
                counter_arg_analysis = {}
                if enhanced_confidence > 0.6 and self.use_llama:
                    try:
                        # Lazy initialization of Llama analyzer
                        self._init_llama_analyzer()
                        if self.llama_analyzer and self.llama_analyzer.is_available:
                            counter_arg_analysis = self.llama_analyzer.analyze_counter_argument(
                                moving_arg, response_arg
                            )
                            
                            # Use the counter argument quality score to slightly adjust the confidence
                            counter_quality = counter_arg_analysis.get("counter_quality_score", 0.0)
                            # Apply a smaller adjustment to avoid overwhelming other factors
                            counter_quality_boost = counter_quality * 0.15
                            enhanced_confidence = min(1.0, enhanced_confidence + counter_quality_boost)
                            
                            # Add to confidence factors
                            confidence_factors["counter_quality"] = {
                                'value': round(counter_quality_boost, 3),
                                'explanation': f"LLM analysis determined this is a {counter_quality*10:.1f}/10 quality counter-argument"
                            }
                            confidence_factors["final_score"]["value"] = round(enhanced_confidence, 3)
                    except Exception as e:
                        print(f"Error in Llama analysis: {str(e)}")
                
                # Add to comparisons
                comparisons.append({
                    'moving_index': i,
                    'response_index': j,
                    'moving_heading': moving_arg["heading"],
                    'response_heading': response_arg["heading"],
                    'moving_content': moving_arg["content"][:1000],  # Limit content size in memory
                    'response_content': response_arg["content"][:1000],  # Limit content size in memory
                    'similarity': comparison['similarity'],
                    'legal_boost': comparison['legal_boost'],
                    'enhanced_similarity': enhanced_confidence,
                    'confidence_factors': confidence_factors,
                    'heading_match': heading_match,
                    'shared_citations': comparison['shared_citations'][:10],  # Limit number of citations
                    'argument_pairs': comparison['argument_pairs'],
                    'counter_analysis': counter_arg_analysis
                })
                
                # Clear memory periodically
                if (i * len(response_arguments) + j + 1) % 20 == 0:
                    import gc
                    gc.collect()
                    print(f"Processed {i * len(response_arguments) + j + 1}/{total_comparisons} comparisons...")
        
        # Sort by enhanced similarity
        comparisons.sort(key=lambda x: x['enhanced_similarity'], reverse=True)
        
        return comparisons
    
    def calculate_enhanced_confidence(self, comparison, heading_match=False, moving_arg=None, response_arg=None):
        """
        Calculate enhanced confidence score with detailed breakdown of factors and explanations
        """
        # Base similarity
        base_similarity = comparison['similarity']
        
        # Citation boost - weighted by importance of citations
        citation_count = len(comparison['shared_citations'])
        citation_boost = min(0.2, citation_count * 0.05)
        
        # Heading match boost
        heading_boost = 0.15 if heading_match else 0.0
        
        # Legal terminology boost
        legal_terminology_boost = comparison.get('legal_boost', 0.0)
        
        # Legal pattern boost (counter-argument patterns)
        pattern_boost = 0.1 if comparison.get('argument_pairs', []) else 0.0
        
        # Length ratio penalty - penalize large disparities in argument length
        if moving_arg and response_arg:
            moving_length = len(moving_arg['content'])
            response_length = len(response_arg['content'])
            length_ratio = min(moving_length, response_length) / max(moving_length, response_length)
            length_penalty = 0.1 * (1 - length_ratio)  # Max penalty of 0.1
        else:
            length_penalty = 0
            length_ratio = 1.0
        
        # Precedent strength impact
        precedent_impact = 0.0
        if 'precedent_analysis' in comparison:
            # Add a small boost if response has strong precedents
            precedent_rel_strength = comparison['precedent_analysis'].get('relative_strength', 0.0)
            precedent_impact = abs(precedent_rel_strength) * 0.05  # Small boost based on precedent strength
        
        # Combined confidence with ceiling
        enhanced_confidence = min(1.0, 
                                 base_similarity 
                                 + citation_boost 
                                 + heading_boost 
                                 + legal_terminology_boost
                                 + pattern_boost
                                 + precedent_impact
                                 - length_penalty)
        
        # Create detailed confidence factors explanation with explanations for the judges
        confidence_factors = {
            'base_similarity': {
                'value': round(base_similarity, 3),
                'explanation': f"Semantic similarity of {round(base_similarity*100, 1)}% between the arguments based on legal-domain language model"
            },
            'citation_boost': {
                'value': round(citation_boost, 3),
                'explanation': f"Boost of {round(citation_boost*100, 1)}% for {citation_count} shared legal citations between the briefs"
            },
            'heading_match': {
                'value': round(heading_boost, 3),
                'explanation': "15% boost applied for matching argument section headings" if heading_match else "No boost applied for headings (no structural match)"
            },
            'legal_terminology': {
                'value': round(legal_terminology_boost, 3),
                'explanation': f"Legal domain terminology boost of {round(legal_terminology_boost*100, 1)}% based on shared legal tests and standards"
            },
            'pattern_match': {
                'value': round(pattern_boost, 3),
                'explanation': "10% boost for detected counter-argument patterns" if pattern_boost > 0 else "No counter-argument patterns detected"
            },
            'length_penalty': {
                'value': round(length_penalty, 3),
                'explanation': f"Penalty of {round(length_penalty*100, 1)}% due to length disparity (ratio: {round(length_ratio, 2)})"
            },
            'precedent_impact': {
                'value': round(precedent_impact, 3),
                'explanation': f"Impact of {round(precedent_impact*100, 1)}% from precedent strength analysis"
            },
            'final_score': {
                'value': round(enhanced_confidence, 3),
                'explanation': f"Final confidence score of {round(enhanced_confidence*100, 1)}% indicates " + 
                              ("strong" if enhanced_confidence > 0.7 else 
                               "moderate" if enhanced_confidence > 0.5 else "weak") + 
                              " match between arguments"
            }
        }
        
        # Simplified return format for existing code compatibility
        simplified_factors = {k: v['value'] for k, v in confidence_factors.items()}
        
        return enhanced_confidence, confidence_factors
    
    def generate_optimal_matches(self, comparisons, moving_count, response_count):
        """Generate optimal matches using Hungarian algorithm with fallback for large matrices"""
        # For large brief pairs, limit the size of analysis to prevent excessive CPU usage
        max_dimensions = 20  # Maximum dimensions to use Hungarian algorithm
        
        if moving_count <= max_dimensions and response_count <= max_dimensions:
            # Initialize cost matrix (negative similarity to convert to cost)
            cost_matrix = np.ones((moving_count, response_count)) * 2.0  # Initialize with high cost
            
            # Fill in costs
            for comp in comparisons:
                i, j = comp['moving_index'], comp['response_index']
                # Convert similarity to cost (lower is better)
                cost_matrix[i, j] = 1.0 - comp['enhanced_similarity']
            
            # Run Hungarian algorithm to find optimal assignments
            try:
                from scipy.optimize import linear_sum_assignment
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                # Extract optimal matches
                optimal_matches = []
                for i, j in zip(row_ind, col_ind):
                    # Only include matches above a minimum confidence threshold
                    match_confidence = 1.0 - cost_matrix[i, j]
                    if match_confidence >= 0.35:  # Minimum threshold for matches
                        # Find the original comparison data
                        for comp in comparisons:
                            if comp['moving_index'] == i and comp['response_index'] == j:
                                optimal_matches.append(comp)
                                break
                
                # Sort by moving index to maintain original brief order
                optimal_matches.sort(key=lambda x: x['moving_index'])
                
                return optimal_matches
            except Exception as e:
                print(f"Error in Hungarian algorithm: {str(e)}, falling back to greedy algorithm")
                # Fall back to greedy algorithm if scipy not available or error occurs
                return self.greedy_matches(comparisons, moving_count)
        else:
            print(f"Brief pair too large ({moving_count}x{response_count}), using greedy matching")
            # For large matrices, use greedy algorithm
            return self.greedy_matches(comparisons, moving_count)
    
    def greedy_matches(self, comparisons, moving_count):
        """Greedy algorithm for matching"""
        matches = []
        used_moving = set()
        used_response = set()
        
        # First pass - Take heading matches with high similarity
        for comp in comparisons:
            if comp['heading_match'] and comp['enhanced_similarity'] > 0.6:
                i, j = comp['moving_index'], comp['response_index']
                if i not in used_moving and j not in used_response:
                    matches.append(comp)
                    used_moving.add(i)
                    used_response.add(j)
        
        # Second pass - Fill in remaining matches
        for comp in comparisons:
            i, j = comp['moving_index'], comp['response_index']
            if i not in used_moving and comp['enhanced_similarity'] > 0.4:
                # Find best available response
                best_j = None
                best_similarity = 0.0
                
                for c in comparisons:
                    if c['moving_index'] == i and c['response_index'] not in used_response:
                        if c['enhanced_similarity'] > best_similarity:
                            best_j = c['response_index']
                            best_similarity = c['enhanced_similarity']
                            best_comp = c
                
                if best_j is not None:
                    matches.append(best_comp)
                    used_moving.add(i)
                    used_response.add(best_j)
        
        # Sort by moving index
        matches.sort(key=lambda x: x['moving_index'])
        
        return matches
    
    def generate_match_explanation(self, match):
        """Generate detailed explanation for match"""
        explanation_parts = []
        
        # Add heading match explanation
        if match['heading_match']:
            if 'injunction_factors' in match['moving_heading'].lower() and 'injunction_factors' in match['response_heading'].lower():
                explanation_parts.append("Both arguments address the same injunction factor but with opposing conclusions")
            elif 'public' in match['moving_heading'].lower() and 'public' in match['response_heading'].lower():
                explanation_parts.append("Both arguments address the public interest factor with opposing evaluations")
            elif 'harm' in match['moving_heading'].lower() and 'harm' in match['response_heading'].lower():
                explanation_parts.append("Moving brief claims harm while response contests its existence or irreparability")
            else:
                explanation_parts.append("Arguments occupy corresponding positions in brief structure")
        
        # Add argument/counter-argument pattern explanation
        if match['argument_pairs']:
            for pair in match['argument_pairs']:
                if pair['category'] == 'injunction' and pair['subtype'] == 'likelihood_success':
                    explanation_parts.append("Moving brief argues likelihood of success while response counters that success is unlikely")
                elif pair['category'] == 'injunction' and pair['subtype'] == 'irreparable_harm':
                    explanation_parts.append("Moving brief claims irreparable harm while response argues harm is not irreparable")
                elif pair['category'] == 'injunction' and pair['subtype'] == 'balance_equities':
                    explanation_parts.append("Moving brief and response offer opposing views on the balance of equities")
                elif pair['category'] == 'injunction' and pair['subtype'] == 'public_interest':
                    explanation_parts.append("Arguments present contrary positions on whether an injunction serves the public interest")
                elif pair['category'] == 'property' and pair['subtype'] == 'trespass':
                    explanation_parts.append("Moving brief alleges trespass while response brief contests this claim")
                elif pair['category'] == 'constitutional' and pair['subtype'] == 'first_amendment':
                    explanation_parts.append("Moving brief raises First Amendment argument while response brief disputes its applicability")
                elif pair['category'] == 'contract' and pair['subtype'] == 'breach':
                    explanation_parts.append("Moving brief argues contract breach while response contests breach elements")
        
        # Add citation explanation
        if match['shared_citations']:
            if len(match['shared_citations']) == 1:
                explanation_parts.append(f"Both arguments cite the same legal authority: {match['shared_citations'][0]}")
            elif len(match['shared_citations']) > 1:
                explanation_parts.append(f"Both arguments cite {len(match['shared_citations'])} of the same legal authorities")
        
        # Add confidence score explanation
        if 'confidence_factors' in match:
            factors = match['confidence_factors']
            significant_factors = []
            
            # Handle dictionary format with 'value' key
            citation_boost_value = factors['citation_boost']['value'] if isinstance(factors['citation_boost'], dict) else factors['citation_boost']
            heading_match_value = factors['heading_match']['value'] if isinstance(factors['heading_match'], dict) else factors['heading_match']
            legal_terminology_value = factors['legal_terminology']['value'] if isinstance(factors['legal_terminology'], dict) else factors['legal_terminology']
            
            if citation_boost_value > 0.05:
                significant_factors.append(f"shared legal citations (contributing +{citation_boost_value:.2f})")
            if heading_match_value > 0:
                significant_factors.append(f"matching argument structure (+{heading_match_value:.2f})")
            if legal_terminology_value > 0.05:
                significant_factors.append(f"shared legal terminology (+{legal_terminology_value:.2f})")
            
            if significant_factors:
                explanation_parts.append(f"Match strength enhanced by: {', '.join(significant_factors)}")
        
        # Default explanation if nothing specific found
        if not explanation_parts:
            explanation_parts.append("Arguments show semantic similarity in addressing related legal issues")
        
        return ". ".join(explanation_parts)
    
    def format_matches(self, matches):
        """Format matches for visualization and display"""
        formatted_matches = []
        
        for match in matches:
            # Generate explanation
            explanation = self.generate_match_explanation(match)
            
            # Format match
            formatted_match = {
                'moving_heading': match['moving_heading'],
                'response_heading': match['response_heading'],
                'moving_content': match['moving_content'],
                'response_content': match['response_content'],
                'moving_index': match['moving_index'],
                'response_index': match['response_index'],
                'confidence': match['enhanced_similarity'],
                'confidence_factors': match.get('confidence_factors', {}),
                'explanation': explanation,
                'shared_citations': match.get('shared_citations', []),
                'shared_terms': match.get('shared_terms', [])
            }
            
            formatted_matches.append(formatted_match)
        
        return formatted_matches