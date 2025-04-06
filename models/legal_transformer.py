import numpy as np
from sentence_transformers import SentenceTransformer
import re
import json
import os

class LegalBERT:
    """Advanced legal-domain specific transformer"""
    
    def __init__(self):
        # Load the best available model for legal text
        try:
            self.model = SentenceTransformer('nlpaueb/legal-bert-base-uncased')
            self.using_legal_model = True
        except:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.using_legal_model = False
        
        # Load specialized legal patterns
        self.load_legal_patterns()
    
    def load_legal_patterns(self):
        """Load specialized legal domain knowledge"""
        # Legal citation patterns
        self.citation_patterns = {
            'case': r'([A-Za-z\s\.\,\']+)\sv\.\s([A-Za-z\s\.\,\']+),\s+(\d+)\s+([A-Za-z\.]+)\s+(\d+)\s*\(([^)]+)\)',
            'statute': r'(\d+)\s+([A-Za-z\.]+)\s+§\s*(\d+[a-z0-9\-]*)',
            'rule': r'(Rule|Fed\.\s*R\.\s*(Civ|App|Evid|Crim)\.\s*P\.)\s+(\d+)(\([a-z]\))?'
        }
        
        # Legal argument structure patterns
        self.argument_patterns = {
            # Injunction factors
            'injunction': {
                'likelihood_success': {
                    'affirm': ['likelihood of success', 'succeed on the merits', 'probability of success'],
                    'counter': ['not likely to succeed', 'no success on merits', 'fails to demonstrate likelihood']
                },
                'irreparable_harm': {
                    'affirm': ['irreparable harm', 'irreparable injury', 'cannot be remedied by damages'],
                    'counter': ['no irreparable harm', 'adequate remedy at law', 'harm is compensable']
                },
                'balance_equities': {
                    'affirm': ['balance of harms', 'balance of equities', 'balance of hardships'],
                    'counter': ['balance not in favor', 'equities do not favor', 'hardships weigh against']
                },
                'public_interest': {
                    'affirm': ['public interest', 'public good', 'serve the public'],
                    'counter': ['against public interest', 'public interest not served', 'contrary to public interest']
                }
            },
            # Property law patterns
            'property': {
                'trespass': {
                    'affirm': ['trespass', 'unauthorized entry', 'physical invasion'],
                    'counter': ['no trespass', 'authorized access', 'permission granted']
                },
                'nuisance': {
                    'affirm': ['nuisance', 'substantial interference', 'unreasonable use'],
                    'counter': ['no nuisance', 'reasonable use', 'trivial interference']
                },
                'takings': {
                    'affirm': ['taking', 'just compensation', 'fifth amendment'],
                    'counter': ['not a taking', 'no compensation required', 'public use']
                }
            },
            # Procedural patterns
            'procedural': {
                'jurisdiction': {
                    'affirm': ['jurisdiction', 'court has jurisdiction', 'subject matter jurisdiction'],
                    'counter': ['lack of jurisdiction', 'no jurisdiction', 'court lacks authority']
                },
                'standing': {
                    'affirm': ['standing', 'injury in fact', 'case or controversy'],
                    'counter': ['no standing', 'lacks standing', 'no injury in fact']
                }
            }
        }
        
        # Common legal tests and standards
        self.legal_tests = {
            'strict_scrutiny': ['compelling interest', 'narrowly tailored', 'least restrictive means'],
            'rational_basis': ['legitimate interest', 'rationally related', 'reasonably related'],
            'intermediate_scrutiny': ['important government interest', 'substantially related'],
            'balancing_test': ['weighing', 'balance', 'outweighs'],
            'rule_of_reason': ['anticompetitive effects', 'procompetitive justifications']
        }
        
        # Standard of review patterns
        self.standards_of_review = {
            'de_novo': ['de novo', 'plenary review', 'independent determination'],
            'abuse_of_discretion': ['abuse of discretion', 'discretionary', 'clearly unreasonable'],
            'clearly_erroneous': ['clearly erroneous', 'definite and firm conviction'],
            'substantial_evidence': ['substantial evidence', 'reasonable mind']
        }
    
    def extract_legal_features(self, text):
        """Extract rich legal features from text"""
        features = {
            'citations': self._extract_citations(text),
            'legal_tests': self._extract_legal_tests(text),
            'standards_of_review': self._extract_standards_of_review(text),
            'argument_patterns': self._extract_argument_patterns(text)
        }
        return features
    
    def _extract_citations(self, text):
        """Extract legal citations from text"""
        citations = {
            'cases': [],
            'statutes': [],
            'rules': [],
            'all': []
        }
        
        # Extract case citations
        case_matches = re.finditer(self.citation_patterns['case'], text)
        for match in case_matches:
            case_name = f"{match.group(1)} v. {match.group(2)}"
            full_citation = match.group(0)
            citations['cases'].append(case_name)
            citations['all'].append(full_citation)
        
        # Extract statute citations
        statute_matches = re.finditer(self.citation_patterns['statute'], text)
        for match in statute_matches:
            full_citation = match.group(0)
            citations['statutes'].append(full_citation)
            citations['all'].append(full_citation)
        
        # Extract rule citations
        rule_matches = re.finditer(self.citation_patterns['rule'], text)
        for match in rule_matches:
            full_citation = match.group(0)
            citations['rules'].append(full_citation)
            citations['all'].append(full_citation)
        
        return citations
    
    def _extract_legal_tests(self, text):
        """Extract legal tests and standards from text"""
        found_tests = {}
        text_lower = text.lower()
        
        for test_name, keywords in self.legal_tests.items():
            matches = [keyword for keyword in keywords if keyword in text_lower]
            if matches:
                found_tests[test_name] = matches
        
        return found_tests
    
    def _extract_standards_of_review(self, text):
        """Extract standards of review from text"""
        found_standards = {}
        text_lower = text.lower()
        
        for standard_name, keywords in self.standards_of_review.items():
            matches = [keyword for keyword in keywords if keyword in text_lower]
            if matches:
                found_standards[standard_name] = matches
        
        return found_standards
    
    def _extract_argument_patterns(self, text):
        """Extract argument patterns from text"""
        found_patterns = {}
        text_lower = text.lower()
        
        for category, subtypes in self.argument_patterns.items():
            found_patterns[category] = {}
            
            for subtype, patterns in subtypes.items():
                affirm_matches = [term for term in patterns['affirm'] if term in text_lower]
                counter_matches = [term for term in patterns['counter'] if term in text_lower]
                
                # Determine if affirmative or counter argument
                if affirm_matches and not counter_matches:
                    found_patterns[category][subtype] = {'type': 'affirmative', 'terms': affirm_matches}
                elif counter_matches and not affirm_matches:
                    found_patterns[category][subtype] = {'type': 'counter', 'terms': counter_matches}
                elif affirm_matches and counter_matches:
                    # Both present - determine which is stronger
                    if len(affirm_matches) > len(counter_matches):
                        found_patterns[category][subtype] = {'type': 'mixed_affirmative', 'terms': affirm_matches + counter_matches}
                    else:
                        found_patterns[category][subtype] = {'type': 'mixed_counter', 'terms': counter_matches + affirm_matches}
        
        return found_patterns
    
    def encode(self, texts, batch_size=32):
        """Encode texts using the model"""
        return self.model.encode(texts, batch_size=batch_size)
    
    def get_legal_embedding_boost(self, text1, text2):
        """Calculate legal similarity boost based on shared legal elements"""
        # Get legal features for both texts
        features1 = self.extract_legal_features(text1)
        features2 = self.extract_legal_features(text2)
        
        # Calculate citation overlap
        citations1 = set(features1['citations']['all'])
        citations2 = set(features2['citations']['all'])
        citation_overlap = len(citations1.intersection(citations2)) / max(1, len(citations1.union(citations2)))
        
        # Calculate legal test overlap
        tests1 = set(features1['legal_tests'].keys())
        tests2 = set(features2['legal_tests'].keys())
        test_overlap = len(tests1.intersection(tests2)) / max(1, len(tests1.union(tests2)))
        
        # Calculate standard of review overlap
        standards1 = set(features1['standards_of_review'].keys())
        standards2 = set(features2['standards_of_review'].keys())
        standards_overlap = len(standards1.intersection(standards2)) / max(1, len(standards1.union(standards2)))
        
        # Check for argument/counter-argument patterns
        argument_boost = 0.0
        for category, subtypes1 in features1['argument_patterns'].items():
            if category in features2['argument_patterns']:
                subtypes2 = features2['argument_patterns'][category]
                
                for subtype, data1 in subtypes1.items():
                    if subtype in subtypes2:
                        data2 = subtypes2[subtype]
                        # Check if one is affirmative and the other is counter
                        if (data1['type'].startswith('affirmative') and data2['type'].startswith('counter')) or \
                           (data1['type'].startswith('counter') and data2['type'].startswith('affirmative')):
                            argument_boost += 0.2  # Strong signal of counter-argument
        
        # Calculate weighted boost
        legal_boost = (citation_overlap * 0.4) + (test_overlap * 0.2) + (standards_overlap * 0.1) + argument_boost
        
        return min(0.5, legal_boost)  # Cap at 0.5 boost
    
    def compare_legal_arguments(self, moving_arg, response_arg):
        """Compare moving and response arguments for legal matching"""
        # Extract headings and content
        moving_heading = moving_arg['heading']
        moving_content = moving_arg['content']
        response_heading = response_arg['heading']
        response_content = response_arg['content']
        
        # Combine heading and content for full analysis
        moving_text = f"{moving_heading} {moving_content}"
        response_text = f"{response_heading} {response_content}"
        
        # Get base embeddings
        moving_embedding = self.encode([moving_text])[0]
        response_embedding = self.encode([response_text])[0]
        
        # Calculate cosine similarity
        sim = np.dot(moving_embedding, response_embedding) / (np.linalg.norm(moving_embedding) * np.linalg.norm(response_embedding))
        
        # Get legal boost
        legal_boost = self.get_legal_embedding_boost(moving_text, response_text)
        
        # Compute overall similarity
        enhanced_similarity = min(1.0, sim + legal_boost)
        
        # Extract detailed legal features for explanation
        moving_features = self.extract_legal_features(moving_text)
        response_features = self.extract_legal_features(response_text)
        
        # Find shared citations
        moving_citations = set(moving_features['citations']['all'])
        response_citations = set(response_features['citations']['all'])
        shared_citations = list(moving_citations.intersection(response_citations))
        
        # Identify argument/counter-argument pairs
        argument_pairs = []
        for category, subtypes in moving_features['argument_patterns'].items():
            if category in response_features['argument_patterns']:
                for subtype, data in subtypes.items():
                    if subtype in response_features['argument_patterns'][category]:
                        response_data = response_features['argument_patterns'][category][subtype]
                        if data['type'].startswith('affirmative') and response_data['type'].startswith('counter'):
                            argument_pairs.append({
                                'category': category, 
                                'subtype': subtype,
                                'moving_terms': data['terms'],
                                'response_terms': response_data['terms']
                            })
                        elif data['type'].startswith('counter') and response_data['type'].startswith('affirmative'):
                            argument_pairs.append({
                                'category': category, 
                                'subtype': subtype,
                                'moving_terms': data['terms'],
                                'response_terms': response_data['terms']
                            })
        
        return {
            'similarity': sim,
            'legal_boost': legal_boost,
            'enhanced_similarity': enhanced_similarity,
            'shared_citations': shared_citations,
            'argument_pairs': argument_pairs,
            'moving_features': moving_features,
            'response_features': response_features
        }