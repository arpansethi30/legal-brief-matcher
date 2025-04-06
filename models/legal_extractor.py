# File: models/legal_extractor.py

import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class LegalFeatureExtractor:
    """Extracts specialized legal features from briefs"""
    
    def __init__(self):
        # Initialize patterns
        self.case_citation_pattern = re.compile(
            r'([A-Za-z\s\.\,\']+\sv\.\s[A-Za-z\s\.\,\']+),\s+(\d+)\s+([A-Za-z\.]+)\s+(\d+)\s*\(([^)]+)\)'
        )
        self.statute_citation_pattern = re.compile(
            r'(\d+)\s+([A-Za-z\.]+)\s+§\s*(\d+[a-z0-9\-]*)'
        )
        
        # Legal terminology dictionary
        self.legal_terms = {
            "injunction_terms": [
                "injunction", "preliminary injunction", "temporary restraining order", "tro",
                "irreparable harm", "irreparable injury", "balance of harms", "balance of equities",
                "public interest", "likelihood of success", "success on the merits"
            ],
            "property_terms": [
                "property rights", "trespass", "nuisance", "easement", "right of way",
                "just compensation", "taking", "eminent domain", "inverse condemnation",
                "reasonable use", "common enemy", "natural flow", "erosion", "sediment"
            ],
            "procedural_terms": [
                "jurisdiction", "standing", "mootness", "ripeness", "exhaustion",
                "certification", "class action", "joinder", "intervention", "removal",
                "summary judgment", "motion to dismiss", "discovery", "appeal"
            ],
            "standards_terms": [
                "de novo", "clearly erroneous", "abuse of discretion", "substantial evidence",
                "rational basis", "strict scrutiny", "intermediate scrutiny", "arbitrary and capricious",
                "preponderance of the evidence", "clear and convincing", "beyond reasonable doubt"
            ]
        }
        
        # Counter-argument markers
        self.counter_markers = [
            "however", "nevertheless", "nonetheless", "on the contrary", "despite",
            "although", "even though", "notwithstanding", "in contrast", "contrary to",
            "plaintiff argues", "plaintiff contends", "plaintiff claims", "plaintiff asserts",
            "plaintiff's argument", "plaintiff's contention", "plaintiff's theory",
            "defendant mistakenly", "defendant incorrectly", "defendant fails to"
        ]
    
    def extract_citations(self, text):
        """Extract detailed citation information from text"""
        # Case citations
        case_citations = []
        for match in self.case_citation_pattern.finditer(text):
            case_info = {
                "case_name": match.group(1).strip(),
                "volume": match.group(2),
                "reporter": match.group(3),
                "page": match.group(4),
                "court_year": match.group(5),
                "full_citation": match.group(0)
            }
            case_citations.append(case_info)
        
        # Statute citations
        statute_citations = []
        for match in self.statute_citation_pattern.finditer(text):
            statute_info = {
                "title": match.group(1),
                "code": match.group(2),
                "section": match.group(3),
                "full_citation": match.group(0)
            }
            statute_citations.append(statute_info)
        
        return {
            "cases": case_citations,
            "statutes": statute_citations,
            "all": [c["full_citation"] for c in case_citations] + [s["full_citation"] for s in statute_citations]
        }
    
    def extract_legal_terminology(self, text):
        """Extract legal terminology from text by category"""
        text_lower = text.lower()
        found_terms = {category: [] for category in self.legal_terms}
        all_found = []
        
        # Search for terms by category
        for category, terms in self.legal_terms.items():
            for term in terms:
                if term in text_lower:
                    found_terms[category].append(term)
                    all_found.append(term)
        
        found_terms["all"] = all_found
        return found_terms
    
    def extract_counter_argument_markers(self, text):
        """Extract markers of counter-argumentation"""
        text_lower = text.lower()
        
        found_markers = []
        for marker in self.counter_markers:
            if marker in text_lower:
                # Get context around marker (to see what's being countered)
                marker_index = text_lower.find(marker)
                start = max(0, marker_index - 50)
                end = min(len(text_lower), marker_index + len(marker) + 100)
                context = text_lower[start:end]
                
                found_markers.append({
                    "marker": marker,
                    "context": context
                })
        
        return found_markers
    
    def extract_key_sentences(self, text, max_sentences=3):
        """Extract key sentences containing legal reasoning"""
        sentences = sent_tokenize(text)
        
        # Score each sentence
        scored_sentences = []
        for sentence in sentences:
            score = 0
            
            # Citations are strong signals
            if self.case_citation_pattern.search(sentence) or self.statute_citation_pattern.search(sentence):
                score += 3
            
            # Legal terminology
            sent_lower = sentence.lower()
            for category, terms in self.legal_terms.items():
                if any(term in sent_lower for term in terms):
                    score += 1
            
            # Length preference (15-30 words ideal)
            words = word_tokenize(sentence)
            word_count = len(words)
            if 15 <= word_count <= 30:
                score += 1
            
            scored_sentences.append((sentence, score))
        
        # Sort and return top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored_sentences[:max_sentences]]
    
    def compare_arguments(self, moving_arg, response_arg):
        """Compare moving and response arguments for shared legal features"""
        # Extract citations
        moving_citations = self.extract_citations(moving_arg["content"])
        response_citations = self.extract_citations(response_arg["content"])
        
        # Find shared citations
        shared_citations = [c for c in moving_citations["all"] 
                           if c in response_citations["all"]]
        
        # Extract terminology
        moving_terms = self.extract_legal_terminology(moving_arg["content"])
        response_terms = self.extract_legal_terminology(response_arg["content"])
        
        # Find shared terminology
        shared_terms = [t for t in moving_terms["all"] 
                       if t in response_terms["all"]]
        
        # Extract counter-argument markers
        counter_markers = self.extract_counter_argument_markers(response_arg["content"])
        
        # Extract key sentences
        moving_key = self.extract_key_sentences(moving_arg["content"])
        response_key = self.extract_key_sentences(response_arg["content"])
        
        # Calculate similarity metrics
        citation_overlap = len(shared_citations) / max(1, len(set(moving_citations["all"] + response_citations["all"])))
        term_overlap = len(shared_terms) / max(1, len(set(moving_terms["all"] + response_terms["all"])))
        counter_argument_score = min(1.0, len(counter_markers) * 0.2)
        
        return {
            "shared_citations": shared_citations,
            "shared_terms": shared_terms,
            "counter_markers": counter_markers,
            "moving_key_sentences": moving_key,
            "response_key_sentences": response_key,
            "citation_overlap": citation_overlap,
            "term_overlap": term_overlap,
            "counter_argument_score": counter_argument_score
        }
