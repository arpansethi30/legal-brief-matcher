# File: utils/legal_extraction.py

import re
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class LegalExtractor:
    """Extract legal-specific features from brief arguments"""
    
    def __init__(self):
        # Legal terminology by category
        self.legal_terminology = {
            "procedural": [
                "jurisdiction", "standing", "ripeness", "mootness", "class action", 
                "certification", "dismissal", "summary judgment", "remand"
            ],
            "standards": [
                "de novo", "clearly erroneous", "abuse of discretion", "substantial evidence",
                "rational basis", "strict scrutiny", "intermediate scrutiny"
            ],
            "remedies": [
                "injunction", "preliminary injunction", "temporary restraining order",
                "specific performance", "declaratory judgment", "damages", "restitution"
            ],
            "injunction_factors": [
                "likelihood of success", "irreparable harm", "irreparable injury",
                "balance of hardships", "balance of equities", "public interest"
            ],
            "constitutional": [
                "first amendment", "second amendment", "fourth amendment", "fifth amendment",
                "fourteenth amendment", "due process", "equal protection", "takings clause"
            ],
            "property": [
                "easement", "trespass", "nuisance", "property rights", "just compensation",
                "eminent domain", "condemnation", "inverse condemnation"
            ]
        }
        
        # Compile regex patterns for efficiency
        self.case_citation_pattern = re.compile(
            r'([A-Za-z\s\.\,\']+)\sv\.\s([A-Za-z\s\.\,\']+),\s+\d+\s+[A-Za-z\.]+\s+\d+\s*\(\w+\.?\s*\d{4}\)'
        )
        self.statute_citation_pattern = re.compile(
            r'\d+\s+[A-Za-z\.]+\s+§\s*\d+[a-z0-9\-]*'
        )
        self.rule_citation_pattern = re.compile(
            r'(Rule|Fed\.\s*R\.\s*(Civ|App|Evid|Crim)\.\s*P\.)\s+\d+(\([a-z]\))?'
        )
    
    def extract_citations(self, text):
        """
        Extract legal citations from text
        
        Returns:
            dict: Citation info categorized by type
        """
        citations = {
            "cases": [],
            "statutes": [],
            "rules": [],
            "all": []
        }
        
        # Extract case citations
        case_matches = self.case_citation_pattern.findall(text)
        for match in case_matches:
            citation = f"{match[0]} v. {match[1]}"
            if citation not in citations["cases"]:
                citations["cases"].append(citation)
                citations["all"].append(citation)
        
        # Extract statute citations
        statute_matches = self.statute_citation_pattern.findall(text)
        for match in statute_matches:
            if match not in citations["statutes"]:
                citations["statutes"].append(match)
                citations["all"].append(match)
        
        # Extract rule citations
        rule_matches = self.rule_citation_pattern.findall(text)
        for match in rule_matches:
            citation = "".join(part for part in match if part)
            if citation not in citations["rules"]:
                citations["rules"].append(citation)
                citations["all"].append(citation)
        
        return citations
    
    def extract_legal_terms(self, text):
        """
        Extract legal terminology from text, categorized by type
        
        Returns:
            dict: Terms found in each category
        """
        text_lower = text.lower()
        found_terms = {category: [] for category in self.legal_terminology}
        found_terms["all"] = []
        
        # Check each category of terms
        for category, terms in self.legal_terminology.items():
            for term in terms:
                if term in text_lower:
                    found_terms[category].append(term)
                    found_terms["all"].append(term)
        
        return found_terms
    
    def extract_legal_entities(self, text):
        """
        Extract parties, laws, and other named entities
        
        Returns:
            dict: Named legal entities
        """
        entities = {
            "parties": [],
            "laws": [],
            "agencies": [],
            "courts": []
        }
        
        # Extract party names from case citations
        case_matches = self.case_citation_pattern.findall(text)
        for match in case_matches:
            entities["parties"].extend([p.strip() for p in match[:2]])
        
        # Extract laws and acts (simple pattern)
        law_pattern = re.compile(r'([A-Z][a-z]+\s)+(Act|Code|Statute|Law)')
        law_matches = law_pattern.findall(text)
        entities["laws"].extend([' '.join(m) for m in law_matches])
        
        # Extract federal agencies
        agency_pattern = re.compile(r'([A-Z][a-z]*\.?\s?)+\(("[A-Z]+"|\w+)\)')
        agency_matches = agency_pattern.findall(text)
        entities["agencies"].extend([m[0] for m in agency_matches])
        
        # Extract courts
        court_pattern = re.compile(r'(District|Circuit|Supreme)\s+Court')
        court_matches = court_pattern.findall(text)
        entities["courts"].extend(court_matches)
        
        return entities
    
    def extract_key_sentences(self, text, max_sentences=3):
        """
        Extract key sentences containing legal reasoning
        
        Returns:
            list: Key sentences from the text
        """
        sentences = sent_tokenize(text)
        key_sentences = []
        
        # Score each sentence based on legal content
        sentence_scores = []
        for sentence in sentences:
            score = 0
            
            # Check for citations
            if (self.case_citation_pattern.search(sentence) or 
                self.statute_citation_pattern.search(sentence) or
                self.rule_citation_pattern.search(sentence)):
                score += 3
            
            # Check for legal terminology
            for category, terms in self.legal_terminology.items():
                for term in terms:
                    if term in sentence.lower():
                        score += 1
                        break
            
            sentence_scores.append((sentence, score))
        
        # Sort by score and select top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        key_sentences = [s[0] for s in sentence_scores[:max_sentences]]
        
        return key_sentences
    
    def compare_arguments(self, moving_arg, response_arg):
        """
        Compare two arguments and extract shared legal elements
        
        Returns:
            dict: Shared legal elements and comparison metrics
        """
        # Extract citations from both arguments
        moving_citations = self.extract_citations(moving_arg["content"])
        response_citations = self.extract_citations(response_arg["content"])
        
        # Find shared citations
        shared_citations = [c for c in moving_citations["all"] 
                           if c in response_citations["all"]]
        
        # Extract legal terminology
        moving_terms = self.extract_legal_terms(moving_arg["content"])
        response_terms = self.extract_legal_terms(response_arg["content"])
        
        # Find shared terminology
        shared_terms = [t for t in moving_terms["all"] 
                       if t in response_terms["all"]]
        
        # Extract key sentences
        moving_key_sentences = self.extract_key_sentences(moving_arg["content"])
        response_key_sentences = self.extract_key_sentences(response_arg["content"])
        
        # Compare entities
        moving_entities = self.extract_legal_entities(moving_arg["content"])
        response_entities = self.extract_legal_entities(response_arg["content"])
        
        # Calculate citation overlap score (weighted by type)
        citation_overlap = len(shared_citations) / max(1, len(set(moving_citations["all"] + response_citations["all"])))
        
        # Calculate terminology overlap score (weighted by category)
        term_overlap = len(shared_terms) / max(1, len(set(moving_terms["all"] + response_terms["all"])))
        
        return {
            "shared_citations": shared_citations,
            "shared_terms": shared_terms,
            "moving_key_sentences": moving_key_sentences,
            "response_key_sentences": response_key_sentences,
            "citation_overlap": citation_overlap,
            "term_overlap": term_overlap
        }