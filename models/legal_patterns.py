# File: models/legal_patterns.py

import re
import json
from collections import defaultdict

class LegalArgumentStructure:
    """Recognizes legal argument structures and counter-argument patterns"""
    
    def __init__(self):
        # Load argument pattern model
        self.pattern_types = {
            "INJUNCTION_FACTOR": [
                ("likelihood_success", ["likelihood of success", "succeed on the merits"], 
                 ["not likely to succeed", "no success on the merits"]),
                ("irreparable_harm", ["irreparable harm", "irreparable injury"], 
                 ["no irreparable harm", "adequate remedy"]),
                ("balance_equities", ["balance of harm", "balance of equities"], 
                 ["balance not in favor", "equity favors"]),
                ("public_interest", ["public interest", "interest of the public"], 
                 ["against public interest", "public interest weighs"])
            ],
            "CONSTITUTIONAL_CLAIM": [
                ("takings_clause", ["taking", "just compensation", "fifth amendment"], 
                 ["not a taking", "no compensation required"]),
                ("due_process", ["due process", "procedural due process"], 
                 ["no due process violation", "process was adequate"]),
                ("equal_protection", ["equal protection", "discriminatory"], 
                 ["rationally related", "no discrimination"])
            ],
            "PROCEDURAL": [
                ("jurisdiction", ["jurisdiction", "subject matter jurisdiction"], 
                 ["lack of jurisdiction", "no jurisdiction"]),
                ("standing", ["standing", "injury in fact"], 
                 ["no standing", "lacks standing"]),
                ("ripeness", ["ripe for review", "justiciable"], 
                 ["not ripe", "premature"])
            ]
        }
        
        # Legal numbering patterns
        self.numbering_maps = {
            "roman_to_arabic": {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5},
            "roman_to_alpha": {"I": "A", "II": "B", "III": "C", "IV": "D", "V": "E"},
            "arabic_to_alpha": {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
        }
    
    def identify_argument_type(self, heading, content):
        """Identify the type of legal argument"""
        text = (heading + " " + content[:500]).lower()
        
        for category, patterns in self.pattern_types.items():
            for pattern_name, positive_terms, _ in patterns:
                for term in positive_terms:
                    if term.lower() in text:
                        return category, pattern_name
        
        return "GENERAL", "general_argument"
    
    def identify_counter_argument_type(self, heading, content):
        """Identify the type of legal counter-argument"""
        text = (heading + " " + content[:500]).lower()
        
        for category, patterns in self.pattern_types.items():
            for pattern_name, _, negative_terms in patterns:
                for term in negative_terms:
                    if term.lower() in text:
                        return category, pattern_name
        
        return "GENERAL", "general_counter"
    
    def extract_section_number(self, heading):
        """Extract section number from heading (Roman, Arabic, Alpha)"""
        # Try Roman numerals
        roman_match = re.match(r'^([IVX]+)[\.\s]', heading)
        if roman_match:
            roman = roman_match.group(1)
            return "roman", roman
        
        # Try Arabic numerals
        arabic_match = re.match(r'^(\d+)[\.\s]', heading)
        if arabic_match:
            arabic = int(arabic_match.group(1))
            return "arabic", arabic
        
        # Try alphabetic
        alpha_match = re.match(r'^([A-Z])[\.\s]', heading)
        if alpha_match:
            alpha = alpha_match.group(1)
            return "alpha", alpha
        
        return None, None
    
    def detect_heading_pattern_match(self, moving_heading, response_heading):
        """Detect if headings follow standard pattern matching"""
        # Get section numbering types
        moving_type, moving_value = self.extract_section_number(moving_heading)
        response_type, response_value = self.extract_section_number(response_heading)
        
        if moving_type and response_type:
            # Direct matches (same type, same value)
            if moving_type == response_type and moving_value == response_value:
                return 1.0, "direct_match"
            
            # Roman -> Arabic mapping
            if moving_type == "roman" and response_type == "arabic":
                if moving_value in self.numbering_maps["roman_to_arabic"]:
                    if self.numbering_maps["roman_to_arabic"][moving_value] == response_value:
                        return 0.9, "roman_to_arabic"
            
            # Roman -> Alpha mapping
            if moving_type == "roman" and response_type == "alpha":
                if moving_value in self.numbering_maps["roman_to_alpha"]:
                    if self.numbering_maps["roman_to_alpha"][moving_value] == response_value:
                        return 0.9, "roman_to_alpha"
            
            # Arabic -> Alpha mapping
            if moving_type == "arabic" and response_type == "alpha":
                if moving_value in self.numbering_maps["arabic_to_alpha"]:
                    if self.numbering_maps["arabic_to_alpha"][moving_value] == response_value:
                        return 0.8, "arabic_to_alpha"
        
        return 0.0, None
    
    def detect_argument_counter_pair(self, moving_heading, moving_content, response_heading, response_content):
        """Detect if arguments form a direct argument/counter-argument pair"""
        # Check for pattern matches
        moving_category, moving_pattern = self.identify_argument_type(moving_heading, moving_content)
        response_category, response_pattern = self.identify_counter_argument_type(response_heading, response_content)
        
        # Direct pattern matches
        if moving_category == response_category and moving_pattern == response_pattern:
            return 1.0, moving_pattern
        
        # Check heading pattern match
        heading_score, heading_pattern = self.detect_heading_pattern_match(moving_heading, response_heading)
        if heading_score > 0.5:
            return heading_score, f"heading_pattern:{heading_pattern}"
        
        # Look for refutation language
        refutation_terms = [
            "contrary to", "unlike", "plaintiff's argument", "defendants disagree",
            "however", "nevertheless", "mistakenly", "incorrectly"
        ]
        
        response_lower = response_content.lower()
        refutation_count = sum(1 for term in refutation_terms if term in response_lower)
        if refutation_count > 0:
            refutation_score = min(0.7, refutation_count * 0.1)
            return refutation_score, "refutation_language"
        
        return 0.0, None