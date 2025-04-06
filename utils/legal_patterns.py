# File: utils/legal_patterns.py

import re
import json
from collections import defaultdict

class LegalPatternMatcher:
    """Specialized pattern matching for legal arguments"""
    
    def __init__(self):
        # Common patterns in legal arguments/counter-arguments
        self.brief_section_patterns = {
            # Standard injunction factors
            "likelihood_of_success": {
                "moving": [r"likelihood\s+of\s+success", r"success\s+on\s+the\s+merits"],
                "response": [r"not\s+likely\s+to\s+succeed", r"no\s+success\s+on\s+the\s+merits", r"fails\s+to\s+demonstrate"]
            },
            "irreparable_harm": {
                "moving": [r"irreparable\s+harm", r"imminent\s+(irreparable\s+)?harm"],
                "response": [r"no\s+irreparable\s+harm", r"harm\s+is\s+not\s+irreparable", r"adequate\s+remedy\s+at\s+law"]
            },
            "balance_of_harms": {
                "moving": [r"balance\s+of\s+(the\s+)?harms", r"balance\s+of\s+hardships"],
                "response": [r"balance\s+of\s+(the\s+)?(equities|hardships)", r"balance.*not\s+in\s+.*favor"]
            },
            "public_interest": {
                "moving": [r"public\s+interest", r"serves\s+the\s+public"],
                "response": [r"not\s+in\s+the\s+public\s+interest", r"public\s+interest"]
            },
            # Constitutional claims
            "takings_clause": {
                "moving": [r"taking", r"just\s+compensation", r"fifth\s+amendment"],
                "response": [r"not\s+a\s+taking", r"inverse\s+condemnation", r"compensation"]
            },
            # Procedural matters
            "jurisdiction": {
                "moving": [r"jurisdiction", r"court\s+has\s+jurisdiction"],
                "response": [r"lack\s+of\s+jurisdiction", r"no\s+jurisdiction", r"jurisdictional\s+bar"]
            },
            "standing": {
                "moving": [r"standing", r"injury\s+in\s+fact"],
                "response": [r"no\s+standing", r"lacks\s+standing", r"failed\s+to\s+demonstrate\s+standing"]
            },
            # Property law
            "property_rights": {
                "moving": [r"property\s+rights", r"right\s+of\s+exclusion", r"trespass"],
                "response": [r"no\s+property\s+right", r"easement", r"common\s+enemy"]
            },
        }
        
        # Legal numbering patterns
        self.numbering_patterns = {
            "roman": r"^([IVX]+)\.",
            "arabic": r"^(\d+)\.",
            "alphabetic": r"^([A-Z])\.",
            "parenthetical": r"^(\d+)\)"
        }
        
        # Counter-argument linguistic markers
        self.counter_argument_markers = [
            "however", "contrary", "unlike", "conversely", "nevertheless", 
            "incorrectly", "mistakenly", "fails to", "improperly", "overlooks"
        ]
    
    def match_section_type(self, heading, content, brief_type):
        """
        Identify the legal section type based on heading and content
        
        Args:
            heading: Section heading text
            content: Section content text
            brief_type: 'moving' or 'response'
            
        Returns:
            dict: Matched section types with confidence scores
        """
        heading_lower = heading.lower()
        content_lower = content.lower()
        combined_text = f"{heading_lower} {content_lower[:500]}"
        
        matches = {}
        
        # Check each pattern type
        for section_type, patterns in self.brief_section_patterns.items():
            # Check if this is a known section type based on brief type
            brief_patterns = patterns.get(brief_type, [])
            
            # Calculate match score based on pattern presence
            heading_score = 0
            content_score = 0
            
            for pattern in brief_patterns:
                # Heading matches are stronger signals
                if re.search(pattern, heading_lower):
                    heading_score = 0.8
                    break
                # Content matches within first few sentences
                elif re.search(pattern, combined_text):
                    content_score = 0.4
            
            # Combine scores
            if heading_score > 0 or content_score > 0:
                matches[section_type] = heading_score + content_score
        
        return matches
    
    def extract_section_number(self, heading):
        """Extract and normalize section numbering"""
        for number_type, pattern in self.numbering_patterns.items():
            match = re.search(pattern, heading)
            if match:
                value = match.group(1)
                
                # Convert to normalized integer value
                if number_type == "roman":
                    # Simple Roman numeral conversion
                    roman_values = {'I': 1, 'V': 5, 'X': 10}
                    arabic = 0
                    prev_value = 0
                    
                    for char in reversed(value):
                        current_value = roman_values.get(char, 0)
                        if current_value >= prev_value:
                            arabic += current_value
                        else:
                            arabic -= current_value
                        prev_value = current_value
                    
                    return arabic
                
                elif number_type == "arabic":
                    return int(value)
                
                elif number_type == "alphabetic":
                    # A=1, B=2, etc.
                    return ord(value) - ord('A') + 1
                
                elif number_type == "parenthetical":
                    return int(value)
        
        return None
    
    def detect_counter_argument_language(self, text):
        """
        Detect linguistic markers of counter-argumentation
        
        Returns:
            float: Score representing strength of counter-argument signals
        """
        text_lower = text.lower()
        
        # Count marker occurrences
        marker_count = sum(1 for marker in self.counter_argument_markers 
                           if marker in text_lower)
        
        # Look for direct references to opposing arguments
        direct_references = len(re.findall(r"plaintiff('s)?\s+(claim|argu|assert|contend)",
                                           text_lower))
        
        # Calculate weighted score
        score = min(1.0, (marker_count * 0.1) + (direct_references * 0.2))
        
        return score
    
    def match_section_pairs(self, moving_heading, moving_content, response_heading, response_content):
        """
        Determine if a moving brief section and response brief section form a matching pair
        
        Returns:
            tuple: (match_score, match_type, explanation)
        """
        # Get section types
        moving_types = self.match_section_type(moving_heading, moving_content, "moving")
        response_types = self.match_section_type(response_heading, response_content, "response")
        
        # Extract section numbers
        moving_number = self.extract_section_number(moving_heading)
        response_number = self.extract_section_number(response_heading)
        
        # Check for counter-argument language
        counter_arg_score = self.detect_counter_argument_language(response_content)
        
        # Calculate match scores
        best_score = 0
        best_type = None
        explanation = ""
        
        # Section type matching (strongest signal)
        for m_type, m_score in moving_types.items():
            for r_type, r_score in response_types.items():
                if m_type == r_type:
                    score = (m_score + r_score) / 2
                    if score > best_score:
                        best_score = score
                        best_type = m_type
        
        # Section numbering matching (medium signal)
        if moving_number is not None and response_number is not None:
            if moving_number == response_number:
                numbering_score = 0.3
                best_score = max(best_score, numbering_score)
                if best_type is None:
                    best_type = "position_match"
        
        # Counter-argument language (weaker signal)
        if counter_arg_score > 0:
            best_score = max(best_score, counter_arg_score * 0.6)
            if best_type is None and counter_arg_score > 0.5:
                best_type = "linguistic_match"
        
        # Generate explanation
        if best_type == "likelihood_of_success":
            explanation = "Both arguments address likelihood of success on the merits, with opposing conclusions"
        elif best_type == "irreparable_harm":
            explanation = "Both arguments discuss irreparable harm standard, with moving brief claiming harm and response brief disputing it"
        elif best_type == "balance_of_harms":
            explanation = "Both arguments analyze balance of harms/equities, reaching different conclusions"
        elif best_type == "public_interest":
            explanation = "Both arguments address whether an injunction would serve the public interest"
        elif best_type == "takings_clause":
            explanation = "Both arguments discuss takings clause and just compensation requirements"
        elif best_type == "property_rights":
            explanation = "Both arguments address property rights with opposing interpretations"
        elif best_type == "position_match":
            explanation = "Arguments occupy corresponding positions in brief structure"
        elif best_type == "linguistic_match":
            explanation = "Response directly counters moving brief argument with opposing language"
        else:
            explanation = "Arguments show related legal reasoning"
        
        return (best_score, best_type, explanation)