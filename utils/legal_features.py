# File: utils/legal_features.py - Advanced legal domain features

import re
import nltk
from nltk.tokenize import sent_tokenize
import string

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def extract_legal_features(moving_arg, response_arg):
    """
    Extract comprehensive legal-specific features from argument pairs
    """
    moving_heading = moving_arg['heading']
    moving_content = moving_arg['content']
    response_heading = response_arg['heading']
    response_content = response_arg['content']
    
    features = {}
    
    # Basic features
    features['heading_similarity'] = calculate_heading_similarity(moving_heading, response_heading)
    features['heading_pattern_match'] = detect_heading_pattern_match(moving_heading, response_heading)
    
    # Extract citations
    moving_citations = extract_citations(moving_content)
    response_citations = extract_citations(response_content)
    citation_overlap = set(moving_citations).intersection(set(response_citations))
    features['citation_overlap'] = len(citation_overlap) / max(1, len(set(moving_citations).union(set(response_citations))))
    features['has_shared_citations'] = 1.0 if citation_overlap else 0.0
    
    # Legal terminology
    moving_terms = extract_legal_terms(moving_content)
    response_terms = extract_legal_terms(response_content)
    term_overlap = set(moving_terms).intersection(set(response_terms))
    features['term_overlap'] = len(term_overlap) / max(1, len(set(moving_terms).union(set(response_terms))))
    
    # Direct reference detection
    features['direct_reference'] = detect_direct_references(moving_heading, response_content)
    features['negation_pattern'] = detect_negation_patterns(moving_content, response_content)
    
    # Advanced legal argument features
    features['standard_of_review_match'] = detect_standard_of_review(moving_content, response_content)
    features['legal_tests_match'] = detect_legal_tests(moving_content, response_content)
    features['procedural_posture_match'] = detect_procedural_posture(moving_content, response_content)
    
    # Argumentation structure features
    features['counter_argument_markers'] = detect_counter_argument_markers(response_content)
    features['response_to_specific_points'] = detect_response_to_specific_points(moving_content, response_content)
    
    # Legal brief structure patterns
    features['common_brief_section_match'] = detect_common_brief_section(moving_heading, response_heading)
    
    return features

def calculate_heading_similarity(moving_heading, response_heading):
    """
    Calculate similarity between argument headings based on legal patterns
    """
    # Convert to lowercase
    moving_lower = moving_heading.lower()
    response_lower = response_heading.lower()
    
    # Common legal heading patterns
    patterns = [
        (r'likelihood of success', r'not likely to succeed|success on.{1,20}merits'),
        (r'(irreparable|imminent) harm', r'(no|not).{1,20}(irreparable|imminent) harm'),
        (r'balance of (the )?harms', r'balance of (the )?(equities|hardships)'),
        (r'public interest', r'public interest'),
        (r'background|introduction', r'(introduction|background|injunctive relief)'),
        (r'preliminary (statement|injunction)', r'standard of (review|relief)'),
        (r'standing', r'(no )?standing'),
        (r'jurisdiction', r'(lack of )?jurisdiction')
    ]
    
    for moving_pattern, response_pattern in patterns:
        if re.search(moving_pattern, moving_lower) and re.search(response_pattern, response_lower):
            return 1.0
    
    return 0.0

def detect_heading_pattern_match(moving_heading, response_heading):
    """
    Detect if headings match typical legal brief numbering patterns
    """
    # Roman numerals to Arabic numbers mapping
    roman_pattern = re.compile(r'^([IVX]+)\.')
    arabic_pattern = re.compile(r'^(\d+)\.')
    letter_pattern = re.compile(r'^([A-Z])\.')
    
    # Extract patterns
    roman_match = roman_pattern.search(moving_heading)
    arabic_match = arabic_pattern.search(response_heading)
    letter_match = letter_pattern.search(response_heading)
    
    # Roman numeral to Arabic mapping (simplified)
    roman_to_arabic = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5}
    
    if roman_match and arabic_match:
        roman_num = roman_match.group(1)
        arabic_num = int(arabic_match.group(1))
        
        # If roman numeral maps to current arabic number, it's likely a match
        if roman_num in roman_to_arabic and roman_to_arabic[roman_num] == arabic_num:
            return 0.9
    
    if roman_match and letter_match:
        # Often section I matches with section A, II with B, etc.
        roman_num = roman_match.group(1)
        letter = letter_match.group(1)
        
        # Map I->A, II->B, etc.
        if roman_num == 'I' and letter == 'A':
            return 0.8
        elif roman_num == 'II' and letter == 'B':
            return 0.8
        elif roman_num == 'III' and letter == 'C':
            return 0.8
        elif roman_num == 'IV' and letter == 'D':
            return 0.8
    
    return 0.0

def extract_citations(text):
    """
    Extract legal citations from text using refined regex patterns
    """
    # Case citation pattern (e.g. "Smith v. Jones, 123 F.3d 456 (9th Cir. 1999)")
    case_pattern = r'[A-Za-z\s\.\,\']+\sv\.\s[A-Za-z\s\.\,\']+,\s\d+\s[A-Za-z\.]+\s\d+\s*\(\w+\.?\s*\d{4}\)'
    
    # Statutory citation pattern (e.g. "42 U.S.C. § 1983")
    statute_pattern = r'\d+\s[A-Za-z\.]+\s§\s*\d+[a-z0-9\-]*'
    
    # Extract all citations
    case_citations = re.findall(case_pattern, text)
    statute_citations = re.findall(statute_pattern, text)
    
    # Clean up and normalize
    all_citations = [c.strip() for c in case_citations + statute_citations]
    return all_citations

def extract_legal_terms(text):
    """
    Extract comprehensive legal terminology from text
    """
    # List of legal terminology commonly used in briefs
    legal_terms = [
        "injunction", "irreparable harm", "balance of equities", "public interest", 
        "likelihood of success", "merits", "trespass", "negligence", "nuisance",
        "property rights", "easement", "taking", "just compensation", "strict liability",
        "compensatory damages", "punitive damages", "standard of review", "relief",
        "condemnation", "due process", "fifth amendment", "inverse condemnation",
        "constitutional", "statute", "regulatory", "jurisdiction", "civil procedure",
        "rule 65", "preliminary injunction", "temporary restraining order",
        "federal question", "diversity jurisdiction", "summary judgment", "dismissal",
        "motion to dismiss", "pleading standard", "standing", "ripeness", "mootness",
        "class action", "certification", "reasonable doubt", "preponderance", "clear and convincing",
        "discretion", "abuse of discretion", "de novo", "clearly erroneous", "judgment as a matter of law",
        "prima facie", "res judicata", "collateral estoppel", "stare decisis", "persuasive authority",
        "binding precedent", "on all fours", "distinguishable", "overruled", "circuit split"
    ]
    
    # Find terms in text
    found_terms = []
    text_lower = text.lower()
    for term in legal_terms:
        if term in text_lower:
            found_terms.append(term)
    
    return found_terms

def detect_direct_references(moving_heading, response_content):
    """
    Detect direct references to the moving brief argument in the response
    """
    # Extract keywords from heading
    keywords = moving_heading.lower().split()
    keywords = [k for k in keywords if len(k) > 3 and k not in {'and', 'the', 'of', 'for', 'in', 'on', 'to', 'with'}]
    
    # Response text in lowercase
    response_lower = response_content.lower()
    
    # Search for reference phrases
    reference_phrases = [
        'plaintiff argues', 'plaintiffs argue', 'plaintiff contends', 'plaintiffs contend',
        'defendant argues', 'defendants argue', 'as stated by', 'according to', 
        'claims that', 'asserts that', 'alleges that', 'plaintiff\'s argument', 
        'contrary to', 'unlike', 'opposing', 'in contrast', 'however', 'nevertheless',
        'plaintiff\'s position', 'plaintiff\'s contention', 'mischaracterizes',
        'plaintiff mistakenly', 'plaintiff incorrectly', 'plaintiff fails to'
    ]
    
    # Check each phrase
    for phrase in reference_phrases:
        if phrase in response_lower:
            # Find the sentence containing the reference
            sentences = sent_tokenize(response_lower)
            for sentence in sentences:
                if phrase in sentence:
                    # Check if any keyword is in the same sentence
                    if any(keyword in sentence for keyword in keywords):
                        return 1.0
    
    # If we found phrases but no keywords, give partial score
    for phrase in reference_phrases:
        if phrase in response_lower:
            return 0.7
    
    # Keywords matching
    keyword_matches = sum(1 for keyword in keywords if keyword in response_lower)
    if len(keywords) > 0:
        return min(0.5, keyword_matches / len(keywords))
    
    return 0.0

def detect_negation_patterns(moving_content, response_content):
    """
    Detect negation patterns between moving and response arguments
    """
    # Extract key statements from moving brief
    moving_sentences = sent_tokenize(moving_content.lower())
    
    # Look for negation of key statements in response
    negation_markers = ['not', 'no', 'never', 'fails', 'incorrect', 'wrong', 'mischaracterizes', 
                        'contrary', 'however', 'mistaken', 'error', 'erroneous', 'misplaced',
                        'without merit', 'lacks', 'groundless', 'baseless', 'unfounded']
    
    # Count negations
    negation_count = 0
    response_lower = response_content.lower()
    
    # Check for general negations
    for marker in negation_markers:
        if marker in response_lower:
            negation_count += 1
    
    # Check for more specific negations of key phrases
    key_phrases = extract_key_phrases(moving_content)
    for phrase in key_phrases:
        phrase_lower = phrase.lower()
        if any(neg + " " + phrase_lower in response_lower for neg in ['not', 'no', 'never']):
            negation_count += 2  # Stronger signal when directly negating a key phrase
    
    # Normalize score
    return min(1.0, negation_count / 8)

def extract_key_phrases(text, max_phrases=5):
    """
    Extract key phrases from text based on legal terminology and frequency
    """
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    # Extract phrases that contain legal terminology
    legal_terms = set([
        "irreparable harm", "balance of equities", "public interest", "likelihood of success",
        "property rights", "just compensation", "fifth amendment", "preliminary injunction"
    ])
    
    phrases = []
    for sentence in sentences:
        for term in legal_terms:
            if term in sentence.lower():
                # Find a reasonable phrase fragment
                parts = sentence.split(', ')
                for part in parts:
                    if term in part.lower() and 3 < len(part.split()) < 12:
                        phrases.append(part.strip())
    
    # If we don't have enough, add sentences with citations
    if len(phrases) < max_phrases:
        for sentence in sentences:
            if re.search(r'\d+\s[A-Za-z\.]+\s\d+', sentence):  # Basic citation pattern
                if len(sentence.split()) < 15:
                    phrases.append(sentence.strip())
    
    # Deduplicate and limit
    unique_phrases = list(set(phrases))
    return unique_phrases[:max_phrases]

def detect_standard_of_review(moving_content, response_content):
    """
    Detect if both briefs address the same standard of review
    """
    standards = [
        "de novo", "clearly erroneous", "abuse of discretion", "substantial evidence",
        "preponderance", "clear and convincing", "beyond a reasonable doubt",
        "arbitrary and capricious", "rational basis"
    ]
    
    moving_lower = moving_content.lower()
    response_lower = response_content.lower()
    
    for standard in standards:
        if standard in moving_lower and standard in response_lower:
            return 1.0
    
    # Check for standard of review section
    if "standard of review" in moving_lower and "standard of review" in response_lower:
        return 0.8
    
    return 0.0

def detect_legal_tests(moving_content, response_content):
    """
    Detect if both briefs discuss the same legal tests
    """
    common_tests = [
        "four[ -]factor", "four[ -]part", "three[ -]factor", "three[ -]part",
        "winter test", "preliminary injunction factors", "likelihood of success",
        "irreparable harm", "balance of (the )?(equities|hardships)", "public interest",
        "strict scrutiny", "intermediate scrutiny", "rational basis"
    ]
    
    moving_lower = moving_content.lower()
    response_lower = response_content.lower()
    
    for test in common_tests:
        if re.search(test, moving_lower) and re.search(test, response_lower):
            return 1.0
    
    return 0.0

def detect_procedural_posture(moving_content, response_content):
    """
    Detect if both briefs address the same procedural posture
    """
    postures = [
        "motion to dismiss", "motion for summary judgment", "preliminary injunction",
        "temporary restraining order", "motion for class certification",
        "motion to compel", "motion to suppress", "motion in limine"
    ]
    
    moving_lower = moving_content.lower()
    response_lower = response_content.lower()
    
    for posture in postures:
        if posture in moving_lower and posture in response_lower:
            return 1.0
    
    return 0.0

def detect_counter_argument_markers(response_content):
    """
    Detect linguistic markers of counter-argumentation
    """
    counter_markers = [
        "however", "nevertheless", "nonetheless", "on the contrary", "in contrast",
        "rather", "instead", "but", "yet", "although", "though", "despite",
        "plaintiff fails to", "plaintiff mistakenly", "plaintiff incorrectly",
        "plaintiff overlooks", "plaintiff misunderstands", "plaintiff ignores",
        "plaintiff's argument", "plaintiff's position", "plaintiff's contention",
        "plaintiff's claim", "plaintiff's assertion", "plaintiff's theory"
    ]
    
    response_lower = response_content.lower()
    
    # Count markers
    marker_count = sum(1 for marker in counter_markers if marker in response_lower)
    
    # Normalize score
    return min(1.0, marker_count / 5)

def detect_response_to_specific_points(moving_content, response_content):
    """
    Detect if the response addresses specific points from the moving brief
    """
    # Extract key nouns and proper nouns from moving brief
    moving_lower = moving_content.lower()
    response_lower = response_content.lower()
    
    # Look for specific entities and terminology
    entities = set()
    
    # Add case names
    case_pattern = r'([A-Z][a-z]+)\s+v\.\s+([A-Z][a-z]+)'
    for match in re.finditer(case_pattern, moving_content):
        entities.add(match.group(1).lower())
        entities.add(match.group(2).lower())
    
    # Add statute names/numbers
    statute_pattern = r'(\d+\s+[A-Za-z\.]+\s+§\s*\d+)'
    for match in re.finditer(statute_pattern, moving_content):
        entities.add(match.group(1).lower())
    
    # Add proper nouns (simplified)
    proper_noun_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
    for match in re.finditer(proper_noun_pattern, moving_content):
        # Filter out common words that might be capitalized
        name = match.group(1)
        if len(name) > 3 and name not in ["The", "This", "That", "These", "Those", "There"]:
            entities.add(name.lower())
    
    # Count matched entities
    matched_entities = sum(1 for entity in entities if entity in response_lower)
    
    if entities:
        return min(1.0, matched_entities / min(len(entities), 10))
    return 0.0

def detect_common_brief_section(moving_heading, response_heading):
    """
    Detect if headings represent common matching sections in legal briefs
    """
    # Common section pairs in moving/response briefs
    section_pairs = [
        (r'likelihood of success', r'(not likely to succeed|no success on.{1,20}merits)'),
        (r'irreparable harm', r'no irreparable harm'),
        (r'balance of (the )?harm', r'balance of (the )?(equities|hardships)'),
        (r'public interest', r'public interest'),
        (r'jurisdiction', r'(lack of )?jurisdiction'),
        (r'standing', r'(no )?standing'),
        (r'background', r'(background|statement of facts)'),
        (r'statement of facts', r'(response to )?statement of facts')
    ]
    
    moving_lower = moving_heading.lower()
    response_lower = response_heading.lower()
    
    for moving_pattern, response_pattern in section_pairs:
        if re.search(moving_pattern, moving_lower) and re.search(response_pattern, response_lower):
            return 1.0
    
    return 0.0