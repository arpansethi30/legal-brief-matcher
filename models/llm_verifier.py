# File: models/llm_verifier.py - Improved LLM-based verification with legal reasoning

import ollama
import json
import re
from collections import Counter

# Define legal reasoning templates for different argument types
LEGAL_REASONING_TEMPLATES = {
    'injunction': """
    For injunction arguments, focus on these standard factors:
    1. Likelihood of success on the merits
    2. Irreparable harm
    3. Balance of equities/hardships
    4. Public interest
    
    Identify how the response brief directly counters the moving brief's arguments on each factor.
    """,
    
    'statutory': """
    For statutory interpretation arguments, analyze how each brief addresses:
    1. Plain meaning of statutory text
    2. Legislative intent
    3. Statutory context and structure
    4. Precedent applying the statute
    
    Determine if the response challenges the interpretation, application, or jurisdiction.
    """,
    
    'constitutional': """
    For constitutional arguments, evaluate how each brief addresses:
    1. Constitutional text
    2. Original meaning/intent
    3. Supreme Court precedent
    4. Application to factual scenario
    
    Identify if the response contests the constitutional framework, application, or relevance.
    """,
    
    'contract': """
    For contract disputes, analyze how each brief addresses:
    1. Contract formation and validity
    2. Contract terms interpretation
    3. Performance/breach
    4. Damages or specific performance
    
    Determine how the response challenges elements of formation, interpretation, breach, or remedies.
    """,
    
    'property': """
    For property disputes, evaluate how each brief addresses:
    1. Nature of property interest
    2. Evidence of ownership/rights
    3. Alleged infringement
    4. Damages or injunctive relief
    
    Identify how the response contests ownership, infringement, or remedies.
    """
}

def detect_legal_argument_type(text):
    """
    Detect the type of legal argument based on text content
    """
    text_lower = text.lower()
    
    # Check for injunction-related terms
    injunction_terms = ['injunction', 'restraining order', 'likelihood of success', 'irreparable harm', 
                        'balance of equities', 'public interest', 'preliminary injunction']
    
    # Check for statutory terms
    statutory_terms = ['statute', 'statutory', 'legislature', 'congress', 'legislative history',
                      'plain meaning', 'legislative intent', 'textual', 'section', 'provision']
    
    # Check for constitutional terms
    constitutional_terms = ['constitution', 'constitutional', 'first amendment', 'fourth amendment',
                          'due process', 'equal protection', 'fundamental right', 'amendment']
    
    # Check for contract terms
    contract_terms = ['contract', 'agreement', 'breach', 'consideration', 'offer', 'acceptance',
                     'damages', 'specific performance', 'term', 'provision']
    
    # Check for property terms
    property_terms = ['property', 'ownership', 'deed', 'title', 'easement', 'trespass',
                     'nuisance', 'covenant', 'fee simple', 'real property', 'possession']
    
    # Count term occurrences
    injunction_count = sum(1 for term in injunction_terms if term in text_lower)
    statutory_count = sum(1 for term in statutory_terms if term in text_lower)
    constitutional_count = sum(1 for term in constitutional_terms if term in text_lower)
    contract_count = sum(1 for term in contract_terms if term in text_lower)
    property_count = sum(1 for term in property_terms if term in text_lower)
    
    # Determine dominant type
    counts = {
        'injunction': injunction_count,
        'statutory': statutory_count,
        'constitutional': constitutional_count,
        'contract': contract_count,
        'property': property_count
    }
    
    # Get the type with highest count, defaulting to 'general' if all are zero
    max_count = max(counts.values())
    if max_count == 0:
        return 'general'
    
    # Return the dominant type
    return max(counts, key=counts.get)

def verify_matches(moving_brief_arguments, response_brief_arguments, candidate_matches, top_n=10):
    """
    Verify candidate matches using advanced LLM prompting with legal reasoning
    """
    verified_matches = []
    
    # Only process top N candidates to save time
    candidates_to_verify = candidate_matches[:top_n]
    
    for candidate in candidates_to_verify:
        # Extract arguments
        moving_heading = candidate['moving_heading']
        moving_content = candidate['moving_content']
        response_heading = candidate['response_heading']
        response_content = candidate['response_content']
        
        # Extract legal citations for comparison
        moving_citations = extract_citations(moving_content)
        response_citations = extract_citations(response_content)
        shared_citations = set(moving_citations).intersection(set(response_citations))
        
        # Extract key legal terms
        moving_terms = extract_legal_terms(moving_content)
        response_terms = extract_legal_terms(response_content)
        shared_terms = set(moving_terms).intersection(set(response_terms))
        
        # Detect legal argument type
        moving_type = detect_legal_argument_type(moving_content)
        response_type = detect_legal_argument_type(response_content)
        
        # Get appropriate legal reasoning template
        if moving_type == response_type and moving_type in LEGAL_REASONING_TEMPLATES:
            reasoning_template = LEGAL_REASONING_TEMPLATES[moving_type]
        else:
            # Default template
            reasoning_template = "Analyze how the response brief directly counters or addresses the legal and factual arguments made in the moving brief."
        
        # Truncate content to fit in prompt
        moving_content_truncated = moving_content[:800] + "..." if len(moving_content) > 800 else moving_content
        response_content_truncated = response_content[:800] + "..." if len(response_content) > 800 else response_content
        
        # Create advanced prompt for LLM with legal framework
        prompt = f"""
        You are a legal expert specializing in analyzing legal arguments. Your task is to verify if a response brief argument directly addresses and counters a moving brief argument. Be specific about legal standards, tests, doctrines and principles.

        Moving Brief Argument:
        Heading: {moving_heading}
        Content: {moving_content_truncated}

        Response Brief Argument:
        Heading: {response_heading}
        Content: {response_content_truncated}

        Detected Argument Type: {moving_type.capitalize()}
        
        Shared Legal Citations: {', '.join(shared_citations) if shared_citations else 'None'}
        Shared Legal Terms: {', '.join(list(shared_terms)[:5]) if shared_terms else 'None'}

        {reasoning_template}

        Answer the following questions:
        1. Does the response brief argument directly address the moving brief argument? (yes/no)
        2. What is your confidence level from 0.0 to 1.0?
        3. What specific legal standards or tests are being addressed in both arguments?
        4. Identify the core legal dispute between these arguments in one sentence.
        5. Explain precisely how the response counters the moving brief's reasoning.
        6. Rate the strength of the legal counterargument on a scale of 0-10.

        Format your response as a JSON object with keys: "matches", "confidence", "legal_standards", "core_dispute", "counter_reasoning", "strength"
        """
        
        try:
            # Call Ollama with Llama 3.1
            response = ollama.chat(model='llama3:latest', messages=[
                {'role': 'system', 'content': 'You are a legal expert assistant specializing in argument analysis.'},
                {'role': 'user', 'content': prompt}
            ])
            
            # Parse LLM response
            llm_response = response['message']['content']
            
            # Extract JSON from the response using regex to be more robust
            json_pattern = r'\{[\s\S]*\}'
            json_match = re.search(json_pattern, llm_response)
            
            if json_match:
                try:
                    result = json.loads(json_match.group())
                except:
                    # Handle malformed JSON by extracting key fields
                    result = extract_fields_from_text(llm_response)
            else:
                # Fallback extraction if JSON format fails
                result = extract_fields_from_text(llm_response)
            
            # Construct comprehensive rationale
            rationale = construct_rationale(result, shared_citations, shared_terms)
            
            # Only include if the LLM thinks it's a match or has high confidence
            if result.get('matches', False) or result.get('confidence', 0) > 0.65:
                verified_matches.append({
                    'moving_heading': moving_heading,
                    'response_heading': response_heading,
                    'moving_content': moving_content,
                    'response_content': response_content,
                    'confidence': result.get('confidence', candidate['enhanced_similarity']),
                    'legal_standards': result.get('legal_standards', []),
                    'core_dispute': result.get('core_dispute', ""),
                    'counter_strength': result.get('strength', 5),
                    'argument_type': moving_type,
                    'shared_citations': list(shared_citations),
                    'shared_terms': list(shared_terms)[:5],
                    'rationale': rationale
                })
        
        except Exception as e:
            print(f"Error calling LLM: {e}")
            # Fallback with more context
            verified_matches.append({
                'moving_heading': moving_heading,
                'response_heading': response_heading,
                'moving_content': moving_content,
                'response_content': response_content,
                'confidence': candidate['enhanced_similarity'],
                'argument_type': moving_type,
                'shared_citations': list(shared_citations),
                'shared_terms': list(shared_terms)[:5],
                'rationale': f"Both arguments address {moving_heading.lower().replace('i.', '').replace('ii.', '').replace('iii.', '').replace('iv.', '').strip()} with opposing viewpoints. They share {len(shared_citations)} citations and {len(shared_terms)} legal terms."
            })
    
    # Sort by confidence
    verified_matches.sort(key=lambda x: x['confidence'], reverse=True)
    
    return verified_matches

def extract_citations(text):
    """Extract legal citations from text using comprehensive regex patterns"""
    # Case citation pattern (e.g. "Smith v. Jones, 123 F.3d 456 (9th Cir. 1999)")
    case_pattern = r'[A-Za-z\s\.\,\']+\sv\.\s[A-Za-z\s\.\,\']+,\s\d+\s[A-Za-z\.]+\s\d+\s*\(\w+\.?\s*\d{4}\)'
    
    # Shorter case citation pattern (e.g. "Smith, 123 F.3d at 460")
    short_case_pattern = r'[A-Za-z\s\.\,\']+,\s\d+\s[A-Za-z\.]+\s(at|at\s)\d+'
    
    # Statutory citation pattern (e.g. "42 U.S.C. § 1983")
    statute_pattern = r'\d+\s[A-Za-z\.]+\s§\s*\d+[a-z0-9\-]*'
    
    # Regulatory citation pattern (e.g. "17 C.F.R. § 240.10b-5")
    regulation_pattern = r'\d+\s[A-Za-z\.]+\s§\s*\d+\.\d+[a-z0-9\-]*'
    
    # Constitutional citation pattern (e.g. "U.S. Const. amend. XIV, § 1")
    constitution_pattern = r'U\.S\.\s+Const\.\s+[aA]mend\.\s+[IVX]+,\s+§\s*\d+'
    
    # Extract all citations
    case_citations = re.findall(case_pattern, text)
    short_citations = re.findall(short_case_pattern, text)
    statute_citations = re.findall(statute_pattern, text)
    regulation_citations = re.findall(regulation_pattern, text)
    constitution_citations = re.findall(constitution_pattern, text)
    
    # Clean up and normalize
    all_citations = [c.strip() for c in case_citations + short_citations + statute_citations + regulation_citations + constitution_citations]
    return all_citations

def extract_legal_terms(text):
    """Extract common legal terminology from text"""
    # Expanded list of common legal terms to look for
    legal_terms = [
        # Injunction terms
        "injunction", "irreparable harm", "balance of equities", "public interest", 
        "likelihood of success", "merits", "temporary restraining order", "preliminary injunction",
        
        # Property terms
        "trespass", "nuisance", "property rights", "easement", "taking", "just compensation", 
        "fee simple", "quiet title", "adverse possession", "eminent domain",
        
        # Constitutional terms
        "first amendment", "fourth amendment", "fifth amendment", "fourteenth amendment",
        "due process", "equal protection", "constitutional", "fundamental right",
        
        # Contract terms
        "contract", "consideration", "offer", "acceptance", "breach", "specific performance",
        "liquidated damages", "material breach", "covenant", "warranty",
        
        # Tort terms
        "negligence", "strict liability", "proximate cause", "damages", "duty of care",
        "standard of care", "reasonable person", "battery", "assault", "intentional",
        
        # Civil procedure
        "jurisdiction", "civil procedure", "summary judgment", "motion to dismiss",
        "rule 12(b)(6)", "rule 56", "class action", "joinder", "discovery",
        
        # Evidence
        "hearsay", "admissible", "relevance", "prejudicial", "testimony", "expert witness",
        
        # Remedies
        "compensatory damages", "punitive damages", "restitution", "specific performance",
        "injunctive relief", "equitable relief", "legal remedies", "laches", "mootness",
        
        # Administrative law
        "arbitrary and capricious", "agency action", "rulemaking", "substantial evidence",
        "administrative procedure act", "chevron deference", "regulatory"
    ]
    
    # Find terms in text
    found_terms = []
    text_lower = text.lower()
    for term in legal_terms:
        if term in text_lower:
            found_terms.append(term)
    
    return found_terms

def extract_fields_from_text(text):
    """Extract key fields from LLM text response when JSON parsing fails"""
    result = {}
    
    # Check for matches
    result["matches"] = "yes" in text.lower() and not "no, the response" in text.lower()
    
    # Extract confidence
    confidence_match = re.search(r'confidence.*?(\d+\.\d+)', text, re.IGNORECASE)
    if confidence_match:
        result["confidence"] = float(confidence_match.group(1))
    else:
        result["confidence"] = 0.7  # Default
    
    # Extract legal standards
    standards_pattern = r'legal standards.*?:(.*?)(?:\n|$)'
    standards_match = re.search(standards_pattern, text, re.IGNORECASE | re.DOTALL)
    if standards_match:
        result["legal_standards"] = standards_match.group(1).strip()
    else:
        result["legal_standards"] = ""
    
    # Extract core dispute
    dispute_pattern = r'core dispute.*?:(.*?)(?:\n|$)'
    dispute_match = re.search(dispute_pattern, text, re.IGNORECASE | re.DOTALL)
    if dispute_match:
        result["core_dispute"] = dispute_match.group(1).strip()
    else:
        result["core_dispute"] = ""
    
    # Extract counter reasoning
    reasoning_pattern = r'counter.*?reasoning.*?:(.*?)(?:\n|$)'
    reasoning_match = re.search(reasoning_pattern, text, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        result["counter_reasoning"] = reasoning_match.group(1).strip()
    else:
        result["counter_reasoning"] = ""
    
    # Extract strength rating
    strength_pattern = r'strength.*?:?\s*(\d+)'
    strength_match = re.search(strength_pattern, text, re.IGNORECASE)
    if strength_match:
        result["strength"] = int(strength_match.group(1))
    else:
        result["strength"] = 5  # Default mid-range
    
    return result

def construct_rationale(result, shared_citations, shared_terms):
    """Construct a comprehensive rationale from LLM output and extracted features"""
    rationale_parts = []
    
    # Add core dispute if available
    if result.get("core_dispute"):
        rationale_parts.append(result["core_dispute"])
    
    # Add counter reasoning if available
    if result.get("counter_reasoning"):
        rationale_parts.append(result["counter_reasoning"])
    
    # Add legal standards if available
    if result.get("legal_standards"):
        rationale_parts.append(f"Both arguments address the legal standard(s): {result['legal_standards']}")
    
    # Add strength assessment if available
    if result.get("strength") is not None:
        strength_value = result["strength"]
        if strength_value >= 8:
            strength_desc = "The response provides an exceptionally strong counter to the moving brief's argument"
        elif strength_value >= 6:
            strength_desc = "The response effectively counters the moving brief's argument"
        elif strength_value >= 4:
            strength_desc = "The response adequately addresses the moving brief's argument"
        else:
            strength_desc = "The response provides a limited counter to the moving brief's argument"
        rationale_parts.append(strength_desc)
    
    # Add citation information
    if shared_citations:
        citation_info = f"They cite {len(shared_citations)} of the same legal authorities"
        if len(shared_citations) <= 2:
            citation_info += f": {', '.join(shared_citations)}"
        rationale_parts.append(citation_info + ".")
    
    # Combine all parts
    if rationale_parts:
        return " ".join(rationale_parts)
    else:
        return "Arguments match based on addressing the same legal standards with opposing conclusions."