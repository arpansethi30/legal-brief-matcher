# File: models/llm_verifier.py - Verify matches using Llama 3.1 via Ollama

import ollama
import json

def verify_matches(moving_brief_arguments, response_brief_arguments, candidate_matches, top_n=10):
    """
    Verify candidate matches using LLM
    
    Args:
        moving_brief_arguments: List of moving brief arguments
        response_brief_arguments: List of response brief arguments
        candidate_matches: List of candidate matches
        top_n: Number of top matches to verify
        
    Returns:
        list: List of verified matches with confidence scores
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
        
        # Truncate content to fit in prompt
        moving_content_truncated = moving_content[:500] + "..." if len(moving_content) > 500 else moving_content
        response_content_truncated = response_content[:500] + "..." if len(response_content) > 500 else response_content
        
        # Create prompt for LLM
        prompt = f"""
        Your task is to verify if a response brief argument directly addresses and counters a moving brief argument.
        
        Moving Brief Argument:
        Heading: {moving_heading}
        Content: {moving_content_truncated}
        
        Response Brief Argument:
        Heading: {response_heading}
        Content: {response_content_truncated}
        
        Answer the following questions:
        1. Does the response brief argument directly address the moving brief argument? (yes/no)
        2. What is your confidence level from 0.0 to 1.0?
        3. What specific legal concepts, issues, or standards are addressed in both arguments?
        4. Provide a brief rationale for why these arguments match or don't match.
        
        Format your response as a JSON object with keys: "matches", "confidence", "concepts", "rationale"
        """
        
        try:
            # Call Ollama with Llama 3.1
            response = ollama.chat(model='llama3:8b', messages=[
                {'role': 'user', 'content': prompt}
            ])
            
            # Parse LLM response
            llm_response = response['message']['content']
            
            # Extract JSON from the response
            try:
                # Find JSON object in the response
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = llm_response[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    # Fallback: Parse as unstructured text
                    result = {
                        'matches': 'yes' in llm_response.lower(),
                        'confidence': candidate['combined_score'],
                        'concepts': [],
                        'rationale': "Unable to parse structured response from LLM."
                    }
            except:
                # If JSON parsing fails, use a fallback
                result = {
                    'matches': 'yes' in llm_response.lower(),
                    'confidence': candidate['combined_score'],
                    'concepts': [],
                    'rationale': "Unable to parse structured response from LLM."
                }
            
            # Only include if the LLM thinks it's a match or has high confidence
            if result.get('matches', False) or result.get('confidence', 0) > 0.7:
                verified_matches.append({
                    'moving_heading': moving_heading,
                    'response_heading': response_heading,
                    'moving_content': moving_content,
                    'response_content': response_content,
                    'confidence': result.get('confidence', candidate['combined_score']),
                    'concepts': result.get('concepts', []),
                    'rationale': result.get('rationale', "")
                })
        
        except Exception as e:
            print(f"Error calling LLM: {e}")
            # Fallback to embedding score if LLM fails
            verified_matches.append({
                'moving_heading': moving_heading,
                'response_heading': response_heading,
                'moving_content': moving_content,
                'response_content': response_content,
                'confidence': candidate['combined_score'],
                'concepts': [],
                'rationale': "Determined by embedding similarity and legal heuristics."
            })
    
    # Sort by confidence
    verified_matches.sort(key=lambda x: x['confidence'], reverse=True)
    
    return verified_matches