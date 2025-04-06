# File: models/similarity.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from models.legal_transformer import LegalTransformer
from models.legal_patterns import LegalArgumentStructure
from models.legal_extractor import LegalFeatureExtractor

class LegalBriefMatcher:
    """State-of-the-art legal brief argument matcher"""
    
    def __init__(self, use_ollama=False):
        self.legal_transformer = LegalTransformer()
        self.pattern_recognizer = LegalArgumentStructure()
        self.feature_extractor = LegalFeatureExtractor()
        self.use_ollama = use_ollama
        
        # Try to import ollama if requested
        self.ollama_available = False
        if use_ollama:
            try:
                import ollama
                self.ollama = ollama
                self.ollama_available = True
                print("Ollama integration enabled")
            except:
                print("Ollama not available - falling back to embedding-only mode")
    
    def encode_arguments(self, arguments):
        """Encode arguments into embeddings"""
        texts = [f"{arg['heading']} {arg['content']}" for arg in arguments]
        return self.legal_transformer.encode(texts)
    
    def calculate_similarity_matrix(self, moving_embeddings, response_embeddings):
        """Calculate semantic similarity matrix"""
        return cosine_similarity(moving_embeddings, response_embeddings)
    
    def analyze_brief_pair(self, brief_pair):
        """Analyze a brief pair to extract legal insights"""
        moving_arguments = brief_pair["moving_brief"]["brief_arguments"]
        response_arguments = brief_pair["response_brief"]["brief_arguments"]
        
        # Step 1: Generate embeddings
        moving_embeddings = self.encode_arguments(moving_arguments)
        response_embeddings = self.encode_arguments(response_arguments)
        
        # Step 2: Calculate base similarity matrix
        similarity_matrix = self.calculate_similarity_matrix(moving_embeddings, response_embeddings)
        
        # Step 3: Extract legal features and pattern matches
        matches = []
        for i, moving_arg in enumerate(moving_arguments):
            for j, response_arg in enumerate(response_arguments):
                # Base similarity score
                similarity = similarity_matrix[i][j]
                
                # Check for pattern match
                pattern_score, pattern_type = self.pattern_recognizer.detect_argument_counter_pair(
                    moving_arg["heading"], moving_arg["content"],
                    response_arg["heading"], response_arg["content"]
                )
                
                # Extract legal features
                legal_comparison = self.feature_extractor.compare_arguments(
                    moving_arg, response_arg
                )
                
                # Calculate component scores
                component_scores = {
                    "semantic_similarity": similarity,
                    "pattern_match": pattern_score,
                    "citation_overlap": legal_comparison["citation_overlap"],
                    "term_overlap": legal_comparison["term_overlap"],
                    "counter_argument_score": legal_comparison["counter_argument_score"]
                }
                
                # Calculate weighted score
                weights = {
                    "semantic_similarity": 0.3,
                    "pattern_match": 0.3,
                    "citation_overlap": 0.2,
                    "term_overlap": 0.1,
                    "counter_argument_score": 0.1
                }
                
                weighted_score = sum(score * weights[key] for key, score in component_scores.items())
                
                # Store match with details
                match = {
                    "moving_index": i,
                    "response_index": j,
                    "moving_heading": moving_arg["heading"],
                    "response_heading": response_arg["heading"],
                    "moving_content": moving_arg["content"],
                    "response_content": response_arg["content"],
                    "similarity": similarity,
                    "pattern_score": pattern_score,
                    "pattern_type": pattern_type,
                    "component_scores": component_scores,
                    "weighted_score": weighted_score,
                    "legal_comparison": legal_comparison
                }
                
                # Use Ollama for verification if available
                if self.ollama_available:
                    try:
                        verified = self._verify_with_ollama(moving_arg, response_arg)
                        match["ollama_verification"] = verified
                        
                        # Boost score if LLM verifies it's a match
                        if verified.get("is_match", False):
                            match["weighted_score"] += 0.1
                        
                        # Add LLM rationale
                        match["rationale"] = verified.get("rationale", "")
                        
                    except Exception as e:
                        print(f"Error using Ollama: {e}")
                
                matches.append(match)
        
        # Sort by weighted score
        matches.sort(key=lambda x: x["weighted_score"], reverse=True)
        
        return matches
    
    def _verify_with_ollama(self, moving_arg, response_arg):
        """Use Ollama LLM to verify match"""
        if not self.ollama_available:
            return {}
        
        # Create prompt for LLM verification
        prompt = f"""
        Analyze these legal argument excerpts and determine if they form a matching argument/counter-argument pair:
        
        Moving Brief Heading: {moving_arg['heading']}
        Moving Brief Excerpt: {moving_arg['content'][:500]}...
        
        Response Brief Heading: {response_arg['heading']}
        Response Brief Excerpt: {response_arg['content'][:500]}...
        
        Does the response brief argument directly counter the moving brief argument? 
        Provide a JSON response with fields:
        - is_match (boolean)
        - confidence (float 0-1)
        - rationale (brief explanation)
        """
        
        try:
            # Call Ollama
            response = self.ollama.chat(model='llama3:8b', messages=[
                {'role': 'user', 'content': prompt}
            ])
            
            # Extract JSON response
            llm_response = response['message']['content']
            
            # Try to parse JSON from response
            try:
                # Look for JSON in response
                import re
                json_pattern = r'\{[\s\S]*\}'
                json_match = re.search(json_pattern, llm_response)
                
                if json_match:
                    result = json.loads(json_match.group(0))
                    return result
                else:
                    # Simple parsing for yes/no
                    is_match = "yes" in llm_response.lower() and not "no," in llm_response.lower()
                    return {
                        "is_match": is_match,
                        "confidence": 0.7 if is_match else 0.3,
                        "rationale": llm_response[:200]
                    }
            except:
                # Failure fallback
                return {
                    "is_match": "match" in llm_response.lower() or "counter" in llm_response.lower(),
                    "confidence": 0.5,
                    "rationale": "Determined by semantic similarity and legal patterns"
                }
                
        except Exception as e:
            print(f"Ollama error: {e}")
            return {}
    
    def select_best_matches(self, matches, moving_arguments, response_arguments):
        """Select optimal matches for the entire brief pair"""
        # Track used indices
        used_moving_indices = set()
        used_response_indices = set()
        
        final_matches = []
        
        # First pass: High-confidence pattern matches
        for match in matches:
            moving_idx = match["moving_index"]
            response_idx = match["response_index"]
            
            if (match["pattern_score"] > 0.7 and match["weighted_score"] > 0.6 and
                moving_idx not in used_moving_indices and 
                response_idx not in used_response_indices):
                
                # Format match for display
                formatted_match = self._format_match(match)
                final_matches.append(formatted_match)
                
                used_moving_indices.add(moving_idx)
                used_response_indices.add(response_idx)
        
        # Second pass: Fill in remaining moving arguments
        for moving_idx in range(len(moving_arguments)):
            if moving_idx not in used_moving_indices:
                # Find best match for this moving argument
                best_match = None
                best_score = 0.4  # Threshold for considering a match
                
                for match in matches:
                    if (match["moving_index"] == moving_idx and 
                        match["response_index"] not in used_response_indices and
                        match["weighted_score"] > best_score):
                        
                        best_match = match
                        best_score = match["weighted_score"]
                
                if best_match:
                    # Format match for display
                    formatted_match = self._format_match(best_match)
                    final_matches.append(formatted_match)
                    
                    used_moving_indices.add(moving_idx)
                    used_response_indices.add(best_match["response_index"])
        
        # Sort by moving brief order
        final_matches.sort(key=lambda x: x["moving_index"])
        
        return final_matches
    
    def _format_match(self, match):
        """Format match for display"""
        # Generate rationale if not already present
        if "rationale" not in match:
            rationale = self._generate_rationale(match)
        else:
            rationale = match["rationale"]
        
        # Format for display
        formatted = {
            "moving_index": match["moving_index"],
            "response_index": match["response_index"],
            "moving_heading": match["moving_heading"],
            "response_heading": match["response_heading"],
            "moving_content": match["moving_content"],
            "response_content": match["response_content"],
            "confidence": match["weighted_score"],
            "pattern_type": match["pattern_type"],
            "shared_citations": match["legal_comparison"]["shared_citations"],
            "shared_terms": match["legal_comparison"]["shared_terms"],
            "moving_key_sentences": match["legal_comparison"]["moving_key_sentences"],
            "response_key_sentences": match["legal_comparison"]["response_key_sentences"],
            "rationale": rationale
        }
        
        return formatted
    
    def _generate_rationale(self, match):
        """Generate a detailed rationale for why arguments match"""
        rationale_parts = []
        
        # Pattern-based rationale
        if match["pattern_score"] > 0.5:
            if match["pattern_type"] == "likelihood_success":
                rationale_parts.append("Both arguments directly address the likelihood of success standard")
            elif match["pattern_type"] == "irreparable_harm":
                rationale_parts.append("Both arguments analyze the irreparable harm requirement")
            elif match["pattern_type"] == "balance_of_equities":
                rationale_parts.append("Both arguments weigh the balance of equities/harms")
            elif match["pattern_type"] == "public_interest":
                rationale_parts.append("Both arguments consider the public interest factor") 
            elif match["pattern_type"] and match["pattern_type"].startswith("heading_pattern"):
                rationale_parts.append("The arguments occupy corresponding positions in brief structure")
            elif match["pattern_type"] == "refutation_language":
                rationale_parts.append("The response brief directly refutes claims from the moving brief")
        
        # Citation-based rationale
        shared_citations = match["legal_comparison"]["shared_citations"]
        if shared_citations:
            if len(shared_citations) == 1:
                rationale_parts.append(f"Both cite {shared_citations[0]}")
            elif len(shared_citations) > 1:
                rationale_parts.append(f"Both cite {len(shared_citations)} shared legal authorities")
        
        # Terminology-based rationale
        shared_terms = match["legal_comparison"]["shared_terms"]
        if shared_terms:
            term_sample = shared_terms[:3]
            rationale_parts.append(f"Both use legal terminology like {', '.join(term_sample)}")
        
        # Counter-argument markers
        counter_markers = match["legal_comparison"]["counter_markers"]
        if counter_markers:
            marker_samples = [m["marker"] for m in counter_markers[:2]]
            if marker_samples:
                rationale_parts.append(f"Response contains counter-argument language: {', '.join(marker_samples)}")
        
        # Combine rationale parts
        if rationale_parts:
            return ". ".join(rationale_parts)
        else:
            return "Arguments match based on semantic similarity and legal content analysis"