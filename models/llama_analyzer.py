import os
import json
import re
from typing import Dict, Any, List, Tuple

# Import for Ollama client
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Ollama package not available. Counter-analysis quality will be disabled.")

class LlamaLegalAnalyzer:
    """Advanced legal reasoning analyzer using Llama 3.1 through Ollama"""
    
    def __init__(self):
        """Initialize Llama 3.1 model for legal analysis using Ollama"""
        self.is_available = False
        self.model_name = "llama3.1:latest"
        
        # Check if Ollama is available at all
        if not OLLAMA_AVAILABLE:
            print("Ollama integration disabled: package not installed")
            return
            
        try:
            # Try to use the Ollama API to check if the model exists
            print(f"Checking for Ollama model: {self.model_name}")
            ollama.list()  # Just check if Ollama is running
            self.is_available = True
            print(f"Ollama integration enabled with model: {self.model_name}")
        except Exception as e:
            print(f"Ollama service not available: {str(e)}")
            print("Counter-argument quality analysis will use fallback methods")
    
    def analyze_counter_argument(self, moving_arg: Dict[str, Any], response_arg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the quality of a counter-argument in relation to a moving argument
        
        Args:
            moving_arg: Dictionary containing moving brief argument
            response_arg: Dictionary containing response brief argument
            
        Returns:
            Dictionary with analysis results including quality score and reasoning
        """
        if not self.is_available:
            return self._fallback_analysis(moving_arg, response_arg)
        
        # Extract key information
        moving_heading = moving_arg.get('heading', '')
        moving_content = moving_arg.get('content', '')[:1000]  # Reduce content length to prevent timeouts
        response_heading = response_arg.get('heading', '')
        response_content = response_arg.get('content', '')[:1000]
        
        # Construct a prompt that elicits structured analysis
        prompt = f"""You are an expert legal analyst evaluating counter-arguments in legal briefs.
        
MOVING BRIEF ARGUMENT:
Heading: {moving_heading}
Content: {moving_content}

RESPONSE BRIEF ARGUMENT:
Heading: {response_heading}
Content: {response_content}

Analyze how effectively the response brief counters the moving brief argument. Provide:
1. A numerical score from 1-10 on the quality of the counter-argument
2. A brief analysis of the reasoning used (max 3 sentences)
3. Three key strengths of the counter-argument
4. Three key weaknesses or missed opportunities in the counter-argument

Format your response as a JSON object with the following fields:
{{
  "counter_quality_score": <score>,
  "reasoning": "<reasoning>",
  "strengths": ["<strength1>", "<strength2>", "<strength3>"],
  "weaknesses": ["<weakness1>", "<weakness2>", "<weakness3>"]
}}
"""
        
        # Generate response from Llama via Ollama
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.1,  # Low temperature for deterministic output
                    "top_p": 0.9,
                    "num_predict": 512   # Reduce token count to prevent timeouts
                }
            )
            
            # Extract the generated text
            generated_text = response['response']
            
            # Parse the JSON response
            try:
                # Find JSON object in the response
                json_match = re.search(r'({[\s\S]*})', generated_text)
                if json_match:
                    json_str = json_match.group(1)
                    result = json.loads(json_str)
                    
                    # Ensure we have all expected fields
                    result["counter_quality_score"] = float(result.get("counter_quality_score", 0)) / 10.0  # Normalize to 0-1
                    if "reasoning" not in result:
                        result["reasoning"] = "Analysis not available"
                    if "strengths" not in result or not result["strengths"]:
                        result["strengths"] = ["No specific strengths identified"]
                    if "weaknesses" not in result or not result["weaknesses"]:
                        result["weaknesses"] = ["No specific weaknesses identified"]
                    
                    return result
                else:
                    # Fallback if JSON parsing fails
                    return self._extract_manual(generated_text)
            except Exception as e:
                print(f"Failed to parse Llama output: {str(e)}")
                # Fallback extraction
                return self._extract_manual(generated_text)
        except Exception as e:
            print(f"Error generating response from Ollama: {str(e)}")
            return self._fallback_analysis(moving_arg, response_arg)
    
    def _extract_manual(self, text: str) -> Dict[str, Any]:
        """Extract data manually if JSON parsing fails"""
        # Extract score
        score_match = re.search(r'counter.quality.score[\"\':\s]+(\d+\.?\d*)', text, re.IGNORECASE)
        score = float(score_match.group(1)) / 10.0 if score_match else 0.5
        
        # Extract reasoning
        reasoning_match = re.search(r'reasoning[\"\':\s]+(.*?)[\"\',}]', text, re.IGNORECASE)
        reasoning = reasoning_match.group(1) if reasoning_match else "Analysis not available"
        
        # Extract strengths
        strengths = re.findall(r'strength\d*[\"\':\s]+(.*?)[\"\',}]', text, re.IGNORECASE)
        strengths = strengths[:3] if strengths else ["No specific strengths identified"]
        
        # Extract weaknesses
        weaknesses = re.findall(r'weakness\d*[\"\':\s]+(.*?)[\"\',}]', text, re.IGNORECASE)
        weaknesses = weaknesses[:3] if weaknesses else ["No specific weaknesses identified"]
        
        return {
            "counter_quality_score": score,
            "reasoning": reasoning,
            "strengths": strengths,
            "weaknesses": weaknesses
        }
        
    def _fallback_analysis(self, moving_arg: Dict[str, Any], response_arg: Dict[str, Any]) -> Dict[str, Any]:
        """Provide a rule-based fallback analysis when Llama is unavailable"""
        # Extract heading and content
        moving_heading = moving_arg.get('heading', '')
        moving_content = moving_arg.get('content', '')
        response_heading = response_arg.get('heading', '')
        response_content = response_arg.get('content', '')
        
        # Simple heuristics for counter-argument quality
        
        # 1. Check if response is significantly shorter (might indicate insufficient response)
        length_ratio = len(response_content) / max(1, len(moving_content))
        
        # 2. Check for negation terms that might indicate counter-arguments
        negation_terms = ['not', 'no', 'never', 'disagree', 'contrary', 'however', 'nevertheless', 
                         'although', 'though', 'but', 'despite', 'notwithstanding']
        negation_count = sum(1 for term in negation_terms if term in response_content.lower())
        
        # 3. Check for citation indicators
        citation_indicators = ['v.', 'see', 'cf.', 'accord', 'supra', 'infra', 'id.', 'ibid.']
        citation_count = sum(1 for term in citation_indicators if term in response_content)
        
        # Calculate a simple quality score based on heuristics
        length_score = min(1.0, length_ratio)
        negation_score = min(1.0, negation_count / 5.0)
        citation_score = min(1.0, citation_count / 3.0)
        
        counter_quality = (length_score * 0.3) + (negation_score * 0.4) + (citation_score * 0.3)
        
        # Generate explanation
        reasoning = "Analysis performed using rule-based heuristics since Llama model is unavailable."
        
        # Generate strengths and weaknesses based on heuristics
        strengths = []
        weaknesses = []
        
        if length_ratio > 0.8:
            strengths.append("Response provides substantial content addressing the argument")
        else:
            weaknesses.append("Response may be too brief relative to the original argument")
            
        if negation_count > 2:
            strengths.append("Contains multiple counter-point indicators")
        else:
            weaknesses.append("Limited use of terms indicating disagreement or counter-points")
            
        if citation_count > 1:
            strengths.append("Includes legal citations to support counter-argument")
        else:
            weaknesses.append("Could be strengthened with more legal authorities")
            
        # Ensure we have three of each
        while len(strengths) < 3:
            strengths.append("Additional analysis requires LLM capabilities")
            
        while len(weaknesses) < 3:
            weaknesses.append("Additional analysis requires LLM capabilities")
        
        return {
            "counter_quality_score": counter_quality,
            "reasoning": reasoning,
            "strengths": strengths[:3],
            "weaknesses": weaknesses[:3]
        } 