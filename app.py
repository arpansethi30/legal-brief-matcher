# File: app.py - Main application

import streamlit as st
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gc
import atexit

from models.legal_matcher import LegalArgumentMatcher
from components.visualization import LegalNetworkVisualizer
from utils.data_pipeline import LegalDataProcessor

# Clean shutdown handling
def clean_exit():
    """Clean up resources before exit"""
    print("Cleaning up resources before exit...")
    # Clear PyTorch cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Force garbage collection
        gc.collect()
    except Exception as e:
        print(f"Cleanup error: {e}")

# Register cleanup handler
atexit.register(clean_exit)

# Set page config with lower resource usage
st.set_page_config(
    page_title="Legal Brief Matcher",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
.legal-header {
    font-family: 'Georgia', serif;
    color: #2C3E50;
    padding-bottom: 10px;
    border-bottom: 2px solid #E74C3C;
    margin-bottom: 20px;
}
.match-box {
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 15px;
}
.moving-brief {
    background-color: #EBF5FB;
    border-left: 5px solid #3498DB;
    color: #2C3E50;  /* Dark blue-gray text color for better readability */
}
.response-brief {
    background-color: #FDEDEC;
    border-left: 5px solid #E74C3C;
    color: #2C3E50;  /* Dark blue-gray text color for better readability */
}
.confidence-high {
    color: #27AE60;
    font-weight: bold;
}
.confidence-medium {
    color: #F39C12;
    font-weight: bold;
}
.confidence-low {
    color: #C0392B;
    font-weight: bold;
}
.citation {
    background-color: #EBF5FB;
    padding: 2px 4px;
    border-radius: 3px;
    font-family: monospace;
    font-size: 0.9em;
}
.metrics-container {
    padding: 15px;
    background-color: #F8F9F9;
    border-radius: 5px;
    margin-top: 15px;
}
.metrics-header {
    font-weight: bold;
    color: #2C3E50;
    margin-bottom: 10px;
}
.stats-card {
    padding: 20px;
    border-radius: 5px;
    background-color: #fff;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin-bottom: 10px;
}
.stats-number {
    font-size: 24px;
    font-weight: bold;
    color: #3498DB;
}
.stats-title {
    font-size: 14px;
    color: #7F8C8D;
}
.match-detail-container {
    padding: 20px;
    border-radius: 5px;
    background-color: #f9f9f9;
    margin-bottom: 20px;
    border: 1px solid #eaeaea;
}
.tab-content {
    padding: 15px 0;
}
.confidence-breakdown {
    padding: 10px;
    background-color: #fff;
    border-radius: 5px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.footer {
    margin-top: 30px;
    padding-top: 10px;
    border-top: 1px solid #BDC3C7;
    font-size: 0.8em;
    color: #7F8C8D;
}
</style>
""", unsafe_allow_html=True)

# Initialize components with error handling
@st.cache_resource
def load_components():
    """Load and cache core components"""
    try:
        print("Initializing Legal Brief Matcher components...")
        matcher = LegalArgumentMatcher()
        visualizer = LegalNetworkVisualizer()
        processor = LegalDataProcessor()
        return matcher, visualizer, processor
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        # Create fallback components with minimal functionality
        print(f"Using fallback components due to error: {str(e)}")
        fallback_matcher = FallbackMatcher()
        visualizer = LegalNetworkVisualizer()
        processor = LegalDataProcessor()
        return fallback_matcher, visualizer, processor

# Handle PyTorch errors preemptively
try:
    # Use CPU only to avoid CUDA issues
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
except:
    pass

# Load components
matcher, visualizer, processor = load_components()

# Function to display matches with detailed explanations for judges
def display_matches_with_explanation(formatted_matches):
    """Display the match results in a visually appealing way with judge explanations"""
    if not formatted_matches:
        st.warning("No matches found. Try adjusting your parameters.")
        return
    
    # Sort by confidence
    sorted_matches = sorted(formatted_matches, key=lambda x: x['confidence'], reverse=True)
    
    for i, match in enumerate(sorted_matches):
        # Get explanations
        explanation = match.get('explanation', '')
        
        # Get confidence class
        confidence = match['confidence']
        if confidence >= 0.7:
            confidence_class = "confidence-high"
        elif confidence >= 0.5:
            confidence_class = "confidence-medium"
        else:
            confidence_class = "confidence-low"
        
        # Display match header with confidence
        st.markdown(f"""
        <div class="match-detail-container">
            <h3>Match {i+1}: <span class='{confidence_class}'>{confidence:.2f} confidence</span></h3>
            <p><em>{explanation}</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs
        tabs = st.tabs(["Side by Side", "Details", "Confidence Analysis", "Explanation for Judges"])
        
        # Side by Side tab
        with tabs[0]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="match-box moving-brief">
                    <h4>{match['moving_heading']}</h4>
                    <div>{match['moving_content'][:500] + "..." if len(match['moving_content']) > 500 else match['moving_content']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="match-box response-brief">
                    <h4>{match['response_heading']}</h4>
                    <div>{match['response_content'][:500] + "..." if len(match['response_content']) > 500 else match['response_content']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Details tab
        with tabs[1]:
            if match.get('shared_citations'):
                st.subheader("Shared Citations")
                for citation in match['shared_citations']:
                    st.markdown(f"- {citation}")
            
            if match.get('shared_terms'):
                st.subheader("Shared Legal Terms")
                st.write(", ".join(match['shared_terms']))
        
        # Confidence Analysis tab
        with tabs[2]:
            if 'confidence_factors' in match:
                confidence_fig = visualizer.create_confidence_breakdown(match)
                st.pyplot(confidence_fig)
        
        # Explanation for Judges tab
        with tabs[3]:
            if 'confidence_factors' in match:
                factors = match['confidence_factors']
                
                # Get final score explanation
                final_score = factors.get('final_score', {})
                if isinstance(final_score, dict) and 'explanation' in final_score:
                    final_score_value = final_score.get('value', 0)
                    final_score_explanation = final_score.get('explanation', '')
                    st.markdown(f"### Overall Match Score: {final_score_value:.3f}")
                    st.markdown(f"**{final_score_explanation}**")
                else:
                    st.markdown(f"### Overall Match Score: {factors.get('final_score', 0):.3f}")
                
                st.markdown("### Detailed Factor Breakdown:")
                
                # Create two columns for the factors
                col1, col2 = st.columns(2)
                
                # List of factors to display with their display names and column
                factors_list = [
                    ('base_similarity', 'Base Similarity', col1),
                    ('citation_boost', 'Citation Boost', col1),
                    ('heading_match', 'Heading Match', col1),
                    ('legal_terminology', 'Legal Terminology', col2),
                    ('pattern_match', 'Argument Pattern Match', col2),
                    ('length_penalty', 'Length Disparity Penalty', col2),
                    ('precedent_impact', 'Precedent Impact', col1)
                ]
                
                # Display each factor with its explanation
                for key, label, col in factors_list:
                    if key in factors:
                        factor = factors[key]
                        if isinstance(factor, dict) and 'value' in factor:
                            with col:
                                st.markdown(f"**{label}**: {factor['value']:.3f}")
                                if 'explanation' in factor:
                                    st.markdown(f"<div style='padding-left:20px; color:#666; font-size:0.9em'>{factor['explanation']}</div>", unsafe_allow_html=True)
                        else:
                            with col:
                                st.markdown(f"**{label}**: {factor:.3f}")
                
                # Provide overall recommendation
                st.markdown("### Recommendation for Judges:")
                if confidence > 0.7:
                    st.markdown("🟢 **Strong match** - High confidence that these arguments are directly addressing each other")
                elif confidence > 0.5:
                    st.markdown("🟡 **Moderate match** - These arguments likely address similar points but may have differences in approach")
                else:
                    st.markdown("🔴 **Weak match** - Some similarities exist, but these may be addressing different aspects")
        
        st.markdown("---")

# Main application header
st.markdown("<h1 class='legal-header'>Legal Brief Matcher:<br>Argument-Counter Argument Analyzer</h1>", unsafe_allow_html=True)

# File uploader section
st.write("Upload a test brief pair JSON file or use the example data.")
uploaded_file = st.file_uploader("Upload JSON file", type=["json"])

# Example data toggle
use_example = st.checkbox("Use example data", value=True)

# Main processing logic
if uploaded_file is not None or use_example:
    with st.spinner("Loading brief data..."):
        # Load data with error handling
        try:
            if uploaded_file is not None:
                # Save uploaded file
                with open("temp_upload.json", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                brief_pairs = processor.load_brief_pair("temp_upload.json")
            else:
                # Use example data
                example_path = "stanford_hackathon_brief_pairs.json"
                if os.path.exists(example_path):
                    brief_pairs = processor.load_brief_pair(example_path)
                else:
                    st.error(f"Example file not found: {example_path}")
                    brief_pairs = []
            
            if not brief_pairs:
                st.error("Error loading brief data. Please check the file format.")
            else:
                # Select brief pair if multiple
                if len(brief_pairs) > 1:
                    st.subheader("Select Brief Pair")
                    brief_options = [f"Brief {i+1}: {pair['moving_brief']['brief_id']} vs {pair['response_brief']['brief_id']}" 
                                   for i, pair in enumerate(brief_pairs)]
                    selected_index = st.selectbox("Choose a brief pair to analyze:", 
                                                range(len(brief_options)), 
                                                format_func=lambda i: brief_options[i])
                    brief_pair = brief_pairs[selected_index]
                else:
                    brief_pair = brief_pairs[0]
                
                # Preprocess brief pair
                processed_pair = processor.preprocess_brief_pair(brief_pair)
                
                # Display brief info and statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="match-box moving-brief">
                        <h3>Moving Brief</h3>
                        <p><strong>ID:</strong> {processed_pair['moving_brief']['brief_id']}</p>
                        <p><strong>Arguments:</strong> {len(processed_pair['moving_brief']['brief_arguments'])}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show arguments in expander
                    with st.expander("View Moving Brief Arguments"):
                        for i, arg in enumerate(processed_pair['moving_brief']['brief_arguments']):
                            st.write(f"**{i+1}. {arg['heading']}**")
                            st.write(arg['content'][:200] + "...")
                
                with col2:
                    st.markdown(f"""
                    <div class="match-box response-brief">
                        <h3>Response Brief</h3>
                        <p><strong>ID:</strong> {processed_pair['response_brief']['brief_id']}</p>
                        <p><strong>Arguments:</strong> {len(processed_pair['response_brief']['brief_arguments'])}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show arguments in expander
                    with st.expander("View Response Brief Arguments"):
                        for i, arg in enumerate(processed_pair['response_brief']['brief_arguments']):
                            st.write(f"**{i+1}. {arg['heading']}**")
                            st.write(arg['content'][:200] + "...")
                
                # Add brief pair statistics
                if 'metadata' in processed_pair:
                    st.subheader("Brief Pair Analysis")
                    meta = processed_pair['metadata']
                    
                    # Display statistics in a grid
                    stat_cols = st.columns(4)
                    
                    with stat_cols[0]:
                        st.markdown("""
                        <div class="stats-card">
                            <div class="stats-number">{}</div>
                            <div class="stats-title">Moving Arguments</div>
                        </div>
                        """.format(meta['moving_arg_count']), unsafe_allow_html=True)
                    
                    with stat_cols[1]:
                        st.markdown("""
                        <div class="stats-card">
                            <div class="stats-number">{}</div>
                            <div class="stats-title">Response Arguments</div>
                        </div>
                        """.format(meta['response_arg_count']), unsafe_allow_html=True)
                    
                    with stat_cols[2]:
                        shared_citation_count = len(meta.get('shared_citations', []))
                        st.markdown("""
                        <div class="stats-card">
                            <div class="stats-number">{}</div>
                            <div class="stats-title">Shared Citations</div>
                        </div>
                        """.format(shared_citation_count), unsafe_allow_html=True)
                    
                    with stat_cols[3]:
                        heading_match_count = len(meta.get('heading_matches', []))
                        st.markdown("""
                        <div class="stats-card">
                            <div class="stats-number">{}</div>
                            <div class="stats-title">Potential Heading Matches</div>
                        </div>
                        """.format(heading_match_count), unsafe_allow_html=True)
                
                # Match arguments button
                if st.button("Match Arguments", key="match_button"):
                    with st.spinner("Analyzing legal arguments and generating matches..."):
                        # Analyze brief pair
                        comparisons = matcher.analyze_brief_pair(brief_pair)
                        
                        # Generate optimal matches
                        moving_count = len(brief_pair['moving_brief']['brief_arguments'])
                        response_count = len(brief_pair['response_brief']['brief_arguments'])
                        
                        matches = matcher.generate_optimal_matches(comparisons, moving_count, response_count)
                        
                        # Format matches
                        formatted_matches = matcher.format_matches(matches)
                        
                        # Display network visualization
                        st.subheader("Argument Network Visualization")
                        network_fig = visualizer.create_network_visualization(formatted_matches, brief_pair)
                        st.pyplot(network_fig)
                        
                        # Display interactive table
                        st.subheader("Match Summary")
                        match_table = visualizer.create_interactive_table(formatted_matches)
                        st.write(match_table.to_html(escape=False), unsafe_allow_html=True)
                        
                        # Add detailed metrics about the match quality
                        match_metrics = {
                            'high_confidence': sum(1 for m in formatted_matches if m['confidence'] >= 0.7),
                            'medium_confidence': sum(1 for m in formatted_matches if 0.5 <= m['confidence'] < 0.7),
                            'low_confidence': sum(1 for m in formatted_matches if m['confidence'] < 0.5),
                            'matched_moving': len(set(m['moving_heading'] for m in formatted_matches)),
                            'matched_response': len(set(m['response_heading'] for m in formatted_matches))
                        }
                        
                        # Display match metrics
                        st.markdown("<h3>Match Quality Metrics</h3>", unsafe_allow_html=True)
                        metric_cols = st.columns(5)
                        
                        with metric_cols[0]:
                            st.metric("Total Matches", len(formatted_matches))
                        
                        with metric_cols[1]:
                            st.metric("High Confidence", match_metrics['high_confidence'])
                        
                        with metric_cols[2]:
                            st.metric("Medium Confidence", match_metrics['medium_confidence'])
                        
                        with metric_cols[3]:
                            st.metric("Low Confidence", match_metrics['low_confidence'])
                        
                        with metric_cols[4]:
                            coverage = round(match_metrics['matched_moving'] / moving_count * 100)
                            st.metric("Moving Brief Coverage", f"{coverage}%")
                        
                        # Display detailed matches
                        st.subheader("Detailed Matches")
                        display_matches_with_explanation(formatted_matches)
                        
                        # Export results
                        export_data = processor.format_results_for_export(formatted_matches)
                        export_json = json.dumps(export_data, indent=2)
                        
                        st.subheader("Export Results")
                        st.download_button(
                            label="Download Matches (JSON)",
                            data=export_json,
                            file_name="brief_argument_matches.json",
                            mime="application/json"
                        )
                        
                        # Display metrics if ground truth available
                        if 'true_links' in processed_pair:
                            metrics = processor.evaluate_matches(export_data, processed_pair['true_links'])
                            
                            st.markdown("""
                            <div class="metrics-container">
                                <div class="metrics-header">Evaluation Metrics</div>
                            """, unsafe_allow_html=True)
                            
                            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                            
                            with metrics_col1:
                                st.metric("Precision", f"{metrics['precision']:.2f}")
                            
                            with metrics_col2:
                                st.metric("Recall", f"{metrics['recall']:.2f}")
                            
                            with metrics_col3:
                                st.metric("F1 Score", f"{metrics['f1']:.2f}")
                            
                            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing briefs: {str(e)}")
            brief_pairs = []

else:
    st.info("Please upload a test brief pair JSON file or use the example data.")

# Add sidebar with information
st.sidebar.title("Legal Brief Matcher")
st.sidebar.subheader("Argument-Counter Argument Analyzer")

st.sidebar.markdown("""
### Features
- **Legal-Domain Transformer:** Specialized for legal text analysis
- **Advanced Citation Analysis:** Identifies shared legal authorities 
- **Argument Pattern Recognition:** Detects standard legal argument structures
- **Enhanced Confidence Metrics:** Detailed factor-based scoring
- **Interactive Visualization:** Explore argument relationships visually
- **Legal-Domain Reasoning:** Domain-specific verification of matches
""")

st.sidebar.markdown("""
### New Features
- **Argument Strength Analysis:** Evaluates counter-argument effectiveness
- **Detailed Confidence Breakdowns:** See what factors drive match quality
- **Citation Classification:** Categorizes and analyzes shared authorities
- **Legal Domain Detection:** Automatically identifies argument types
- **Enhanced Visualization:** Better understand argument relationships
""")

st.sidebar.markdown("""
### Implementation Details
- Hybrid approach combining:
  - Embedding-based similarity
  - Structure-based matching
  - Citation analysis
  - Legal domain knowledge
  - LLM-powered verification
""")

# Add footer
st.markdown("""
<div class="footer">
    Legal Brief Matcher - Argument-Counter Argument Analyzer
</div>
""", unsafe_allow_html=True)

# Define a fallback matcher with minimal functionality
class FallbackMatcher:
    def __init__(self):
        self.use_llama = False
        print("Using fallback matcher with reduced functionality")
    
    def analyze_brief_pair(self, brief_pair):
        """Simple fallback implementation"""
        moving_arguments = brief_pair["moving_brief"]["brief_arguments"]
        response_arguments = brief_pair["response_brief"]["brief_arguments"]
        
        # Generate simple comparisons
        comparisons = []
        for i, moving_arg in enumerate(moving_arguments):
            for j, response_arg in enumerate(response_arguments):
                # Extremely simple matching using headings
                heading_match = False
                
                # Basic content overlap (very simplified)
                words1 = set(moving_arg["content"].lower().split())
                words2 = set(response_arg["content"].lower().split())
                overlap = len(words1.intersection(words2)) / max(1, len(words1.union(words2)))
                
                comparisons.append({
                    'moving_index': i,
                    'response_index': j,
                    'moving_heading': moving_arg["heading"],
                    'response_heading': response_arg["heading"],
                    'moving_content': moving_arg["content"],
                    'response_content': response_arg["content"],
                    'similarity': overlap,
                    'legal_boost': 0.0,
                    'enhanced_similarity': overlap,
                    'confidence_factors': {
                        'base_similarity': overlap,
                        'citation_boost': 0.0,
                        'heading_match': 0.0,
                        'legal_terminology': 0.0,
                        'pattern_match': 0.0,
                        'length_penalty': 0.0,
                        'final_score': overlap
                    },
                    'heading_match': heading_match,
                    'shared_citations': [],
                    'argument_pairs': []
                })
        
        # Sort by similarity
        comparisons.sort(key=lambda x: x['enhanced_similarity'], reverse=True)
        return comparisons
    
    def generate_optimal_matches(self, comparisons, moving_count, response_count):
        """Simple greedy matching"""
        # Take top matches for each moving argument
        matches = []
        used_response = set()
        
        for i in range(moving_count):
            best_match = None
            best_score = 0.0
            
            for comp in comparisons:
                if comp['moving_index'] == i and comp['response_index'] not in used_response:
                    if comp['enhanced_similarity'] > best_score:
                        best_score = comp['enhanced_similarity']
                        best_match = comp
            
            if best_match and best_score > 0.2:
                matches.append(best_match)
                used_response.add(best_match['response_index'])
        
        return matches
    
    def format_matches(self, matches):
        """Format matches for visualization"""
        formatted_matches = []
        
        for match in matches:
            formatted_match = {
                'moving_heading': match['moving_heading'],
                'response_heading': match['response_heading'],
                'moving_content': match['moving_content'],
                'response_content': match['response_content'],
                'moving_index': match['moving_index'],
                'response_index': match['response_index'],
                'confidence': match['enhanced_similarity'],
                'confidence_factors': match.get('confidence_factors', {}),
                'explanation': "Simple text similarity match (reduced functionality mode)",
                'shared_citations': [],
                'shared_terms': []
            }
            
            formatted_matches.append(formatted_match)
        
        return formatted_matches

# Clean up when exiting
def _handle_exit():
    """Additional cleanup on exit"""
    try:
        # Remove temporary files
        if os.path.exists("temp_upload.json"):
            os.remove("temp_upload.json")
    except Exception as e:
        print(f"Cleanup error: {e}")

# Register additional cleanup
atexit.register(_handle_exit)