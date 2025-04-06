# File: app.py - Modified to handle array of brief pairs

import streamlit as st
import pandas as pd
import json
import os
from utils.data_loader import load_brief_pair
from models.embedding_model import get_embeddings
from models.similarity_model import calculate_similarity, get_top_matches
from models.llm_verifier import verify_matches

st.set_page_config(page_title="Legal Brief Argument Matcher", layout="wide")

st.title("Legal Brief Argument Matcher")
st.write("Bloomberg Hackathon Challenge - Match moving brief arguments with response brief arguments")

# File uploader for test brief pairs
uploaded_file = st.file_uploader("Upload a test brief pair JSON file", type="json")

if uploaded_file is not None:
    # Load the brief pairs list
    brief_pairs = json.load(uploaded_file)
    
    # If it's a list, let the user select which brief pair to analyze
    if isinstance(brief_pairs, list):
        st.subheader("Select a Brief Pair")
        brief_pair_ids = [f"Brief Pair {i+1}: {pair['moving_brief']['brief_id']} vs {pair['response_brief']['brief_id']}" 
                          for i, pair in enumerate(brief_pairs)]
        selected_index = st.selectbox("Choose a brief pair to analyze:", range(len(brief_pair_ids)), 
                                     format_func=lambda i: brief_pair_ids[i])
        brief_pair = brief_pairs[selected_index]
    else:
        # If it's already a single object
        brief_pair = brief_pairs
    
    # Display the brief pair info
    st.subheader("Brief Pair Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Moving Brief ID:", brief_pair["moving_brief"]["brief_id"])
        st.write("Number of Arguments:", len(brief_pair["moving_brief"]["brief_arguments"]))
    
    with col2:
        st.write("Response Brief ID:", brief_pair["response_brief"]["brief_id"])
        st.write("Number of Arguments:", len(brief_pair["response_brief"]["brief_arguments"]))
    
    # Process the brief pair
    if st.button("Match Arguments"):
        with st.spinner("Matching arguments..."):
            # Get embeddings for both briefs
            moving_brief_embeddings = get_embeddings(brief_pair["moving_brief"]["brief_arguments"])
            response_brief_embeddings = get_embeddings(brief_pair["response_brief"]["brief_arguments"])
            
            # Calculate similarity scores
            similarity_matrix = calculate_similarity(moving_brief_embeddings, response_brief_embeddings)
            
            # Get top matches
            matches = get_top_matches(
                similarity_matrix, 
                brief_pair["moving_brief"]["brief_arguments"],
                brief_pair["response_brief"]["brief_arguments"]
            )
            
            # Verify matches with LLM
            verified_matches = verify_matches(
                brief_pair["moving_brief"]["brief_arguments"],
                brief_pair["response_brief"]["brief_arguments"],
                matches
            )
            
            # Display results
            st.subheader("Matched Arguments")
            
            for i, match in enumerate(verified_matches):
                st.markdown(f"### Match {i+1}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Moving Brief Argument:**")
                    st.markdown(f"**Heading:** {match['moving_heading']}")
                    st.markdown("**Content Preview:**")
                    st.markdown(match['moving_content'][:300] + "...")
                
                with col2:
                    st.markdown("**Response Brief Argument:**")
                    st.markdown(f"**Heading:** {match['response_heading']}")
                    st.markdown("**Content Preview:**")
                    st.markdown(match['response_content'][:300] + "...")
                
                st.markdown(f"**Confidence Score:** {match['confidence']:.2f}")
                st.markdown(f"**Matching Rationale:** {match['rationale']}")
                st.markdown("---")

else:
    st.info("Please upload a test brief pair JSON file to get started.")

st.sidebar.title("About")
st.sidebar.info(
    "This application matches arguments from moving briefs with corresponding "
    "counter-arguments from response briefs using a hybrid approach with legal-domain "
    "embeddings and LLM verification."
)