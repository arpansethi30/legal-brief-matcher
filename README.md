# Legal Brief Argument Matcher

A machine learning application that matches arguments from moving legal briefs with corresponding counter-arguments from response briefs using NLP techniques and LLM verification.

## Overview

This application was developed for the Bloomberg Hackathon Challenge. It analyzes pairs of legal briefs (moving and response) and identifies which arguments in the response brief are addressing specific arguments in the moving brief.

## Features

- Upload JSON files containing pairs of legal briefs
- Analyze and match arguments using semantic similarity
- Verify matches using LLM reasoning
- Display matched arguments with confidence scores and rationales
- Interactive Streamlit web interface

## Technical Implementation

The solution employs a hybrid approach:
1. **Embedding-based similarity**: Arguments are converted to vector embeddings using legal-domain optimized models
2. **LLM verification**: Potential matches are verified using an LLM to ensure semantic validity
3. **Confidence scoring**: Each match is assigned a confidence score based on multiple factors

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/legal-brief-matcher.git
cd legal-brief-matcher
```

2. Create and activate a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the dependencies:
```
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```
streamlit run app.py
```

2. Upload a JSON file containing brief pairs
3. Select a brief pair to analyze
4. Click "Match Arguments" to process the briefs
5. Review the matched arguments and their confidence scores

## Project Structure

- `app.py`: Main Streamlit application
- `main.py`: Core processing logic
- `models/`: Embedding and similarity models
- `utils/`: Helper functions and data processing
- `data/`: Example data and test files

## Dependencies

- streamlit
- pandas
- numpy
- sentence-transformers
- huggingface-hub
- transformers
- torch
- scikit-learn

## License

[Your chosen license]

## Contributors

[Your name/team]
