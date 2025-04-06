#!/bin/bash

# Set the working directory
cd "$(dirname "$0")" || { echo "Failed to change directory"; exit 1; }

# Ensure all resources are cleaned up on exit
cleanup() {
    echo "Cleaning up processes..."
    
    # Find and kill streamlit processes started by this script
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        pkill -f "streamlit run app.py" || true
    else
        # Linux
        pkill -P $$ || true
    fi
    
    echo "Cleanup complete"
    exit 0
}

# Set up trap for all signals that should cause termination
trap cleanup SIGINT SIGTERM EXIT

# Set resource limits for better performance
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2  # Limit OpenMP threads
export MKL_NUM_THREADS=2  # Limit MKL threads 
export TOKENIZERS_PARALLELISM=false  # Disable tokenizers parallelism

# Disable PyTorch CUDA to use CPU only
export CUDA_VISIBLE_DEVICES=""

# Run the app
echo "Starting Legal Brief Matcher Application..."

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Error: Streamlit is not installed. Please install with: pip install streamlit"
    exit 1
fi

# Run streamlit with error handling
streamlit run app.py

# Wait for any background processes
wait

echo "Application has exited" 