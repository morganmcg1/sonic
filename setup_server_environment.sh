#!/bin/bash

# Setup script for server-based LLM benchmarking
# Sets up both vLLM and MAX for server-based comparison

set -e

echo "Setting up server-based LLM benchmarking environment..."

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable not set."
    echo "You'll need to set it for accessing gated models like Llama-3-8B-Instruct:"
    echo "export HF_TOKEN=\"your_token_here\""
fi

# Create/activate virtual environment for vLLM
if [ ! -d "llm_benchmark_server" ]; then
    echo "Creating virtual environment for vLLM server..."
    uv venv llm_benchmark_server
fi

echo "Activating virtual environment and installing vLLM dependencies..."
source llm_benchmark_server/bin/activate

# Install vLLM with async support
uv pip install vllm[async]
uv pip install aiohttp
uv pip install psutil
uv pip install requests

echo "vLLM environment setup complete!"

# Check if MAX/pixi environment exists
if [ ! -d "max_test" ]; then
    echo "Creating MAX environment..."
    mkdir -p max_test
    cd max_test
    
    # Initialize pixi project for MAX
    pixi init .
    pixi add max>=24.5
    
    cd ..
else
    echo "MAX environment already exists"
fi

echo "Testing MAX installation..."
cd max_test
if pixi run max --version; then
    echo "MAX installation verified!"
else
    echo "MAX installation failed or not working"
    exit 1
fi
cd ..

echo ""
echo "Setup complete! To use the server-based benchmark:"
echo ""
echo "1. Set your Hugging Face token (if not already set):"
echo "   export HF_TOKEN=\"your_token_here\""
echo ""
echo "2. Start servers manually:"
echo "   # Terminal 1 - vLLM server"
echo "   source llm_benchmark_server/bin/activate"
echo "   python -m vllm.entrypoints.openai.api_server \\"
echo "     --model meta-llama/Meta-Llama-3-8B-Instruct \\"
echo "     --port 8001 \\"
echo "     --trust-remote-code \\"
echo "     --dtype bfloat16"
echo ""
echo "   # Terminal 2 - MAX server"
echo "   cd max_test"
echo "   pixi run max serve --model-path=meta-llama/Meta-Llama-3-8B-Instruct --port=8002"
echo ""
echo "3. Run benchmark:"
echo "   source llm_benchmark_server/bin/activate"
echo "   python benchmark_servers.py --quick"
echo ""
echo "Or use automatic server management:"
echo "   python benchmark_servers.py --quick --start-servers"
echo ""