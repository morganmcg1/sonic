#!/bin/bash

# Improved Benchmarking Setup Script - vLLM vs MAX with Official Tools
# Uses MAX's official benchmark_serving.py for fair comparison

set -e

echo "🚀 Setting up Improved vLLM vs MAX Benchmarking Environment"
echo "================================================================"

# Check if we're on the right platform for GPU benchmarking
if [[ "$(uname -m)" == "arm64" ]] && [[ "$(uname)" == "Darwin" ]]; then
    echo "⚠️  Warning: Detected macOS ARM64. GPU benchmarking requires NVIDIA GPUs."
    echo "This setup is intended for Linux systems with NVIDIA GPUs."
    echo "Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN environment variable not set."
    echo "Set it with: export HF_TOKEN=\"your_token_here\""
    exit 1
else
    echo "✅ HF_TOKEN is set"
fi

# Check GPU availability (if on Linux)
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  NVIDIA GPU not detected or drivers not installed"
fi

# Setup vLLM environment with optimal configuration
echo ""
echo "🔧 Setting up vLLM environment with optimal performance flags..."

if [ ! -d "vllm_optimized_env" ]; then
    echo "Creating optimized vLLM virtual environment..."
    python -m venv vllm_optimized_env
fi

# Activate vLLM environment
source vllm_optimized_env/bin/activate

# Install optimized vLLM with all performance dependencies
echo "Installing vLLM with performance optimizations..."
pip install --upgrade pip
pip install wheel setuptools
pip install vllm[async]
pip install aiohttp requests psutil
pip install transformers tokenizers
pip install huggingface_hub
pip install tqdm numpy

# Install Flash Attention for optimal performance (if on CUDA)
if command -v nvidia-smi &> /dev/null; then
    echo "Installing Flash Attention for GPU optimization..."
    pip install flash-attn --no-build-isolation || echo "⚠️  Flash Attention installation failed, using fallback"
fi

echo "✅ vLLM environment setup complete!"
deactivate

# Setup MAX environment
echo ""
echo "🔧 Setting up MAX environment..."

# Check if pixi is installed
if ! command -v pixi &> /dev/null; then
    echo "Installing pixi package manager..."
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
fi

# Create proper MAX environment for the target platform
if [ ! -d "max_optimized" ]; then
    echo "Creating optimized MAX environment..."
    mkdir -p max_optimized
    cd max_optimized
    
    # Initialize with proper platform detection
    if [[ "$(uname)" == "Darwin" ]]; then
        if [[ "$(uname -m)" == "arm64" ]]; then
            PLATFORM="osx-arm64"
        else
            PLATFORM="osx-64"
        fi
    else
        PLATFORM="linux-64"
    fi
    
    cat > pixi.toml << EOF
[project]
name = "max_optimized"
version = "0.1.0"
authors = ["benchmark-user"]
channels = ["https://conda.modular.com/max-nightly", "conda-forge"]
platforms = ["$PLATFORM"]

[dependencies]
modular = ">=25.5.0.dev2025070207,<26"
python = ">=3.9,<3.13"

[tasks]
max = "max"
serve = "max serve"
EOF

    # Install MAX
    pixi install
    cd ..
else
    echo "MAX environment already exists"
fi

# Test MAX installation
echo "🧪 Testing MAX installation..."
cd max_optimized
if pixi run max --version; then
    echo "✅ MAX installation verified!"
else
    echo "❌ MAX installation failed"
    cd ..
    exit 1
fi
cd ..

# Create optimized server startup scripts
echo ""
echo "📝 Creating optimized server startup scripts..."

# vLLM startup script with optimal configuration
cat > start_vllm_optimized.sh << 'EOF'
#!/bin/bash

# Optimized vLLM Server Startup
# Based on vLLM benchmarking best practices

set -e

echo "🚀 Starting Optimized vLLM Server"
echo "=================================="

# Kill any existing vLLM processes
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

# Set optimal environment variables
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_FLASH_ATTN_FORCE_ENABLE=1
export CUDA_VISIBLE_DEVICES=0

# Activate environment
source vllm_optimized_env/bin/activate

# Start vLLM with optimal configuration
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --port 8001 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256 \
  --max-model-len 8192 \
  --disable-log-stats \
  --tensor-parallel-size 1 &

VLLM_PID=$!
echo "vLLM started with PID: $VLLM_PID"

# Wait for server to be ready
echo "⏳ Waiting for vLLM server to be ready..."
for i in {1..120}; do
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo "✅ vLLM server is ready at http://localhost:8001"
        echo "Environment variables:"
        echo "  VLLM_USE_V1=$VLLM_USE_V1"
        echo "  VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND"
        echo "  VLLM_FLASH_ATTN_FORCE_ENABLE=$VLLM_FLASH_ATTN_FORCE_ENABLE"
        exit 0
    fi
    if [ $i -eq 120 ]; then
        echo "❌ vLLM server failed to start after 2 minutes"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
    echo -n "."
done
EOF

# MAX startup script with optimal configuration
cat > start_max_optimized.sh << 'EOF'
#!/bin/bash

# Optimized MAX Server Startup
# Based on MAX official serving best practices

set -e

echo "🚀 Starting Optimized MAX Server"
echo "================================"

# Kill any existing MAX processes
pkill -f "max serve" || true
sleep 2

# Set optimal environment variables
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="${HF_TOKEN}"

if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN not set. Set it with: export HF_TOKEN=\"your_token\""
    exit 1
fi

# Start MAX server
cd max_optimized
pixi run max serve \
  --model-path=meta-llama/Meta-Llama-3-8B-Instruct \
  --port=8002 \
  --host=0.0.0.0 &

MAX_PID=$!
echo "MAX started with PID: $MAX_PID"

# Wait for server to be ready
echo "⏳ Waiting for MAX server to be ready..."
for i in {1..180}; do
    if curl -s http://localhost:8002/health > /dev/null 2>&1; then
        echo "✅ MAX server is ready at http://localhost:8002"
        echo "Environment variables:"
        echo "  HF_HUB_ENABLE_HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER"
        echo "  HF_TOKEN=***set***"
        cd ..
        exit 0
    fi
    if [ $i -eq 180 ]; then
        echo "❌ MAX server failed to start after 3 minutes"
        kill $MAX_PID 2>/dev/null || true
        cd ..
        exit 1
    fi
    sleep 1
    echo -n "."
done
EOF

# Make scripts executable
chmod +x start_vllm_optimized.sh
chmod +x start_max_optimized.sh

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Next steps:"
echo "1. Start servers in separate terminals:"
echo "   Terminal 1: ./start_vllm_optimized.sh"
echo "   Terminal 2: ./start_max_optimized.sh"
echo ""
echo "2. Run benchmarks using the official MAX benchmark script:"
echo "   # For vLLM"
echo "   python benchmark_serving_official.py --backend vllm --host localhost --port 8001 \\"
echo "     --model meta-llama/Meta-Llama-3-8B-Instruct --num-prompts 100 --dataset-name sharegpt"
echo ""
echo "   # For MAX"
echo "   python benchmark_serving_official.py --backend modular --host localhost --port 8002 \\"
echo "     --model meta-llama/Meta-Llama-3-8B-Instruct --num-prompts 100 --dataset-name sharegpt"
echo ""
echo "📁 Files created:"
echo "   - benchmark_serving_official.py (official MAX benchmark tool)"
echo "   - start_vllm_optimized.sh (optimized vLLM server)"
echo "   - start_max_optimized.sh (optimized MAX server)"
echo "   - vllm_optimized_env/ (Python environment)"
echo "   - max_optimized/ (MAX pixi environment)"