#!/bin/bash

# Setup script for vLLM vs Mojo/MAX benchmarking environment
# Supports both macOS (Apple Silicon) and Linux (GPU)

set -e

PYTHON_VERSION="3.10"
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect platform
detect_platform() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if [[ $(uname -m) == "arm64" ]]; then
            PLATFORM="macos_arm64"
            print_status "Detected: macOS Apple Silicon"
        else
            PLATFORM="macos_intel"
            print_status "Detected: macOS Intel"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        PLATFORM="linux"
        print_status "Detected: Linux"
    else
        print_error "Unsupported platform: $OSTYPE"
        exit 1
    fi
}

# Check if running in Google Cloud
check_gcp() {
    if curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/ > /dev/null 2>&1; then
        print_status "Running on Google Cloud Platform"
        GCP_INSTANCE=true
    else
        GCP_INSTANCE=false
    fi
}

# Setup Python environment
setup_python_env() {
    print_status "Setting up Python environment..."
    
    # Check if conda is available
    if command -v conda &> /dev/null; then
        print_status "Using conda for environment management"
        
        # Create conda environment
        if conda info --envs | grep -q "llm_benchmark"; then
            print_warning "Conda environment 'llm_benchmark' already exists. Activating..."
        else
            print_status "Creating conda environment with Python ${PYTHON_VERSION}..."
            conda create -n llm_benchmark python=${PYTHON_VERSION} -y
        fi
        
        # Activate environment
        eval "$(conda shell.bash hook)"
        conda activate llm_benchmark
        
    else
        print_status "Using venv for environment management"
        
        # Create virtual environment
        if [[ ! -d "venv_llm_benchmark" ]]; then
            print_status "Creating virtual environment..."
            python${PYTHON_VERSION} -m venv venv_llm_benchmark
        fi
        
        # Activate virtual environment
        source venv_llm_benchmark/bin/activate
    fi
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
}

# Install common dependencies
install_common_deps() {
    print_status "Installing common dependencies..."
    
    pip install \
        torch torchvision torchaudio \
        transformers \
        accelerate \
        datasets \
        requests \
        psutil \
        numpy \
        matplotlib \
        seaborn \
        pandas \
        tqdm
}

# Install vLLM
install_vllm() {
    print_status "Installing vLLM..."
    
    if [[ "$PLATFORM" == "macos_arm64" ]]; then
        print_warning "Installing vLLM on macOS (limited functionality, CPU-only)"
        
        # Install from source for better macOS compatibility
        if [[ ! -d "vllm" ]]; then
            print_status "Cloning vLLM repository..."
            git clone https://github.com/vllm-project/vllm.git
        fi
        
        cd vllm
        pip install -r requirements-build.txt
        pip install -e . --no-build-isolation
        cd ..
        
    elif [[ "$PLATFORM" == "linux" ]]; then
        print_status "Installing vLLM for Linux with GPU support..."
        
        # Check for CUDA
        if command -v nvidia-smi &> /dev/null; then
            print_status "NVIDIA GPU detected, installing with CUDA support"
            pip install vllm
        else
            print_warning "No NVIDIA GPU detected, installing CPU-only version"
            pip install vllm --no-deps
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
    else
        print_warning "Installing vLLM with basic support for $PLATFORM"
        pip install vllm
    fi
}

# Install Mojo/MAX
install_mojo_max() {
    print_status "Installing Mojo/Modular MAX..."
    
    if [[ "$PLATFORM" == "macos_arm64" ]]; then
        print_status "Installing MAX for macOS Apple Silicon..."
        
        # Method 1: Try pip installation first
        if pip install max-python 2>/dev/null; then
            print_status "MAX installed via pip"
        else
            print_warning "pip installation failed, trying alternative methods..."
            
            # Method 2: Install Modular CLI
            if ! command -v modular &> /dev/null; then
                print_status "Installing Modular CLI..."
                curl -s https://get.modular.com | sh -
                
                # Add to PATH
                echo 'export PATH="$HOME/.modular/bin:$PATH"' >> ~/.zshrc
                echo 'export PATH="$HOME/.modular/bin:$PATH"' >> ~/.bashrc
                export PATH="$HOME/.modular/bin:$PATH"
            fi
            
            # Authenticate and install
            print_status "Please authenticate with Modular (you may need to sign up at https://developer.modular.com/)"
            modular auth || print_warning "Authentication failed - you may need to sign up first"
            
            # Install MAX
            modular install max || print_warning "MAX installation may have failed"
        fi
        
    elif [[ "$PLATFORM" == "linux" ]]; then
        print_status "Installing MAX for Linux..."
        
        # Try pip installation first
        if pip install max-python 2>/dev/null; then
            print_status "MAX installed via pip"
        else
            # Install via Modular CLI
            if ! command -v modular &> /dev/null; then
                curl -s https://get.modular.com | sh -
                export PATH="$HOME/.modular/bin:$PATH"
            fi
            
            modular auth
            modular install max
        fi
    else
        print_error "Mojo/MAX not supported on $PLATFORM"
        return 1
    fi
}

# Download and cache model
cache_model() {
    print_status "Caching model: $MODEL_NAME"
    
    python3 << EOF
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("$MODEL_NAME", trust_remote_code=True)
    
    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained("$MODEL_NAME", trust_remote_code=True)
    
    print("Model cached successfully!")
    print(f"Model size: {model.num_parameters():,} parameters")
    
except Exception as e:
    print(f"Error caching model: {e}")
    exit(1)
EOF
}

# Verify installations
verify_installations() {
    print_status "Verifying installations..."
    
    # Test vLLM
    print_status "Testing vLLM..."
    python3 << 'EOF'
try:
    from vllm import LLM, SamplingParams
    print("✓ vLLM imported successfully")
except ImportError as e:
    print(f"✗ vLLM import failed: {e}")
EOF

    # Test Mojo/MAX
    print_status "Testing Mojo/MAX..."
    if command -v max &> /dev/null; then
        print_status "✓ MAX CLI available"
        max --version || print_warning "MAX version check failed"
    else
        print_warning "✗ MAX CLI not found in PATH"
    fi
    
    # Test Python dependencies
    python3 << 'EOF'
import sys
required_packages = ['torch', 'transformers', 'requests', 'psutil', 'numpy']

for package in required_packages:
    try:
        __import__(package)
        print(f"✓ {package}")
    except ImportError:
        print(f"✗ {package} not available")
EOF
}

# Create benchmark configuration
create_config() {
    print_status "Creating benchmark configuration..."
    
    cat > benchmark_config.json << EOF
{
    "model": "$MODEL_NAME",
    "frameworks": ["vllm", "mojo"],
    "test_scenarios": [
        {
            "name": "single_short",
            "prompt_length": 64,
            "max_tokens": 128,
            "concurrent_requests": 1
        },
        {
            "name": "single_medium",
            "prompt_length": 256,
            "max_tokens": 256,
            "concurrent_requests": 1
        },
        {
            "name": "single_long",
            "prompt_length": 1024,
            "max_tokens": 512,
            "concurrent_requests": 1
        },
        {
            "name": "concurrent_5",
            "prompt_length": 64,
            "max_tokens": 128,
            "concurrent_requests": 5
        },
        {
            "name": "concurrent_10",
            "prompt_length": 256,
            "max_tokens": 256,
            "concurrent_requests": 10
        }
    ],
    "platform": "$PLATFORM",
    "gcp_instance": $GCP_INSTANCE
}
EOF
}

# Main installation flow
main() {
    print_status "Starting LLM Benchmark Environment Setup"
    print_status "========================================"
    
    detect_platform
    check_gcp
    
    setup_python_env
    install_common_deps
    
    # Install frameworks
    print_status "Installing frameworks..."
    
    if install_vllm; then
        print_status "✓ vLLM installation completed"
    else
        print_error "✗ vLLM installation failed"
    fi
    
    if install_mojo_max; then
        print_status "✓ Mojo/MAX installation completed"
    else
        print_error "✗ Mojo/MAX installation failed"
    fi
    
    # Cache model
    cache_model
    
    # Verify everything works
    verify_installations
    
    # Create configuration
    create_config
    
    print_status "Setup completed!"
    print_status "=================="
    print_status "Next steps:"
    print_status "1. Run: python benchmark_llm_frameworks.py --quick"
    print_status "2. For full benchmark: python benchmark_llm_frameworks.py"
    print_status "3. Analyze results: python results_analysis.py"
    
    if [[ "$PLATFORM" == "macos_arm64" ]]; then
        print_warning "Note: On macOS, performance will be CPU-only. For GPU testing, use the Linux setup."
    fi
}

# Handle command line arguments
case "${1:-}" in
    --vllm-only)
        detect_platform
        setup_python_env
        install_common_deps
        install_vllm
        ;;
    --mojo-only)
        detect_platform
        setup_python_env
        install_common_deps
        install_mojo_max
        ;;
    --verify)
        verify_installations
        ;;
    --help)
        echo "Usage: $0 [--vllm-only|--mojo-only|--verify|--help]"
        echo "  --vllm-only: Install only vLLM"
        echo "  --mojo-only: Install only Mojo/MAX"
        echo "  --verify: Verify existing installations"
        echo "  --help: Show this help"
        exit 0
        ;;
    *)
        main
        ;;
esac