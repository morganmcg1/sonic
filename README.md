# Fair vLLM vs MAX Benchmarking Suite

A comprehensive, industry-standard benchmarking framework for comparing vLLM and Modular MAX frameworks using official tools and methodology.

## 🎯 Overview

This project provides **fair, accurate comparison** between:
- **vLLM**: High-throughput inference and serving engine  
- **Modular MAX**: AI compute platform with optimized inference

**Key Features:**
- ✅ Uses MAX's official `benchmark_serving.py` for industry-standard methodology
- ✅ Accurate TTFT measurement via streaming APIs (not estimation)
- ✅ Standardized configurations ensuring fair comparison
- ✅ Comprehensive validation to prevent configuration errors
- ✅ Multiple realistic workloads (ShareGPT, Sonnet, etc.)
- ✅ Statistical rigor with Poisson request distribution

## 🚀 Quick Start

### Prerequisites
```bash
# Required: Hugging Face token for model access
export HF_TOKEN="your_huggingface_token_here"

# Optional: GPU available (NVIDIA recommended)
nvidia-smi  # Check GPU status
```

### 1. Complete Setup
```bash
# Download and run complete environment setup
./setup_improved_benchmark.sh
```

### 2. Validate Environment
```bash
# Verify all components are properly configured
python validate_benchmark_setup.py

# Fix any issues shown before proceeding
```

### 3. Run Benchmarks

#### Quick Benchmark (2 scenarios)
```bash
./run_complete_benchmark.sh --quick
```

#### Full Benchmark (5 scenarios)  
```bash
./run_complete_benchmark.sh
```

#### Validation Only
```bash
./run_complete_benchmark.sh --validate-only
```

## 📊 Benchmark Parameters

### Model & Configuration
- **Model**: `meta-llama/Meta-Llama-3-8B-Instruct`
- **Precision**: `bfloat16` (consistent across frameworks)
- **API**: OpenAI-compatible endpoints for both frameworks

### vLLM Optimizations
```bash
# Server configuration
--dtype bfloat16
--gpu-memory-utilization 0.9  
--enable-chunked-prefill
--max-num-batched-tokens 8192
--max-num-seqs 256

# Environment variables  
VLLM_USE_V1=1                    # Latest v1 engine
VLLM_ATTENTION_BACKEND=FLASH_ATTN # Flash Attention backend
VLLM_FLASH_ATTN_FORCE_ENABLE=1   # Force optimization
```

### MAX Optimizations
```bash
# Server configuration
--model-path=meta-llama/Meta-Llama-3-8B-Instruct
--port=8002

# Environment variables
HF_HUB_ENABLE_HF_TRANSFER=1      # Fast model loading
HF_TOKEN=your_token              # Model access
```

### Test Scenarios

#### Quick Mode (--quick)
- **2 scenarios**: Request rates 1.0, 2.0 req/s
- **50 prompts** per scenario
- **~10 minutes** total runtime

#### Full Mode (default)
- **5 scenarios**: Request rates 0.5, 1.0, 2.0, 4.0, 8.0 req/s
- **100 prompts** per scenario  
- **~30 minutes** total runtime

### Datasets Used
- **ShareGPT**: Realistic conversational prompts
- **Sonnet**: Poetry generation tasks
- **Code Debug**: Programming contexts

## 📈 Metrics Measured

### Performance Metrics
- **Request Throughput**: Requests processed per second
- **Input Token Throughput**: Input tokens processed per second
- **Output Token Throughput**: Output tokens generated per second
- **TTFT (Time to First Token)**: Mean, Median, P99 latencies
- **TPOT (Time Per Output Token)**: Inter-token generation time

### System Metrics
- **GPU Utilization**: Memory and compute usage
- **Peak Memory**: Maximum memory consumption
- **Error Rates**: Failed requests and reasons

## 📁 Project Structure

```
.
├── benchmark_serving_official.py    # Official MAX benchmark tool
├── setup_improved_benchmark.sh      # Complete environment setup
├── validate_benchmark_setup.py      # Configuration validation
├── run_complete_benchmark.sh        # Full benchmark orchestration
├── benchmark_fair_comparison.py     # Automated comparison runner
├── start_vllm_optimized.sh         # Optimized vLLM server startup (generated)
├── start_max_optimized.sh          # Optimized MAX server startup (generated)
├── vllm_optimized_env/             # vLLM Python environment (generated)
├── max_optimized/                  # MAX pixi environment (generated)
└── benchmark_results_[timestamp]/  # Generated results directory
```

**Core Scripts:**
- `setup_improved_benchmark.sh` - Creates all environments and startup scripts
- `run_complete_benchmark.sh` - Runs complete benchmark suite using official tools

## 📊 Understanding Results

### Result Files
After benchmarking, you'll find in `benchmark_results_[timestamp]/`:
```
validation_results.json          # Environment validation report
vllm_scenario_0.json            # vLLM scenario results
max_scenario_0.json             # MAX scenario results  
analyze_results.py              # Automated analysis script
benchmark_summary.md            # Human-readable summary
```

### Sample Result Structure
```json
{
  "request_throughput": 2.45,           # req/s
  "input_token_throughput": 1580.3,     # tokens/s
  "output_token_throughput": 312.7,     # tokens/s
  "mean_ttft_ms": 187.2,               # milliseconds
  "median_ttft_ms": 156.8,             # milliseconds  
  "p99_ttft_ms": 456.1,               # milliseconds
  "mean_tpot_ms": 3.2,                # milliseconds
  "median_tpot_ms": 3.1,              # milliseconds
  "p99_tpot_ms": 4.8                  # milliseconds
}
```

### Analyzing Results
```bash
cd benchmark_results_[timestamp]
python analyze_results.py           # Detailed comparison
cat benchmark_summary.md            # Human-readable report
```

## 🛠️ Advanced Usage

### Manual Server Management
```bash
# Start servers manually in separate terminals
Terminal 1: ./start_vllm_optimized.sh
Terminal 2: ./start_max_optimized.sh

# Run benchmarks against running servers
python benchmark_serving_official.py --backend vllm --port 8001 --model meta-llama/Meta-Llama-3-8B-Instruct
python benchmark_serving_official.py --backend modular --port 8002 --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Custom Scenarios
```bash
# Custom request rate and prompt count
python benchmark_serving_official.py \
  --backend vllm \
  --port 8001 \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --request-rate 3.0 \
  --num-prompts 200 \
  --dataset-name sonnet
```

### Different Datasets
```bash
# Test with different workload types
--dataset-name sharegpt      # Conversational (default)
--dataset-name sonnet        # Poetry generation  
--dataset-name random        # Synthetic prompts
--dataset-name code_debug    # Programming tasks
```

## 🐛 Troubleshooting

### Validation Failures
```bash
# Run validation to identify issues
python validate_benchmark_setup.py --output validation_report.json

# Common fixes
export HF_TOKEN="your_token"                    # Missing token
export VLLM_USE_V1=1                           # vLLM optimization  
export VLLM_ATTENTION_BACKEND=FLASH_ATTN       # Flash Attention backend
export HF_HUB_ENABLE_HF_TRANSFER=1             # MAX optimization
```

### Server Startup Issues
```bash
# Check server logs
./start_vllm_optimized.sh  # Check vLLM startup output
./start_max_optimized.sh   # Check MAX startup output

# Test server health
curl http://localhost:8001/health  # vLLM health
curl http://localhost:8002/health  # MAX health
```

### Environment Issues
```bash
# Recreate environments
rm -rf vllm_optimized_env max_optimized
./setup_improved_benchmark.sh

# Check GPU availability
nvidia-smi
```

### Memory Issues
```bash
# Monitor GPU memory
nvidia-smi -l 1

# Kill existing processes
pkill -f vllm
pkill -f "max serve"
```

### Flash Attention Installation Issues
```bash
# ERROR: "nvcc not found" or "CUDA_HOME not set"
# SOLUTION: Use pre-compiled wheels instead of compilation
wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.12/flash_attn-2.8.0+cu124torch2.7-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.8.0+cu124torch2.7-cp312-cp312-linux_x86_64.whl

# ERROR: "metadata-generation-failed"
# SOLUTION: Verify PyTorch and CUDA versions match wheel requirements
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

# ERROR: "ModuleNotFoundError: No module named 'flash_attn'"
# SOLUTION: Activate correct virtual environment
source vllm_optimized_env/bin/activate
```

### Package Manager Lock Issues
```bash
# ERROR: "dpkg frontend lock" or "apt held by process"
# SOLUTION: Wait for background processes or force cleanup
sudo fuser -v /var/lib/dpkg/lock-frontend
sudo kill -9 <process_id>  # Use PID from above command
sudo dpkg --configure -a
sudo apt update
```

### VM Access and SSH Issues
```bash
# ERROR: "Permission denied (publickey)"
# SOLUTION: Ensure proper gcloud authentication
gcloud auth login
gcloud config set project your-project-id

# ERROR: "External IP address was not found"
# SOLUTION: This is normal for IAP tunneling, connection will work
```

## ⚡ Critical Setup Learnings (Must Read!)

### 🚨 Essential vLLM Setup Requirements

**1. Correct Attention Backend Configuration:**
```bash
# ❌ WRONG: Will cause "flashinfer module not found" error
export VLLM_ATTENTION_BACKEND=FLASHINFER

# ✅ CORRECT: Use with Flash Attention 2.8.0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

**2. Required Additional Dependencies:**
```bash
# Install in vLLM environment before benchmarking
source vllm_optimized_env/bin/activate
pip install hf_transfer  # Required for HF_HUB_ENABLE_HF_TRANSFER=1
```

**3. Flash Attention Installation Process:**
```bash
# Step 1: Check PyTorch/CUDA versions for compatibility
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

# Step 2: Download correct pre-compiled wheel (saves 1+ hours vs compilation)
# For PyTorch 2.7.0 + CUDA 12.6 + Python 3.12:
wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.12/flash_attn-2.8.0+cu124torch2.7-cp312-cp312-linux_x86_64.whl

# Step 3: Install wheel
pip install flash_attn-2.8.0+cu124torch2.7-cp312-cp312-linux_x86_64.whl

# Step 4: Verify installation
python -c "import flash_attn; print(f'Flash Attention version: {flash_attn.__version__}')"
```

**4. vLLM Server Startup Expectations:**
```bash
# First startup takes 2-3 minutes due to:
# - Model downloading (~15GB for Llama-3-8B)
# - CUDA graph compilation (torch.compile)
# - Flash Attention initialization

# Subsequent startups: ~30-60 seconds
```

### 🔧 Essential MAX Setup Requirements

**1. Correct Model Path Parameter:**
```bash
# ❌ WRONG: Will cause "model-path must be provided" error
max serve meta-llama/Meta-Llama-3-8B-Instruct

# ✅ CORRECT: Use --model-path parameter
max serve --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 8002 --host 0.0.0.0
```

**2. MAX Worker Debugging:**
```bash
# Check MAX startup logs for worker issues
tail -f max_startup.log

# Common issues:
# - "Worker died" - Usually CUDA/driver compatibility
# - "GLIBC version" - Requires Ubuntu 22.04+ (GLIBC 2.35+)
# - "Memory allocation" - GPU memory insufficient
```

**3. MAX Environment Activation:**
```bash
# Navigate to MAX directory first
cd max_optimized

# Use full path to max binary
.pixi/envs/default/bin/max serve --model-path meta-llama/Meta-Llama-3-8B-Instruct
```

### 🎯 Benchmarking Process Best Practices

**1. Sequential Framework Testing (CRITICAL):**
```bash
# A100-40GB cannot run both frameworks simultaneously
# Must clear GPU memory between tests

# Step 1: Run vLLM benchmark
./start_vllm_optimized.sh
python benchmark_serving_official.py --backend vllm --port 8001 [options]

# Step 2: Clear GPU memory
pkill -f vllm
sleep 10  # Wait for memory clearance

# Step 3: Run MAX benchmark  
cd max_optimized
.pixi/envs/default/bin/max serve --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 8002 &
cd ..
python benchmark_serving_official.py --backend modular --port 8002 [options]
```

**2. Memory Monitoring Commands:**
```bash
# Monitor GPU memory in real-time
nvidia-smi -l 1

# Check memory before/after tests
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# Expected memory usage:
# - vLLM: ~20-25GB for Llama-3-8B
# - MAX: ~30-40GB for Llama-3-8B
```

**3. Environment Variable Persistence:**
```bash
# Variables don't persist across SSH sessions!
# Create a script for easy setup:
cat > setup_env.sh << 'EOF'
export HF_TOKEN="your_huggingface_token"
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_FLASH_ATTN_FORCE_ENABLE=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_VISIBLE_DEVICES=0
EOF

# Source before each benchmarking session
source setup_env.sh
```

### 📊 Expected Performance Benchmarks

**vLLM with Flash Attention 2.8.0 (A100 GPU):**
```
✅ Proven Results (meta-llama/Meta-Llama-3-8B-Instruct):
- Request throughput: ~1.5 req/s
- Input token throughput: ~350 tok/s  
- Output token throughput: ~310 tok/s
- Mean TTFT: ~42ms
- Mean TPOT: ~15ms
- Success rate: 100%
```

**Performance Troubleshooting:**
```bash
# If performance is significantly lower:
# 1. Verify Flash Attention is active
python -c "import flash_attn; print('Flash Attention available')"

# 2. Check attention backend in server logs
grep "Using Flash Attention backend" vllm_startup.log

# 3. Verify environment variables
echo $VLLM_ATTENTION_BACKEND  # Should be FLASH_ATTN
echo $VLLM_USE_V1            # Should be 1

# 4. Monitor GPU utilization during benchmark
nvidia-smi -l 1
```

### 🔄 Quick Debug Commands

**Reset Everything:**
```bash
# Kill all processes
pkill -f vllm; pkill -f "max serve"

# Clear GPU memory
sleep 10

# Check memory cleared
nvidia-smi

# Restart from clean state
source setup_env.sh
```

**Validation Quick Check:**
```bash
# Test vLLM environment
source vllm_optimized_env/bin/activate && python validate_benchmark_setup.py
# Should show 23/25 tests passed

# Test individual components
curl http://localhost:8001/health  # vLLM server health
curl http://localhost:8002/health  # MAX server health
```

## 🔍 Methodology Details

### Why This Approach is Better

**Previous Issues Fixed:**
- ❌ Custom benchmark with TTFT estimation → ✅ Official tool with streaming measurement
- ❌ Inconsistent configurations → ✅ Standardized optimizations
- ❌ Limited test scenarios → ✅ Multiple realistic workloads
- ❌ No validation → ✅ Comprehensive environment checking

**Industry Standards:**
- Uses MAX's official `benchmark_serving.py` (accepted by both teams)
- Follows vLLM's performance optimization guidelines
- Implements proper statistical sampling with Poisson distribution
- Measures real TTFT via streaming APIs

### Configuration Verification
The validation suite ensures:
- Environment variables are correctly set
- Dependencies are properly installed
- Model access is configured
- Server configurations match specifications
- Response consistency between frameworks

## 🚀 GPU vs CPU Performance

### Recommended Hardware
- **GPU**: NVIDIA A100, H100, or L4 for optimal performance
- **CPU**: Apple Silicon (M1/M2/M3/M4) or high-end Intel/AMD
- **Memory**: 16GB+ RAM, 40GB+ VRAM for Llama-3-8B

### Platform Support
- **Linux + NVIDIA GPU**: Full acceleration for both frameworks
- **macOS Apple Silicon**: CPU inference, limited GPU support
- **Other platforms**: CPU inference only

## ☁️ Google Cloud GPU Setup

### VM Creation for Optimal Performance

#### A100 GPU Instance (Recommended)
```bash
# Create A100 GPU VM for maximum performance
gcloud compute instances create llm-benchmark-a100 \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-1g \
    --accelerator=count=1,type=nvidia-tesla-a100 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True"
```

**🎯 Recommended VM Image Selection:**
- **Best**: `pytorch-latest-gpu` from `deeplearning-platform-release`
  - ✅ Pre-installed NVIDIA drivers (550+)
  - ✅ CUDA 12.4+ ready
  - ✅ Python 3.10 (upgrade to 3.12 required)
  - ✅ PyTorch ecosystem ready
  - ✅ Docker and containerization tools
- **Alternative**: `ubuntu-2204-lts` (requires more manual setup)
- **Avoid**: Standard compute images (missing GPU drivers and CUDA)

#### L4 GPU Instance (Cost-Effective)  
```bash
# Create L4 GPU VM for cost-effective testing
gcloud compute instances create llm-benchmark-l4 \
    --zone=us-central1-a \
    --machine-type=g2-standard-4 \
    --accelerator=count=1,type=nvidia-l4 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True"
```

### VM Requirements & Specifications

#### System Requirements
- **OS**: Ubuntu 22.04 LTS (GLIBC 2.35+)
- **CUDA**: Version 12.4+ with NVIDIA driver 550+
- **Python**: 3.10+ (included in GPU images)
- **Storage**: 100GB+ boot disk for models and dependencies

#### GPU Memory Requirements
- **A100-40GB**: Can run both frameworks (sequential benchmarking required)
  - vLLM: ~20GB memory usage
  - MAX: ~36GB memory usage (90% utilization)
- **L4-24GB**: Suitable for testing, may require smaller models
- **Memory Constraint**: Cannot run both frameworks simultaneously on A100-40GB

### VM Setup Process

#### Current Benchmarking VM
- **Instance Name**: `llm-benchmark-a100-max`
- **Zone**: `us-central1-b`
- **Machine Type**: `a2-highgpu-1g` (A100 GPU)

```bash
# 1. Connect to the benchmarking VM
gcloud compute ssh llm-benchmark-a100-max --zone=us-central1-b

# 2. Install Python 3.12 and set as default (REQUIRED)
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y && sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev python3.12-pip
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# 3. Verify GPU and Python
nvidia-smi && python --version

# 4. Navigate to project (if already cloned) or clone repository
cd sonic  # if already exists
# OR
git clone https://github.com/morganmcg1/sonic.git && cd sonic

# 5. Set environment variables (REQUIRED FOR EVERY SESSION)
export HF_TOKEN="your_huggingface_token"
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_FLASH_ATTN_FORCE_ENABLE=1  
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_VISIBLE_DEVICES=0

# 6. Run complete setup (takes 15-20 minutes)
./setup_improved_benchmark.sh

# 7. Validate environment
python validate_benchmark_setup.py

# 8. Run benchmarks
./run_complete_benchmark.sh --quick
```

### Expected Memory Usage (A100 GPU)

Based on framework specifications for meta-llama/Meta-Llama-3-8B-Instruct:

#### GPU Memory Requirements
- **vLLM**: ~15-25GB (depending on configuration)
- **MAX**: ~30-40GB (more aggressive memory utilization)
- **Constraint**: Cannot run both frameworks simultaneously on A100-40GB
- **Solution**: Sequential benchmarking (stop one framework before starting the other)

### GPU VM Best Practices

#### Instance Management
```bash
# Start benchmarking VM when needed
gcloud compute instances start llm-benchmark-a100-max --zone=us-central1-b

# Stop VM to save costs (A100 is expensive!)
gcloud compute instances stop llm-benchmark-a100-max --zone=us-central1-b

# Check VM status
gcloud compute instances list --filter="name=llm-benchmark-a100-max"

# Monitor costs
gcloud billing budgets list
```

#### Environment Optimization
```bash
# Set optimal environment variables for GPU
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_FLASH_ATTN_FORCE_ENABLE=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_VISIBLE_DEVICES=0

# Install Flash Attention for maximum vLLM performance
pip install flash-attn --no-build-isolation
```

#### Critical Requirements for MAX
- **NVIDIA Driver**: 550+ (required for MAX compatibility)
- **CUDA**: 12.4+ (verified working version)
- **GLIBC**: 2.35+ (Ubuntu 22.04 provides 2.35)
- **Python**: 3.12+ (required for optimal performance)

#### VM Setup Requirements & Gotchas
**⚠️ Critical Setup Steps (Required):**
1. **Python 3.12 Installation**: GPU VMs come with Python 3.10, but 3.12+ is recommended
2. **Python Command**: VMs only have `python3`, need to create `python` symlink
3. **Environment Persistence**: Environment variables don't persist across SSH sessions
4. **Package Manager**: Install `pip` and upgrade to latest version
5. **Flash Attention**: Use pre-compiled wheels to avoid 1+ hour compilation
6. **CUDA Development**: Pre-compiled wheels eliminate need for nvcc/cuda-toolkit
7. **MAX Environment**: Requires Pixi package manager for proper setup

**🔧 VM Initialization Commands:**
```bash
# Connect to VM
gcloud compute ssh llm-benchmark-a100-max --zone=us-central1-b

# Install Python 3.12 and make it default
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev python3.12-pip
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
python --version  # Should show Python 3.12.x

# Verify GPU and drivers
nvidia-smi
```

**⚡ Flash Attention Installation (CRITICAL for Fair Comparison):**
```bash
# Navigate to project directory
cd sonic && source vllm_optimized_env/bin/activate

# Download pre-compiled Flash Attention wheel (much faster than compilation)
wget https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.12/flash_attn-2.8.0+cu124torch2.7-cp312-cp312-linux_x86_64.whl

# Install Flash Attention wheel (takes seconds vs hours of compilation)
pip install flash_attn-2.8.0+cu124torch2.7-cp312-cp312-linux_x86_64.whl

# Verify installation
python -c "import flash_attn; print(f'Flash Attention version: {flash_attn.__version__}')"
```

**📦 Alternative Flash Attention Sources:**
- **Primary**: [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases)
- **Windows**: [sunsetcoder/flash-attention-windows](https://github.com/sunsetcoder/flash-attention-windows)
- **Official**: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) (requires compilation)

**🔐 Environment Variables (Set Every Session):**
```bash
# Required for every benchmarking session
export HF_TOKEN="your_huggingface_token"
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN  
export VLLM_FLASH_ATTN_FORCE_ENABLE=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_VISIBLE_DEVICES=0
```


#### Memory Management Tips
- **Monitor Usage**: `nvidia-smi -l 1` for real-time monitoring
- **Kill Processes**: `pkill -f vllm` and `pkill -f "max serve"` between tests
- **Clear Memory**: Wait 10+ seconds between framework switches
- **Sequential Testing**: Cannot run both frameworks simultaneously on A100-40GB

## 📚 References

- [MAX Official Benchmark Documentation](https://docs.modular.com/max/tutorials/benchmark-max-serve/)
- [MAX Official Benchmark Script](https://github.com/modular/modular/blob/main/benchmark/benchmark_serving.py)
- [vLLM Performance Best Practices](https://docs.vllm.ai/en/latest/design/v1/p2p_nccl_connector.html)
- [vLLM Docker Deployment](https://docs.vllm.ai/en/latest/deployment/docker.html)

## 🤝 Contributing

1. Test with different models and configurations
2. Report issues with environment details
3. Submit improvements via pull requests
4. Add validation checks for edge cases

## 📄 License

This project is for research and educational purposes. Respects the licenses of:
- **vLLM**: Apache 2.0 License
- **MAX**: Modular License Agreement

---

**Note**: This benchmarking suite implements industry-standard methodology for fair comparison between vLLM and MAX frameworks, addressing critical measurement accuracy issues in previous approaches.