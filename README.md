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
VLLM_ATTENTION_BACKEND=FLASHINFER # Fastest attention
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
export VLLM_ATTENTION_BACKEND=FLASHINFER       # Attention backend
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
export VLLM_ATTENTION_BACKEND=FLASHINFER
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
export VLLM_ATTENTION_BACKEND=FLASHINFER
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
5. **Flash Attention**: Requires compilation, can take 10+ minutes on first install

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

**🔐 Environment Variables (Set Every Session):**
```bash
# Required for every benchmarking session
export HF_TOKEN="your_huggingface_token"
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASHINFER  
export VLLM_FLASH_ATTN_FORCE_ENABLE=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_VISIBLE_DEVICES=0
```

**⏱️ Expected Setup Times:**
- Environment setup: 15-20 minutes
- Flash Attention compilation: 5-10 minutes  
- Full benchmark (quick): ~10 minutes
- Full benchmark (complete): ~30 minutes

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