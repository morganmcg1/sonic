# LLM Framework Benchmarking: vLLM vs Mojo/Modular MAX

A comprehensive benchmarking suite comparing vLLM and Mojo/Modular MAX frameworks for LLM inference performance on macOS (M4 Max) and GPU environments.

## Overview

This project provides tools to benchmark and compare:
- **vLLM**: High-throughput inference and serving engine
- **Mojo/Modular MAX**: Experimental AI compute platform with Mojo language

### Key Metrics Measured
- **Time to First Token (TTFT)**: Latency until first token generation
- **Tokens per Second (TPS)**: Generation throughput 
- **Request Throughput**: Requests processed per second
- **Resource Usage**: Memory and CPU utilization
- **Latency Percentiles**: P50, P95, P99 response times

## Project Structure

```
.
├── benchmark_llm_frameworks.py  # Main benchmarking script
├── results_analysis.py          # Results visualization and analysis
├── setup_environments.sh        # Environment setup script
├── README.md                     # This documentation
├── llm_benchmark/               # Python virtual environment
├── max_test/                    # Mojo/MAX pixi environment
└── benchmark_results.json       # Generated benchmark results
```

## Quick Start

### 1. Environment Setup

```bash
# Make setup script executable
chmod +x setup_environments.sh

# Run full setup (installs both frameworks)
./setup_environments.sh

# Or install individually:
./setup_environments.sh --vllm-only
./setup_environments.sh --mojo-only
```

### 2. Run Benchmarks

```bash
# Activate Python environment
source llm_benchmark/bin/activate

# Quick benchmark (both frameworks)
python benchmark_llm_frameworks.py --quick

# Quick benchmark (single framework)
python benchmark_llm_frameworks.py --quick --frameworks vllm
python benchmark_llm_frameworks.py --quick --frameworks mojo

# Full comprehensive benchmark
python benchmark_llm_frameworks.py

# Custom model
python benchmark_llm_frameworks.py --model "allenai/OLMo-1B-hf"
```

### 3. Analyze Results

```bash
# Generate analysis and visualizations
python results_analysis.py

# Summary only
python results_analysis.py --summary-only

# Custom results file
python results_analysis.py custom_results.json
```

## Requirements

### System Requirements
- **macOS**: Apple Silicon (M1/M2/M3/M4) recommended
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **Python**: 3.10-3.12

### Dependencies
- **uv**: Python package manager (recommended)
- **pixi**: Package manager for Mojo/MAX
- PyTorch, transformers, vLLM
- MAX/Modular platform

## Benchmark Configuration

### Test Scenarios
The benchmark runs multiple scenarios varying:
- **Prompt lengths**: 64, 256, 1024 tokens
- **Output lengths**: 128, 256, 512 tokens  
- **Concurrency**: 1, 5, 10 concurrent requests
- **Batch sizes**: Different batching strategies

### Supported Models
- **OLMo-1B-hf**: Primary test model (works with both frameworks)
- **Qwen2.5-0.5B-Instruct**: Alternative small model
- **TinyLlama-1.1B**: Another option for testing

> **Note**: Model compatibility varies between frameworks and CPU/GPU configurations.

## Results Analysis

The analysis script generates:
- **Performance comparison tables**
- **Visualization charts** (TTFT, TPS, throughput, resource usage)
- **Statistical analysis** (percentiles, averages, ratios)
- **Detailed text reports**

### Sample Output Structure
```json
{
  "framework": "VLLM",
  "model": "allenai/OLMo-1B-hf", 
  "ttft_ms": 229.17,
  "tokens_per_second": 55.85,
  "throughput_req_per_sec": 0.44,
  "peak_memory_mb": 2048.5,
  "latency_p95_ms": 2291.70
}
```

## Performance Insights (Preliminary)

Based on initial M4 Max testing with OLMo-1B model:

### vLLM Strengths
- **Higher throughput**: 55-134 tokens/sec
- **Better concurrent processing**: Scales well with multiple requests
- **Mature optimization**: Production-ready with extensive optimizations

### Mojo/MAX Strengths  
- **Lower TTFT**: ~197ms vs 229-478ms (vLLM)
- **Efficient startup**: Faster model compilation
- **Memory efficiency**: Lower memory footprint per request

### Trade-offs
- **vLLM**: Better for high-throughput batch processing
- **Mojo/MAX**: Better for low-latency single requests

## GPU Testing

For GPU testing on Google Cloud:

```bash
# Check available GPU instances  
gcloud compute instances list --filter="status=RUNNING"

# Connect to GPU instance
gcloud compute ssh your-gpu-instance

# Run GPU benchmarks
python benchmark_llm_frameworks.py --frameworks vllm  # vLLM with GPU
python benchmark_llm_frameworks.py --frameworks mojo  # MAX with GPU
```

## Troubleshooting

### Common Issues

1. **vLLM on macOS**: Limited to CPU inference, full GPU support requires Linux
2. **MAX installation**: Requires Apple Silicon, use pixi for best compatibility
3. **Model compatibility**: Not all models work with both frameworks
4. **Memory issues**: Reduce batch size or use smaller models

### Debug Commands
```bash
# Verify installations
./setup_environments.sh --verify

# Test individual frameworks
python -c "import vllm; print('vLLM OK')"
cd max_test && pixi run max --version
```

## Advanced Usage

### Custom Benchmarks
```python
# Add custom test scenarios
custom_scenarios = [
    {"prompt_length": 512, "max_tokens": 1024, "concurrent_requests": 3}
]
```

### Framework Extensions
- Add new metrics in `BenchmarkResult` dataclass
- Implement custom analysis in `results_analysis.py`
- Extend framework classes for new LLM engines

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## License

This project is for research and educational purposes. Please respect the licenses of the underlying frameworks:
- vLLM: Apache 2.0
- Mojo/MAX: Modular License Agreement

## Acknowledgments

- vLLM team for the high-performance inference engine
- Modular team for the innovative Mojo/MAX platform
- Hugging Face for model hosting and transformers library