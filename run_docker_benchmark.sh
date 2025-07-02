#!/bin/bash

# Docker-based vLLM vs MAX benchmark on GPU
set -e

echo "GPU DOCKER BENCHMARK RUNNER"
echo "==========================="
echo "Running on: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

# Kill any existing processes
pkill -f "vllm" || true
pkill -f "max serve" || true
docker stop vllm-server || true
docker rm vllm-server || true
sleep 5

# Set environment variables
export HF_TOKEN="${HF_TOKEN:-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}"
export CUDA_VISIBLE_DEVICES=0

echo "ROUND 1: vLLM DOCKER WITH FULL OPTIMIZATIONS"
echo "============================================"

# Start vLLM Docker container with all optimizations
echo "Starting vLLM Docker container..."
docker run -d \
  --gpus all \
  --name vllm-server \
  -p 8001:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  -e VLLM_USE_V1=1 \
  -e VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  vllm/vllm-openai:latest \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256 \
  --max-model-len 8192

echo "Waiting for vLLM Docker to initialize..."
for i in {1..180}; do
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo "vLLM Docker ready!"
        break
    fi
    if [ $i -eq 180 ]; then
        echo "vLLM Docker failed to start"
        docker logs vllm-server
        exit 1
    fi
    sleep 1
    echo -n "."
done

# Test vLLM
echo ""
echo "Testing vLLM Docker inference..."
curl -X POST http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 5,
    "temperature": 0.0
  }' | jq '.'

# Run vLLM benchmark
echo ""
echo "Running vLLM Docker benchmark..."
source llm_benchmark_server/bin/activate
python benchmark_servers.py --quick --frameworks vllm
cp server_benchmark_results.json vllm_docker_results.json

# Stop vLLM
echo "Stopping vLLM Docker..."
docker stop vllm-server
docker rm vllm-server
sleep 10

echo ""
echo "ROUND 2: MAX OPTIMIZED"
echo "======================"

# Start MAX server
cd max_project
export HF_HUB_ENABLE_HF_TRANSFER=1
pixi run max serve --model-path=meta-llama/Meta-Llama-3-8B-Instruct --port=8002 &
MAX_PID=$!
cd ..

echo "Waiting for MAX to initialize..."
for i in {1..180}; do
    if curl -s http://localhost:8002/health > /dev/null 2>&1; then
        echo "MAX ready!"
        break
    fi
    if [ $i -eq 180 ]; then
        echo "MAX failed to start"
        kill $MAX_PID || true
        exit 1
    fi
    sleep 1
    echo -n "."
done

# Test MAX
echo ""
echo "Testing MAX inference..."
curl -X POST http://localhost:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 5,
    "temperature": 0.0
  }' | jq '.'

# Run MAX benchmark
echo ""
echo "Running MAX benchmark..."
python benchmark_servers.py --quick --frameworks max
cp server_benchmark_results.json max_docker_results.json

# Stop MAX
kill $MAX_PID || true

echo ""
echo "PERFORMANCE COMPARISON"
echo "====================="

echo "vLLM Docker Results:"
python -c "
import json
with open('vllm_docker_results.json') as f:
    results = json.load(f)
    for r in results:
        print(f'  TTFT: {r[\"ttft_ms\"]:.2f}ms, TPS: {r[\"tokens_per_second\"]:.2f}, Throughput: {r[\"throughput_req_per_sec\"]:.2f} req/s')
        break
"

echo "MAX Results:"
python -c "
import json
with open('max_docker_results.json') as f:
    results = json.load(f)
    for r in results:
        print(f'  TTFT: {r[\"ttft_ms\"]:.2f}ms, TPS: {r[\"tokens_per_second\"]:.2f}, Throughput: {r[\"throughput_req_per_sec\"]:.2f} req/s')
        break
"

echo ""
echo "WINNER ANALYSIS:"
python -c "
import json
try:
    with open('vllm_docker_results.json') as f:
        vllm = json.load(f)[0]
    with open('max_docker_results.json') as f:
        max_r = json.load(f)[0]
    
    if vllm['ttft_ms'] < max_r['ttft_ms']:
        print(f'TTFT Winner: vLLM Docker ({vllm[\"ttft_ms\"]:.1f}ms vs {max_r[\"ttft_ms\"]:.1f}ms)')
    else:
        print(f'TTFT Winner: MAX ({max_r[\"ttft_ms\"]:.1f}ms vs {vllm[\"ttft_ms\"]:.1f}ms)')
    
    if vllm['tokens_per_second'] > max_r['tokens_per_second']:
        print(f'TPS Winner: vLLM Docker ({vllm[\"tokens_per_second\"]:.1f} vs {max_r[\"tokens_per_second\"]:.1f})')
    else:
        print(f'TPS Winner: MAX ({max_r[\"tokens_per_second\"]:.1f} vs {vllm[\"tokens_per_second\"]:.1f})')
        
except Exception as e:
    print(f'Error: {e}')
"

echo ""
echo "Docker benchmark complete!"
echo "Files: vllm_docker_results.json, max_docker_results.json"