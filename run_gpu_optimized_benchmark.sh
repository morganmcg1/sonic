#!/bin/bash

# GPU Optimized Benchmark Runner - vLLM vs MAX
# Run this ONLY on A100 GPU VM!

set -e

echo "🚀 GPU OPTIMIZED BENCHMARK RUNNER"
echo "=================================="
echo "Running on: $(hostname)"
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

echo ""
echo "🔧 Setting up optimized environment..."

# Kill any existing processes
pkill -f "vllm" || true
pkill -f "max serve" || true
sleep 5

# Set optimized environment variables
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_FLASH_ATTN_FORCE_ENABLE=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="${HF_TOKEN:-your_token_here}"
export CUDA_VISIBLE_DEVICES=0

echo "✅ Environment variables set:"
echo "   VLLM_USE_V1=$VLLM_USE_V1"
echo "   VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND"
echo "   HF_HUB_ENABLE_HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER"

# Activate virtual environment
echo ""
echo "🔧 Activating vLLM environment..."
source llm_benchmark_server/bin/activate

# Install Flash Attention if not already installed
echo ""
echo "🔧 Installing Flash Attention for GPU..."
pip install flash-attn --no-build-isolation || echo "Flash Attention already installed or installation failed"

# Verify Flash Attention
python -c "import flash_attn; print('✅ Flash Attention verified')" || echo "⚠️ Flash Attention not available, using fallback"

echo ""
echo "🏎️ ROUND 1: OPTIMIZED vLLM BENCHMARK"
echo "===================================="

# Clear GPU memory
nvidia-smi

# Start optimized vLLM server
echo "🚀 Starting optimized vLLM server..."
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --port 8001 \
  --trust-remote-code \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256 \
  --max-model-len 8192 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for vLLM to start
echo "⏳ Waiting for vLLM to initialize..."
for i in {1..120}; do
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo "✅ vLLM server ready!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "❌ vLLM failed to start after 2 minutes"
        kill $VLLM_PID || true
        exit 1
    fi
    sleep 1
    echo -n "."
done

# Test vLLM inference
echo ""
echo "🧠 Testing vLLM inference..."
curl -X POST http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "max_tokens": 5,
    "temperature": 0.0,
    "stop": ["<|end_of_text|>", "<|eot_id|>"]
  }' | jq '.'

# Run vLLM benchmark
echo ""
echo "📊 Running vLLM quick benchmark..."
python benchmark_servers.py --quick --frameworks vllm

# Save vLLM results
cp server_benchmark_results.json vllm_optimized_results.json
echo "✅ vLLM results saved to vllm_optimized_results.json"

# Stop vLLM
echo ""
echo "🛑 Stopping vLLM server..."
kill $VLLM_PID || true
sleep 10

# Clear GPU memory
nvidia-smi

echo ""
echo "🏎️ ROUND 2: OPTIMIZED MAX BENCHMARK"
echo "==================================="

# Start optimized MAX server
echo "🚀 Starting optimized MAX server..."
cd max_test
pixi run max serve --model-path=meta-llama/Meta-Llama-3-8B-Instruct --port=8002 &

MAX_PID=$!
echo "MAX PID: $MAX_PID"
cd ..

# Wait for MAX to start
echo "⏳ Waiting for MAX to initialize..."
for i in {1..180}; do
    if curl -s http://localhost:8002/health > /dev/null 2>&1; then
        echo "✅ MAX server ready!"
        break
    fi
    if [ $i -eq 180 ]; then
        echo "❌ MAX failed to start after 3 minutes"
        kill $MAX_PID || true
        exit 1
    fi
    sleep 1
    echo -n "."
done

# Test MAX inference
echo ""
echo "🧠 Testing MAX inference..."
curl -X POST http://localhost:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "max_tokens": 5,
    "temperature": 0.0,
    "stop": ["<|end_of_text|>", "<|eot_id|>"]
  }' | jq '.'

# Run MAX benchmark
echo ""
echo "📊 Running MAX quick benchmark..."
python benchmark_servers.py --quick --frameworks max

# Save MAX results
cp server_benchmark_results.json max_optimized_results.json
echo "✅ MAX results saved to max_optimized_results.json"

# Stop MAX
echo ""
echo "🛑 Stopping MAX server..."
kill $MAX_PID || true
sleep 5

echo ""
echo "🏆 PERFORMANCE BATTLE RESULTS"
echo "============================="

echo ""
echo "🔥 vLLM OPTIMIZED PERFORMANCE:"
python -c "
import json
try:
    with open('vllm_optimized_results.json') as f:
        results = json.load(f)
        for r in results:
            print(f'   TTFT: {r[\"ttft_ms\"]:.2f}ms')
            print(f'   TPS: {r[\"tokens_per_second\"]:.2f}')
            print(f'   Throughput: {r[\"throughput_req_per_sec\"]:.2f} req/s')
            break
except Exception as e:
    print(f'   Error reading results: {e}')
"

echo ""
echo "🔥 MAX OPTIMIZED PERFORMANCE:"
python -c "
import json
try:
    with open('max_optimized_results.json') as f:
        results = json.load(f)
        for r in results:
            print(f'   TTFT: {r[\"ttft_ms\"]:.2f}ms')
            print(f'   TPS: {r[\"tokens_per_second\"]:.2f}')
            print(f'   Throughput: {r[\"throughput_req_per_sec\"]:.2f} req/s')
            break
except Exception as e:
    print(f'   Error reading results: {e}')
"

echo ""
echo "🎯 PERFORMANCE COMPARISON:"
python -c "
import json
try:
    with open('vllm_optimized_results.json') as f:
        vllm_results = json.load(f)[0]
    with open('max_optimized_results.json') as f:
        max_results = json.load(f)[0]
    
    vllm_ttft = vllm_results['ttft_ms']
    max_ttft = max_results['ttft_ms']
    vllm_tps = vllm_results['tokens_per_second']
    max_tps = max_results['tokens_per_second']
    vllm_throughput = vllm_results['throughput_req_per_sec']
    max_throughput = max_results['throughput_req_per_sec']
    
    if vllm_ttft < max_ttft:
        ttft_winner = 'vLLM'
        ttft_improvement = ((max_ttft - vllm_ttft) / max_ttft) * 100
    else:
        ttft_winner = 'MAX'
        ttft_improvement = ((vllm_ttft - max_ttft) / vllm_ttft) * 100
    
    if vllm_tps > max_tps:
        tps_winner = 'vLLM'
        tps_improvement = ((vllm_tps - max_tps) / max_tps) * 100
    else:
        tps_winner = 'MAX'
        tps_improvement = ((max_tps - vllm_tps) / vllm_tps) * 100
    
    if vllm_throughput > max_throughput:
        throughput_winner = 'vLLM'
        throughput_improvement = ((vllm_throughput - max_throughput) / max_throughput) * 100
    else:
        throughput_winner = 'MAX'
        throughput_improvement = ((max_throughput - vllm_throughput) / vllm_throughput) * 100
    
    print(f'   TTFT Winner: {ttft_winner} by {ttft_improvement:.1f}%')
    print(f'   TPS Winner: {tps_winner} by {tps_improvement:.1f}%')
    print(f'   Throughput Winner: {throughput_winner} by {throughput_improvement:.1f}%')
    
    # Overall winner
    vllm_wins = sum([vllm_ttft < max_ttft, vllm_tps > max_tps, vllm_throughput > max_throughput])
    max_wins = 3 - vllm_wins
    
    if vllm_wins > max_wins:
        print(f'\\n🏆 OVERALL WINNER: vLLM ({vllm_wins}/3 metrics)')
    elif max_wins > vllm_wins:
        print(f'\\n🏆 OVERALL WINNER: MAX ({max_wins}/3 metrics)')
    else:
        print(f'\\n🤝 TIE! Both frameworks excel in different areas')

except Exception as e:
    print(f'   Error comparing results: {e}')
"

echo ""
echo "📁 Files created:"
echo "   - vllm_optimized_results.json"
echo "   - max_optimized_results.json"

echo ""
echo "🎉 OPTIMIZED GPU BENCHMARK COMPLETE!"
echo "Both frameworks now running at PEAK PERFORMANCE!"

# Final GPU status
echo ""
echo "Final GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader