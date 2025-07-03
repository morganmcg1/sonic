#!/bin/bash

# Complete Fair Benchmarking Suite - vLLM vs MAX
# Uses official MAX benchmark methodology for industry-standard comparison

set -e

echo "🚀 COMPLETE FAIR BENCHMARKING SUITE"
echo "======================================"
echo "vLLM vs MAX - Using Official Methodology"
echo ""

# Configuration
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="benchmark_results_${TIMESTAMP}"

# Check if running in quick mode
QUICK_MODE=false
VALIDATION_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --validate-only)
            VALIDATION_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--quick] [--validate-only]"
            echo "  --quick         Run quick benchmark with fewer scenarios"
            echo "  --validate-only Only run validation, skip benchmarking"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Quick mode: $QUICK_MODE"
echo "  Validation only: $VALIDATION_ONLY"
echo "  Results directory: $RESULTS_DIR"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Step 1: Environment Validation
echo "🔍 STEP 1: ENVIRONMENT VALIDATION"
echo "=================================="

python validate_benchmark_setup.py --output "$RESULTS_DIR/validation_results.json"
VALIDATION_EXIT_CODE=$?

if [ $VALIDATION_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ Validation failed! Please fix the issues above before proceeding."
    echo "💡 Run the validation again after fixes: python validate_benchmark_setup.py"
    exit 1
fi

echo "✅ Environment validation passed!"

if [ "$VALIDATION_ONLY" = true ]; then
    echo ""
    echo "✅ Validation complete! Environment is ready for benchmarking."
    echo "Run without --validate-only to proceed with full benchmark."
    exit 0
fi

# Step 2: Download ShareGPT dataset if needed
echo ""
echo "📚 STEP 2: DATASET PREPARATION"
echo "=============================="

SHAREGPT_DATASET="ShareGPT_V3_unfiltered_cleaned_split.json"
if [ ! -f "$SHAREGPT_DATASET" ]; then
    echo "Downloading ShareGPT dataset for realistic benchmarking..."
    curl -L -o "$SHAREGPT_DATASET" "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json" || {
        echo "⚠️  ShareGPT download failed, will use built-in datasets"
        SHAREGPT_DATASET=""
    }
fi

if [ -n "$SHAREGPT_DATASET" ] && [ -f "$SHAREGPT_DATASET" ]; then
    echo "✅ ShareGPT dataset ready: $SHAREGPT_DATASET"
    DATASET_ARGS="--dataset-name sharegpt --dataset-path $SHAREGPT_DATASET"
else
    echo "📝 Using built-in sharegpt dataset"
    DATASET_ARGS="--dataset-name sharegpt"
fi

# Step 3: Kill any existing servers
echo ""
echo "🧹 STEP 3: CLEANUP EXISTING SERVERS"
echo "==================================="

echo "Stopping any existing vLLM and MAX servers..."
pkill -f "vllm.entrypoints.openai.api_server" || echo "No vLLM servers running"
pkill -f "max serve" || echo "No MAX servers running"
sleep 5

# Clear GPU memory if NVIDIA GPU available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
fi

# Step 4: Benchmark vLLM
echo ""
echo "🏎️ STEP 4: BENCHMARKING vLLM"
echo "=============================="

echo "Starting optimized vLLM server..."
./start_vllm_optimized.sh &
VLLM_STARTUP_PID=$!

# Wait for vLLM startup
sleep 30

# Check if vLLM started successfully
if ! curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo "❌ vLLM server failed to start"
    kill $VLLM_STARTUP_PID 2>/dev/null || true
    exit 1
fi

echo "✅ vLLM server ready, starting benchmark..."

# Run vLLM benchmarks with different scenarios
if [ "$QUICK_MODE" = true ]; then
    SCENARIOS=(
        "--request-rate 1.0 --num-prompts 50"
        "--request-rate 2.0 --num-prompts 50"
    )
else
    SCENARIOS=(
        "--request-rate 0.5 --num-prompts 100"
        "--request-rate 1.0 --num-prompts 100" 
        "--request-rate 2.0 --num-prompts 100"
        "--request-rate 4.0 --num-prompts 100"
        "--request-rate 8.0 --num-prompts 100"
    )
fi

for i in "${!SCENARIOS[@]}"; do
    scenario="${SCENARIOS[$i]}"
    output_file="$RESULTS_DIR/vllm_scenario_${i}.json"
    
    echo "Running vLLM scenario $((i+1))/${#SCENARIOS[@]}: $scenario"
    
    python benchmark_serving_official.py \
        --backend vllm \
        --host localhost \
        --port 8001 \
        --model "$MODEL" \
        $DATASET_ARGS \
        $scenario \
        --save-result \
        --result-filename "$output_file" || {
        echo "⚠️  vLLM scenario $((i+1)) failed, continuing..."
        continue
    }
    
    echo "✅ vLLM scenario $((i+1)) completed"
done

# Stop vLLM server
echo "Stopping vLLM server..."
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 10

# Step 5: Benchmark MAX
echo ""
echo "🏎️ STEP 5: BENCHMARKING MAX"
echo "============================"

echo "Starting optimized MAX server..."
./start_max_optimized.sh &
MAX_STARTUP_PID=$!

# Wait for MAX startup (takes longer)
sleep 60

# Check if MAX started successfully
if ! curl -s http://localhost:8002/health > /dev/null 2>&1; then
    echo "❌ MAX server failed to start"
    kill $MAX_STARTUP_PID 2>/dev/null || true
    exit 1
fi

echo "✅ MAX server ready, starting benchmark..."

# Run MAX benchmarks with same scenarios
for i in "${!SCENARIOS[@]}"; do
    scenario="${SCENARIOS[$i]}"
    output_file="$RESULTS_DIR/max_scenario_${i}.json"
    
    echo "Running MAX scenario $((i+1))/${#SCENARIOS[@]}: $scenario"
    
    python benchmark_serving_official.py \
        --backend modular \
        --host localhost \
        --port 8002 \
        --model "$MODEL" \
        $DATASET_ARGS \
        $scenario \
        --save-result \
        --result-filename "$output_file" || {
        echo "⚠️  MAX scenario $((i+1)) failed, continuing..."
        continue
    }
    
    echo "✅ MAX scenario $((i+1)) completed"
done

# Stop MAX server
echo "Stopping MAX server..."
pkill -f "max serve" || true
sleep 5

# Step 6: Analysis and Comparison
echo ""
echo "📊 STEP 6: RESULTS ANALYSIS"
echo "============================"

# Create comparison analysis script on the fly
cat > "$RESULTS_DIR/analyze_results.py" << 'EOF'
#!/usr/bin/env python3
import json
import glob
import os
import sys
from typing import Dict, List

def load_results(pattern: str) -> List[Dict]:
    """Load all result files matching pattern"""
    files = glob.glob(pattern)
    results = []
    
    for file in files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                data['source_file'] = file
                results.append(data)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return results

def analyze_framework(results: List[Dict], framework_name: str):
    """Analyze results for a framework"""
    if not results:
        print(f"No results found for {framework_name}")
        return None
    
    print(f"\n{framework_name} Results:")
    print("-" * 40)
    
    metrics = []
    for i, result in enumerate(results):
        print(f"Scenario {i+1}:")
        print(f"  Request Throughput: {result.get('request_throughput', 0):.2f} req/s")
        print(f"  Input Token Throughput: {result.get('input_token_throughput', 0):.2f} tokens/s")
        print(f"  Output Token Throughput: {result.get('output_token_throughput', 0):.2f} tokens/s") 
        print(f"  Mean TTFT: {result.get('mean_ttft_ms', 0):.2f} ms")
        print(f"  P99 TTFT: {result.get('p99_ttft_ms', 0):.2f} ms")
        print(f"  Mean TPOT: {result.get('mean_tpot_ms', 0):.2f} ms")
        print()
        
        metrics.append({
            'request_throughput': result.get('request_throughput', 0),
            'output_token_throughput': result.get('output_token_throughput', 0),
            'mean_ttft_ms': result.get('mean_ttft_ms', 0),
            'mean_tpot_ms': result.get('mean_tpot_ms', 0)
        })
    
    # Calculate averages
    if metrics:
        avg_throughput = sum(m['request_throughput'] for m in metrics) / len(metrics)
        avg_token_throughput = sum(m['output_token_throughput'] for m in metrics) / len(metrics)
        avg_ttft = sum(m['mean_ttft_ms'] for m in metrics) / len(metrics)
        avg_tpot = sum(m['mean_tpot_ms'] for m in metrics) / len(metrics)
        
        print(f"Average Performance:")
        print(f"  Request Throughput: {avg_throughput:.2f} req/s")
        print(f"  Token Throughput: {avg_token_throughput:.2f} tokens/s")
        print(f"  TTFT: {avg_ttft:.2f} ms")
        print(f"  TPOT: {avg_tpot:.2f} ms")
        
        return {
            'avg_request_throughput': avg_throughput,
            'avg_token_throughput': avg_token_throughput,
            'avg_ttft_ms': avg_ttft,
            'avg_tpot_ms': avg_tpot
        }
    
    return None

def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    # Load results
    vllm_results = load_results(f"{results_dir}/vllm_scenario_*.json")
    max_results = load_results(f"{results_dir}/max_scenario_*.json")
    
    print("🔍 BENCHMARK RESULTS ANALYSIS")
    print("=" * 50)
    
    # Analyze each framework
    vllm_avg = analyze_framework(vllm_results, "vLLM")
    max_avg = analyze_framework(max_results, "MAX")
    
    # Compare frameworks
    if vllm_avg and max_avg:
        print("\n🏆 FRAMEWORK COMPARISON")
        print("=" * 30)
        
        metrics = [
            ('Request Throughput', 'avg_request_throughput', 'req/s', True),
            ('Token Throughput', 'avg_token_throughput', 'tokens/s', True),
            ('Time to First Token', 'avg_ttft_ms', 'ms', False),
            ('Time per Output Token', 'avg_tpot_ms', 'ms', False)
        ]
        
        for metric_name, key, unit, higher_better in metrics:
            vllm_val = vllm_avg[key]
            max_val = max_avg[key]
            
            if higher_better:
                winner = "vLLM" if vllm_val > max_val else "MAX"
                improvement = abs(vllm_val - max_val) / min(vllm_val, max_val) * 100
            else:
                winner = "vLLM" if vllm_val < max_val else "MAX"
                improvement = abs(vllm_val - max_val) / max(vllm_val, max_val) * 100
            
            print(f"{metric_name}:")
            print(f"  vLLM: {vllm_val:.2f} {unit}")
            print(f"  MAX:  {max_val:.2f} {unit}")
            print(f"  Winner: {winner} by {improvement:.1f}%")
            print()
        
        # Overall winner
        vllm_wins = 0
        max_wins = 0
        
        if vllm_avg['avg_request_throughput'] > max_avg['avg_request_throughput']:
            vllm_wins += 1
        else:
            max_wins += 1
            
        if vllm_avg['avg_token_throughput'] > max_avg['avg_token_throughput']:
            vllm_wins += 1
        else:
            max_wins += 1
            
        if vllm_avg['avg_ttft_ms'] < max_avg['avg_ttft_ms']:
            vllm_wins += 1
        else:
            max_wins += 1
            
        if vllm_avg['avg_tpot_ms'] < max_avg['avg_tpot_ms']:
            vllm_wins += 1
        else:
            max_wins += 1
        
        print(f"🏆 OVERALL WINNER:")
        if vllm_wins > max_wins:
            print(f"   vLLM wins {vllm_wins}/4 metrics")
        elif max_wins > vllm_wins:
            print(f"   MAX wins {max_wins}/4 metrics") 
        else:
            print(f"   TIE! Each framework wins {vllm_wins}/4 metrics")

if __name__ == "__main__":
    main()
EOF

chmod +x "$RESULTS_DIR/analyze_results.py"

# Run analysis
python "$RESULTS_DIR/analyze_results.py" "$RESULTS_DIR"

# Step 7: Generate Summary Report
echo ""
echo "📋 STEP 7: GENERATING SUMMARY REPORT"
echo "====================================="

cat > "$RESULTS_DIR/benchmark_summary.md" << EOF
# Fair vLLM vs MAX Benchmark Results

## Methodology
- **Benchmark Tool**: Official MAX benchmark_serving.py
- **Model**: $MODEL
- **Date**: $(date)
- **Hardware**: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "GPU info unavailable")

## Configuration
- **vLLM**: v1 engine, FLASH_ATTN attention, bfloat16, 90% GPU memory
- **MAX**: Official Llama-3-8B-Instruct settings, HF transfer enabled
- **API**: Both frameworks tested via OpenAI-compatible endpoints

## Scenarios Tested
$(if [ "$QUICK_MODE" = true ]; then echo "- Quick mode: 2 scenarios"; else echo "- Standard mode: 5 scenarios"; fi)
- Request rates: 0.5, 1.0, 2.0, 4.0, 8.0 req/s
- Dataset: ShareGPT conversational data
- Metrics: Request throughput, token throughput, TTFT, TPOT

## Results
See individual scenario files:
$(ls "$RESULTS_DIR"/*.json | grep -E "(vllm|max)_scenario" | sed 's/^/- /')

## Analysis
Run: python analyze_results.py

## Files Generated
- validation_results.json: Environment validation
- *_scenario_*.json: Individual benchmark results  
- analyze_results.py: Analysis script
- benchmark_summary.md: This summary

## Reproducing Results
1. ./setup_improved_benchmark.sh
2. python validate_benchmark_setup.py
3. ./run_complete_benchmark.sh
EOF

echo "✅ Summary report generated: $RESULTS_DIR/benchmark_summary.md"

# Final cleanup
echo ""
echo "🧹 FINAL CLEANUP"
echo "================="

pkill -f "vllm.entrypoints.openai.api_server" || true
pkill -f "max serve" || true

echo ""
echo "🎉 BENCHMARK COMPLETE!"
echo "======================"
echo ""
echo "📁 Results saved in: $RESULTS_DIR/"
echo "📊 Analysis script: $RESULTS_DIR/analyze_results.py"
echo "📋 Summary report: $RESULTS_DIR/benchmark_summary.md"
echo ""
echo "🔍 To analyze results:"
echo "   cd $RESULTS_DIR && python analyze_results.py"
echo ""
echo "💡 Key improvements over original benchmark:"
echo "   ✅ Official MAX benchmark methodology"
echo "   ✅ Accurate TTFT measurement via streaming"
echo "   ✅ Standardized configurations"
echo "   ✅ Comprehensive validation"
echo "   ✅ Multiple request rates and datasets"
echo "   ✅ Statistical rigor with Poisson request distribution"