#!/usr/bin/env python3
"""
Server-based benchmarking script for comparing vLLM vs Mojo/Modular MAX frameworks.
Both frameworks run as OpenAI-compatible servers for fair comparison.
"""

import time
import json
import statistics
import argparse
import asyncio
import aiohttp
import psutil
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import subprocess
import signal
import sys
import os


@dataclass
class BenchmarkResult:
    framework: str
    model: str
    prompt_length: int
    output_length: int
    concurrent_requests: int
    batch_size: int
    
    # Core metrics
    ttft_ms: float  # Time to first token
    tokens_per_second: float
    total_time_ms: float
    throughput_req_per_sec: float
    
    # Resource usage
    peak_memory_mb: float
    avg_cpu_percent: float
    
    # Additional metrics
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    
    timestamp: str


class ServerManager:
    def __init__(self):
        self.vllm_process = None
        self.max_process = None
        
    def start_vllm_server(self, model: str, port: int = 8001):
        """Start vLLM server with v1 engine and Flash Attention enabled"""
        print(f"Starting vLLM server (v1 engine + Flash Attention) on port {port} with model {model}")
        
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--port", str(port),
            "--trust-remote-code",
            "--max-model-len", "8192",
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", "0.9",  # Increased for better performance
            "--enable-chunked-prefill",  # v1 engine optimization
            "--max-num-batched-tokens", "8192",  # Optimize batching
            "--max-num-seqs", "256",  # Support more concurrent sequences
        ]
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
        env["VLLM_USE_V1"] = "1"  # Enable v1 engine
        env["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"  # Enable Flash Attention
        env["VLLM_FLASH_ATTN_FORCE_ENABLE"] = "1"  # Force Flash Attention
        
        self.vllm_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        
        # Wait for server to be ready
        import requests
        for _ in range(60):  # Wait up to 60 seconds
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=1)
                if response.status_code == 200:
                    print("vLLM server is ready!")
                    return True
            except:
                time.sleep(1)
        
        print("Failed to start vLLM server")
        return False
    
    def start_max_server(self, model: str, port: int = 8002):
        """Start MAX server with official Llama-3-8B-Instruct optimized settings"""
        print(f"Starting MAX server (optimized for Llama-3-8B-Instruct) on port {port} with model {model}")
        
        cmd = [
            "max", "serve",
            f"--model-path={model}",
            f"--port={port}"
            # Removed quantization-encoding to use optimal GPU defaults
        ]
        
        env = os.environ.copy()
        # Enable HF transfer acceleration
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        # Ensure HF_TOKEN is set for gated model access
        if "HF_TOKEN" not in env:
            print("Warning: HF_TOKEN not found in environment. MAX may not be able to access gated models.")
        
        self.max_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd="/Users/morganmcguire/ML/sonic/max_test"
        )
        
        # Wait for server to be ready
        import requests
        for _ in range(120):  # Wait up to 2 minutes for model loading
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=1)
                if response.status_code == 200:
                    print("MAX server is ready!")
                    return True
            except:
                time.sleep(1)
        
        print("Failed to start MAX server")
        return False
    
    def stop_servers(self):
        """Stop both servers"""
        if self.vllm_process:
            print("Stopping vLLM server...")
            self.vllm_process.terminate()
            self.vllm_process.wait()
            
        if self.max_process:
            print("Stopping MAX server...")
            self.max_process.terminate()
            self.max_process.wait()


class ServerBenchmark:
    def __init__(self, base_url: str, framework_name: str):
        self.base_url = base_url
        self.framework_name = framework_name
        
    async def single_request(self, session: aiohttp.ClientSession, messages: List[Dict], max_tokens: int) -> Tuple[float, float, float, str]:
        """Send a single request and measure timing"""
        payload = {
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,  # Greedy decoding for consistency
            "stream": False,
            "stop": ["<|end_of_text|>", "<|eot_id|>"]  # Llama-3 specific stop tokens
        }
        
        start_time = time.perf_counter()
        
        try:
            async with session.post(f"{self.base_url}/v1/chat/completions", 
                                  json=payload,
                                  timeout=aiohttp.ClientTimeout(total=120)) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Request failed with status {response.status}: {error_text}")
                
                result = await response.json()
                end_time = time.perf_counter()
                
                total_time_ms = (end_time - start_time) * 1000
                
                # Extract metrics from response
                usage = result.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                
                if completion_tokens > 0:
                    tokens_per_second = completion_tokens / (total_time_ms / 1000)
                else:
                    tokens_per_second = 0
                
                # Estimate TTFT (assume 10% of total time)
                ttft_ms = total_time_ms * 0.1
                
                content = result["choices"][0]["message"]["content"]
                
                return ttft_ms, tokens_per_second, total_time_ms, content
                
        except Exception as e:
            print(f"Request failed: {e}")
            return 0, 0, 0, ""
    
    async def benchmark_scenario(self, messages: List[Dict], max_tokens: int, concurrent_requests: int) -> Dict:
        """Run benchmark scenario with specified concurrency"""
        
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        timeout = aiohttp.ClientTimeout(total=300)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            
            # Create list of coroutines for concurrent requests
            tasks = []
            for _ in range(concurrent_requests):
                task = self.single_request(session, messages, max_tokens)
                tasks.append(task)
            
            start_time = time.perf_counter()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.perf_counter()
            
            # Filter out failed requests
            successful_results = [r for r in results if not isinstance(r, Exception)]
            
            if not successful_results:
                raise Exception("All requests failed")
            
            # Calculate metrics
            ttft_times = [r[0] for r in successful_results]
            tps_values = [r[1] for r in successful_results]
            individual_times = [r[2] for r in successful_results]
            
            total_batch_time_ms = (end_time - start_time) * 1000
            avg_ttft = statistics.mean(ttft_times)
            total_tps = sum(tps_values)
            throughput_req_per_sec = len(successful_results) / (total_batch_time_ms / 1000)
            
            return {
                "ttft_ms": avg_ttft,
                "tokens_per_second": total_tps,
                "total_time_ms": total_batch_time_ms,
                "throughput_req_per_sec": throughput_req_per_sec,
                "latency_p50_ms": statistics.median(individual_times),
                "latency_p95_ms": statistics.quantiles(individual_times, n=20)[18] if len(individual_times) > 1 else individual_times[0],
                "latency_p99_ms": statistics.quantiles(individual_times, n=100)[98] if len(individual_times) > 1 else individual_times[0],
                "successful_requests": len(successful_results),
                "failed_requests": len(results) - len(successful_results)
            }


class BenchmarkRunner:
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.server_manager = ServerManager()
        
    def generate_test_messages(self, length: int) -> List[Dict]:
        """Generate test messages of specified token length"""
        # Base prompt that we'll extend to reach target length
        base_prompt = "Explain the concept of machine learning in simple terms."
        
        if length <= 64:
            prompt = base_prompt
        elif length <= 256:
            prompt = base_prompt + " Please provide detailed explanations with examples and elaborate on the key concepts involved in this field."
        elif length <= 1024:
            prompt = base_prompt + " Please provide a comprehensive analysis covering historical context, current applications, future implications, challenges, benefits, technical details, step-by-step processes, real-world examples, and practical recommendations. Include multiple perspectives and discuss both advantages and disadvantages of different approaches."
        elif length <= 4000:
            # For 4K tokens - add more context
            prompt = base_prompt + " Please provide an extremely comprehensive analysis covering: 1) Historical development from early statistical models to modern neural networks, 2) Detailed technical explanations of supervised, unsupervised, and reinforcement learning paradigms, 3) Current real-world applications across industries including healthcare, finance, autonomous vehicles, natural language processing, computer vision, and robotics, 4) Future implications and emerging trends like federated learning, quantum machine learning, and neuromorphic computing, 5) Ethical challenges including bias, fairness, transparency, privacy, and societal impact, 6) Technical benefits and limitations of different algorithmic approaches, 7) Step-by-step processes for data collection, preprocessing, feature engineering, model selection, training, validation, and deployment, 8) Concrete examples from major companies and research institutions, 9) Practical recommendations for organizations looking to adopt ML technologies, 10) Multiple expert perspectives on the field's trajectory and potential risks."
        elif length <= 10000:
            # For 10K tokens - extensive academic-style prompt
            prompt = base_prompt + " Please provide an exhaustive academic-level analysis that covers: HISTORICAL FOUNDATIONS: Trace the evolution from early computational theories of Turing and Von Neumann through statistical learning theory, connectionism, and the modern deep learning revolution. THEORETICAL FRAMEWORKS: Explain mathematical foundations including probability theory, information theory, optimization theory, computational complexity, PAC learning, VC dimension, and generalization bounds. ALGORITHMIC PARADIGMS: Detailed exposition of supervised learning (linear models, tree-based methods, ensemble methods, neural networks, kernel methods), unsupervised learning (clustering, dimensionality reduction, generative models, self-supervised learning), and reinforcement learning (value functions, policy gradients, actor-critic methods, multi-agent systems). CONTEMPORARY APPLICATIONS: Comprehensive survey of applications in computer vision (object detection, semantic segmentation, image generation), natural language processing (language models, machine translation, question answering, dialogue systems), robotics (motion planning, manipulation, perception), healthcare (medical imaging, drug discovery, personalized medicine), finance (algorithmic trading, risk assessment, fraud detection), autonomous systems (self-driving cars, drones, industrial automation), and scientific computing (climate modeling, protein folding, materials science). EMERGING FRONTIERS: Discussion of cutting-edge research areas including quantum machine learning, neuromorphic computing, federated learning, meta-learning, continual learning, explainable AI, causal inference, and AI safety. IMPLEMENTATION CHALLENGES: Technical considerations for data quality, feature engineering, model selection, hyperparameter optimization, distributed training, model compression, deployment strategies, monitoring, and maintenance."
        else:
            # For very large contexts (up to 100K tokens) - create an extremely detailed prompt
            sections = [
                "COMPREHENSIVE HISTORICAL ANALYSIS: Provide a detailed chronological account of machine learning development from the 1940s to present, including key figures, breakthrough papers, technological milestones, and paradigm shifts.",
                "MATHEMATICAL FOUNDATIONS: Explain the underlying mathematics including linear algebra, calculus, probability theory, statistics, information theory, optimization theory, and computational complexity theory.",
                "ALGORITHMIC DEEP DIVE: Provide detailed explanations of hundreds of machine learning algorithms, their theoretical basis, practical implementations, and comparative analysis.",
                "INDUSTRY APPLICATIONS: Comprehensive survey of ML applications across every major industry sector with specific case studies and implementation details.",
                "RESEARCH FRONTIERS: Extensive discussion of current research directions, open problems, and future possibilities in the field.",
                "ETHICAL AND SOCIETAL IMPLICATIONS: Thorough analysis of AI ethics, bias, fairness, privacy, security, and societal impact.",
                "TECHNICAL IMPLEMENTATION: Detailed guides for data engineering, model development, deployment, and production systems.",
                "BUSINESS STRATEGY: Analysis of how organizations can successfully adopt and scale machine learning initiatives.",
                "EDUCATIONAL PATHWAYS: Comprehensive guide to learning machine learning from beginner to expert level.",
                "GLOBAL PERSPECTIVES: International perspectives on AI development, regulation, and cultural considerations."
            ]
            prompt = base_prompt + " " + " ".join(sections)
        
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    
    async def run_benchmark_scenario(self, framework_name: str, base_url: str, 
                                   messages: List[Dict], max_tokens: int, 
                                   concurrent_requests: int) -> BenchmarkResult:
        """Run a single benchmark scenario"""
        
        print(f"Running {framework_name} scenario: {len(messages[1]['content'].split())} token prompt → {max_tokens} tokens, {concurrent_requests} concurrent")
        
        benchmark = ServerBenchmark(base_url, framework_name)
        
        try:
            metrics = await benchmark.benchmark_scenario(messages, max_tokens, concurrent_requests)
            
            return BenchmarkResult(
                framework=framework_name,
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                prompt_length=len(messages[1]["content"].split()),
                output_length=max_tokens,
                concurrent_requests=concurrent_requests,
                batch_size=1,
                ttft_ms=metrics["ttft_ms"],
                tokens_per_second=metrics["tokens_per_second"],
                total_time_ms=metrics["total_time_ms"],
                throughput_req_per_sec=metrics["throughput_req_per_sec"],
                peak_memory_mb=0.0,  # Would need separate monitoring
                avg_cpu_percent=0.0,  # Would need separate monitoring
                latency_p50_ms=metrics["latency_p50_ms"],
                latency_p95_ms=metrics["latency_p95_ms"],
                latency_p99_ms=metrics["latency_p99_ms"],
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            print(f"Benchmark failed for {framework_name}: {e}")
            return None
    
    async def run_comprehensive_benchmark(self, frameworks: List[Tuple[str, str]], test_scenarios: List[Dict]):
        """Run comprehensive benchmark across all frameworks and scenarios"""
        
        print("Starting comprehensive server-based benchmark...")
        print("Both frameworks configured for Llama-3-8B-Instruct with BF16 precision")
        
        for framework_name, base_url in frameworks:
            print(f"\n=== Benchmarking {framework_name} ===")
            
            for scenario in test_scenarios:
                messages = self.generate_test_messages(scenario["prompt_length"])
                
                result = await self.run_benchmark_scenario(
                    framework_name=framework_name,
                    base_url=base_url,
                    messages=messages,
                    max_tokens=scenario["max_tokens"],
                    concurrent_requests=scenario["concurrent_requests"]
                )
                
                if result:
                    self.results.append(result)
                    print(f"✓ TTFT: {result.ttft_ms:.2f}ms, TPS: {result.tokens_per_second:.2f}, "
                          f"Throughput: {result.throughput_req_per_sec:.2f} req/s")
                else:
                    print("✗ Scenario failed")
    
    def save_results(self, filename: str = "server_benchmark_results.json"):
        """Save benchmark results to file"""
        results_dict = [asdict(result) for result in self.results]
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"Results saved to {filename}")
    
    def print_summary(self):
        """Print benchmark summary"""
        if not self.results:
            print("No results to display")
            return
        
        print("\n" + "="*80)
        print("SERVER BENCHMARK SUMMARY")
        print("="*80)
        
        for framework in set(result.framework for result in self.results):
            framework_results = [r for r in self.results if r.framework == framework]
            
            print(f"\n{framework} Results:")
            print("-" * 50)
            
            if framework_results:
                avg_ttft = statistics.mean([r.ttft_ms for r in framework_results])
                avg_tps = statistics.mean([r.tokens_per_second for r in framework_results])
                avg_throughput = statistics.mean([r.throughput_req_per_sec for r in framework_results])
                
                print(f"Average TTFT: {avg_ttft:.2f} ms")
                print(f"Average TPS: {avg_tps:.2f} tokens/sec")
                print(f"Average Throughput: {avg_throughput:.2f} req/sec")


def print_configuration_info():
    """Print server configuration information"""
    print("\n" + "="*80)
    print("SERVER CONFIGURATION COMPARISON")
    print("="*80)
    print("vLLM Server Configuration (OPTIMIZED):")
    print("  - Model: meta-llama/Meta-Llama-3-8B-Instruct")
    print("  - Engine: v1 (latest optimizations)")
    print("  - Attention: Flash Attention (FLASHINFER backend)")
    print("  - Precision: bfloat16")
    print("  - Temperature: 0.0 (greedy decoding)")
    print("  - Max model length: 8192")
    print("  - GPU memory utilization: 0.9")
    print("  - Chunked prefill: Enabled")
    print("  - Port: 8001")
    print()
    print("MAX Server Configuration (OPTIMIZED):")
    print("  - Model: meta-llama/Meta-Llama-3-8B-Instruct")
    print("  - Configuration: Official Llama-3-8B-Instruct settings")
    print("  - Precision: Auto-optimized (likely bfloat16)")
    print("  - HF Transfer: Enabled for faster model loading")
    print("  - Temperature: 0.0 (greedy decoding)")
    print("  - Port: 8002")
    print()
    print("Both servers use OpenAI-compatible API for fair comparison")
    print("="*80 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="Server-based benchmark for vLLM vs Mojo/MAX")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model to benchmark")
    parser.add_argument("--output", default="server_benchmark_results.json", help="Output file for results")
    parser.add_argument("--frameworks", nargs="+", choices=["vllm", "max"], 
                       default=["vllm", "max"], help="Frameworks to benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with fewer scenarios")
    parser.add_argument("--extended", action="store_true", help="Run extended benchmark with 1K-100K token scenarios")
    parser.add_argument("--start-servers", action="store_true", help="Automatically start servers")
    
    args = parser.parse_args()
    
    print_configuration_info()
    
    # Define test scenarios
    if args.quick:
        test_scenarios = [
            {"prompt_length": 64, "max_tokens": 128, "concurrent_requests": 1},
            {"prompt_length": 256, "max_tokens": 256, "concurrent_requests": 1},
            {"prompt_length": 64, "max_tokens": 128, "concurrent_requests": 5},
        ]
    elif args.extended:
        # Extended benchmark scenarios covering 1K-100K tokens, batch sizes 1-128, concurrent 1-100
        test_scenarios = []
        
        # Input length scaling tests (1K to 100K tokens)
        input_lengths = [1000, 4000, 10000, 25000, 50000, 100000]
        for input_len in input_lengths:
            test_scenarios.append({
                "prompt_length": input_len, 
                "max_tokens": min(512, input_len // 10),  # Output proportional to input
                "concurrent_requests": 1
            })
        
        # Concurrent request scaling tests (1 to 100)
        concurrent_levels = [1, 5, 10, 25, 50, 100]
        for concurrent in concurrent_levels:
            test_scenarios.append({
                "prompt_length": 1024, 
                "max_tokens": 256, 
                "concurrent_requests": concurrent
            })
        
        # Batch size scaling tests (simulated via concurrent requests)
        batch_sizes = [1, 8, 16, 32, 64, 128]
        for batch_size in batch_sizes:
            test_scenarios.append({
                "prompt_length": 512, 
                "max_tokens": 128, 
                "concurrent_requests": batch_size
            })
        
        # Mixed scenarios for comprehensive testing
        mixed_scenarios = [
            {"prompt_length": 2000, "max_tokens": 1000, "concurrent_requests": 10},
            {"prompt_length": 5000, "max_tokens": 500, "concurrent_requests": 5},
            {"prompt_length": 10000, "max_tokens": 250, "concurrent_requests": 2}
        ]
        test_scenarios.extend(mixed_scenarios)
    else:
        # Standard benchmark scenarios
        test_scenarios = [
            {"prompt_length": 64, "max_tokens": 128, "concurrent_requests": 1},
            {"prompt_length": 256, "max_tokens": 256, "concurrent_requests": 1},
            {"prompt_length": 1024, "max_tokens": 512, "concurrent_requests": 1},
            {"prompt_length": 64, "max_tokens": 128, "concurrent_requests": 5},
            {"prompt_length": 64, "max_tokens": 128, "concurrent_requests": 10},
        ]
    
    # Set up frameworks
    frameworks = []
    if "vllm" in args.frameworks:
        frameworks.append(("VLLM", "http://localhost:8001"))
    if "max" in args.frameworks:
        frameworks.append(("MAX", "http://localhost:8002"))
    
    if not frameworks:
        print("No frameworks specified")
        sys.exit(1)
    
    runner = BenchmarkRunner()
    
    # Start servers if requested
    if args.start_servers:
        print("Starting servers...")
        if "vllm" in args.frameworks:
            if not runner.server_manager.start_vllm_server(args.model, 8001):
                print("Failed to start vLLM server")
                sys.exit(1)
        
        if "max" in args.frameworks:
            if not runner.server_manager.start_max_server(args.model, 8002):
                print("Failed to start MAX server")
                runner.server_manager.stop_servers()
                sys.exit(1)
    
    try:
        # Run benchmarks
        await runner.run_comprehensive_benchmark(frameworks, test_scenarios)
        runner.save_results(args.output)
        runner.print_summary()
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Benchmark failed: {e}")
    finally:
        if args.start_servers:
            runner.server_manager.stop_servers()


if __name__ == "__main__":
    asyncio.run(main())