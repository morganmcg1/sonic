#!/usr/bin/env python3
"""
Comprehensive benchmarking script for comparing vLLM vs Mojo/Modular MAX frameworks.
Tests: Time to First Token (TTFT), Tokens per Second (TPS), and Throughput.
"""

import asyncio
import json
import time
import statistics
import psutil
import argparse
import traceback
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import subprocess
import tempfile
import shlex
import sys
import os

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available. Install with: pip install vllm")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not available. Install with: pip install requests")


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


class SystemMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.memory_samples = []
        self.cpu_samples = []
        self.monitoring = False

    def start_monitoring(self):
        self.monitoring = True
        self.memory_samples = []
        self.cpu_samples = []

    def stop_monitoring(self):
        self.monitoring = False

    def sample_resources(self):
        if self.monitoring:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent()
            self.memory_samples.append(memory_mb)
            self.cpu_samples.append(cpu_percent)

    def get_peak_memory(self) -> float:
        return max(self.memory_samples) if self.memory_samples else 0.0

    def get_avg_cpu(self) -> float:
        return statistics.mean(self.cpu_samples) if self.cpu_samples else 0.0


class VLLMBenchmark:
    def __init__(self, model_name: str = "allenai/OLMo-1B-hf"):
        self.model_name = model_name
        self.llm = None
        
    def setup(self):
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM not available")
        
        print(f"Loading vLLM model: {self.model_name}")
        try:
            self.llm = LLM(
                model=self.model_name,
                trust_remote_code=True,
                max_model_len=2048,
                gpu_memory_utilization=0.8 if self.has_gpu() else 0.0,
                tensor_parallel_size=1
                # Using vLLM's default precision (fp16) for optimal performance
            )
            print("vLLM model loaded successfully with default precision")
        except Exception as e:
            print(f"Error loading vLLM model: {e}")
            raise

    def has_gpu(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def benchmark_single_request(self, prompt: str, max_tokens: int, monitor: SystemMonitor) -> Tuple[float, float, float]:
        sampling_params = SamplingParams(
            temperature=0.0,  # Greedy decoding to match MAX's deterministic behavior
            max_tokens=max_tokens,
            top_p=1.0  # Disable nucleus sampling for deterministic output
        )
        
        monitor.start_monitoring()
        start_time = time.perf_counter()
        
        outputs = self.llm.generate([prompt], sampling_params)
        
        end_time = time.perf_counter()
        monitor.stop_monitoring()
        
        total_time_ms = (end_time - start_time) * 1000
        generated_tokens = len(outputs[0].outputs[0].token_ids) if outputs[0].outputs else 0
        tokens_per_second = generated_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0
        
        # Approximate TTFT as 10% of total time for single request
        ttft_ms = total_time_ms * 0.1
        
        return ttft_ms, tokens_per_second, total_time_ms

    def benchmark_batch(self, prompts: List[str], max_tokens: int, monitor: SystemMonitor) -> Tuple[float, float, float]:
        sampling_params = SamplingParams(
            temperature=0.0,  # Greedy decoding to match MAX's deterministic behavior
            max_tokens=max_tokens,
            top_p=1.0  # Disable nucleus sampling for deterministic output
        )
        
        monitor.start_monitoring()
        start_time = time.perf_counter()
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        end_time = time.perf_counter()
        monitor.stop_monitoring()
        
        total_time_ms = (end_time - start_time) * 1000
        total_tokens = sum(len(output.outputs[0].token_ids) if output.outputs else 0 for output in outputs)
        tokens_per_second = total_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0
        
        # Approximate TTFT
        ttft_ms = total_time_ms * 0.1
        
        return ttft_ms, tokens_per_second, total_time_ms


class MojoBenchmark:
    def __init__(self, model_name: str = "allenai/OLMo-1B-hf"):
        self.model_name = model_name
        self.max_test_dir = "/Users/morganmcguire/ML/sonic/max_test"
        
    def setup(self):
        print(f"Setting up Mojo/MAX for model: {self.model_name}")
        # Just verify MAX is available
        try:
            result = subprocess.run(
                ["pixi", "run", "max", "--version"],
                cwd=self.max_test_dir,
                capture_output=True,
                text=True,
                env={**os.environ, "PATH": f"{os.environ.get('HOME')}/.pixi/bin:{os.environ.get('PATH', '')}"}
            )
            if result.returncode == 0:
                print("Mojo/MAX setup successful")
            else:
                raise RuntimeError(f"MAX not available: {result.stderr}")
        except Exception as e:
            print(f"Error setting up Mojo/MAX: {e}")
            raise

    def cleanup(self):
        # No cleanup needed for direct MAX usage
        pass

    def run_max_generate(self, prompt: str, max_tokens: int) -> Tuple[float, str]:
        """Run MAX generate command and return timing and output."""
        # Create temporary file for prompt to avoid shell escaping issues
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(prompt)
            temp_prompt_file = f.name
        
        try:
            cmd = [
                "pixi", "run", "max", "generate",
                "--prompt", prompt,
                "--model-path", self.model_name,
                "--max-new-tokens", str(max_tokens),
                "--trust-remote-code",
                "--quantization-encoding", "float32"
            ]
            
            start_time = time.perf_counter()
            result = subprocess.run(
                cmd,
                cwd=self.max_test_dir,
                capture_output=True,
                text=True,
                env={**os.environ, "PATH": f"{os.environ.get('HOME')}/.pixi/bin:{os.environ.get('PATH', '')}"}
            )
            end_time = time.perf_counter()
            
            total_time_ms = (end_time - start_time) * 1000
            
            if result.returncode != 0:
                raise RuntimeError(f"MAX generate failed with code {result.returncode}: {result.stderr}")
            
            # MAX outputs to stdout, but check if we got actual generation output
            if not result.stdout.strip():
                raise RuntimeError(f"MAX generate produced no output. Stderr: {result.stderr}")
            
            return total_time_ms, result.stdout
            
        finally:
            os.unlink(temp_prompt_file)

    def parse_max_output(self, output: str) -> Dict[str, float]:
        """Parse MAX output to extract metrics."""
        metrics = {}
        lines = output.split('\n')
        
        for line in lines:
            if "Time to first token:" in line:
                try:
                    metrics['ttft_ms'] = float(line.split(':')[1].strip().replace(' ms', ''))
                except:
                    pass
            elif "Time per Output Token:" in line:
                try:
                    metrics['time_per_token_ms'] = float(line.split(':')[1].strip().replace(' ms', ''))
                except:
                    pass
            elif "Eval throughput (token-generation):" in line:
                try:
                    metrics['tokens_per_second'] = float(line.split(':')[1].strip().replace(' tokens per second', ''))
                except:
                    pass
            elif "Total Latency:" in line:
                try:
                    metrics['total_latency_ms'] = float(line.split(':')[1].strip().replace(' ms', ''))
                except:
                    pass
        
        return metrics

    def benchmark_single_request(self, prompt: str, max_tokens: int, monitor: SystemMonitor) -> Tuple[float, float, float]:
        monitor.start_monitoring()
        
        total_time_ms, output = self.run_max_generate(prompt, max_tokens)
        metrics = self.parse_max_output(output)
        
        # Extract metrics or use defaults
        ttft_ms = metrics.get('ttft_ms', total_time_ms * 0.1)  # Default to 10% of total time
        tokens_per_second = metrics.get('tokens_per_second', 0)
        
        # If we don't have TPS from output, calculate it roughly
        if tokens_per_second == 0:
            # Count words in first line of output (rough approximation)
            output_lines = output.split('\n')
            for line in output_lines:
                if line.strip() and not any(keyword in line.lower() for keyword in ['prompt', 'output', 'time', 'throughput', 'startup']):
                    word_count = len(line.split())
                    if word_count > 0:
                        tokens_per_second = word_count / (total_time_ms / 1000)
                    break
        
        monitor.stop_monitoring()
        return ttft_ms, tokens_per_second, total_time_ms

    def benchmark_batch(self, prompts: List[str], max_tokens: int, monitor: SystemMonitor) -> Tuple[float, float, float]:
        monitor.start_monitoring()
        start_time = time.perf_counter()
        
        total_tokens = 0
        ttft_times = []
        
        for prompt in prompts:
            total_time_ms, output = self.run_max_generate(prompt, max_tokens)
            metrics = self.parse_max_output(output)
            
            ttft_ms = metrics.get('ttft_ms', total_time_ms * 0.1)
            tps = metrics.get('tokens_per_second', 0)
            
            # Rough token count if TPS not available
            if tps == 0:
                output_lines = output.split('\n')
                for line in output_lines:
                    if line.strip() and not any(keyword in line.lower() for keyword in ['prompt', 'output', 'time', 'throughput', 'startup']):
                        word_count = len(line.split())
                        total_tokens += word_count
                        break
            else:
                total_tokens += tps * (total_time_ms / 1000)
            
            ttft_times.append(ttft_ms)
            monitor.sample_resources()
        
        end_time = time.perf_counter()
        monitor.stop_monitoring()
        
        total_batch_time_ms = (end_time - start_time) * 1000
        tokens_per_second = total_tokens / (total_batch_time_ms / 1000) if total_batch_time_ms > 0 else 0
        avg_ttft_ms = statistics.mean(ttft_times) if ttft_times else 0
        
        return avg_ttft_ms, tokens_per_second, total_batch_time_ms


class BenchmarkRunner:
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.monitor = SystemMonitor()

    def validate_framework_consistency(self, frameworks: List[Any]) -> bool:
        """Test that both frameworks produce similar output lengths for the same prompt."""
        if len(frameworks) < 2:
            return True
            
        test_prompt = "Explain the concept of machine learning."
        test_max_tokens = 50
        
        print("Validating framework consistency...")
        outputs = {}
        
        for framework in frameworks:
            framework_name = framework.__class__.__name__.replace("Benchmark", "")
            try:
                framework.setup()
                
                if framework_name == "VLLM":
                    sampling_params = SamplingParams(temperature=0.0, max_tokens=test_max_tokens, top_p=1.0)
                    results = framework.llm.generate([test_prompt], sampling_params)
                    output_text = results[0].outputs[0].text if results[0].outputs else ""
                    token_count = len(results[0].outputs[0].token_ids) if results[0].outputs else 0
                elif framework_name == "Mojo":
                    _, output_text = framework.run_max_generate(test_prompt, test_max_tokens)
                    # Parse output to extract generated text (rough approximation)
                    output_lines = output_text.split('\n')
                    for line in output_lines:
                        if line.strip() and not any(keyword in line.lower() for keyword in ['prompt', 'output', 'time', 'throughput', 'startup']):
                            output_text = line.strip()
                            break
                    token_count = len(output_text.split())
                
                outputs[framework_name] = {
                    'text': output_text[:100] + "..." if len(output_text) > 100 else output_text,
                    'token_count': token_count
                }
                print(f"{framework_name} generated {token_count} tokens")
                
            except Exception as e:
                print(f"Warning: Could not validate {framework_name}: {e}")
                return False
            finally:
                if hasattr(framework, 'cleanup'):
                    framework.cleanup()
        
        # Compare outputs
        if len(outputs) == 2:
            frameworks_names = list(outputs.keys())
            count1, count2 = outputs[frameworks_names[0]]['token_count'], outputs[frameworks_names[1]]['token_count']
            ratio = max(count1, count2) / min(count1, count2) if min(count1, count2) > 0 else float('inf')
            
            print(f"Token count ratio: {ratio:.2f}")
            if ratio > 3.0:  # If one framework generates 3x more tokens than the other
                print(f"WARNING: Large difference in output lengths ({count1} vs {count2} tokens)")
                print("This may indicate configuration differences affecting generation behavior.")
                return False
            else:
                print("✓ Frameworks appear to generate similar output lengths")
                
        return True

    def generate_test_prompts(self, length: int, count: int = 1) -> List[str]:
        base_prompts = [
            "Explain the concept of machine learning in simple terms.",
            "Write a short story about a robot discovering emotion.",
            "Describe the benefits of renewable energy sources.",
            "Explain how photosynthesis works in plants.",
            "Write a recipe for making chocolate chip cookies."
        ]
        
        if length == 64:  # Short
            prompts = base_prompts[:count]
        elif length == 256:  # Medium
            prompts = [prompt + " Please provide detailed explanations with examples and elaborate on the key concepts involved." for prompt in base_prompts[:count]]
        else:  # Long (1024)
            long_suffix = " Please provide a comprehensive analysis covering historical context, current applications, future implications, challenges, benefits, technical details, step-by-step processes, real-world examples, and practical recommendations. Include multiple perspectives and discuss both advantages and disadvantages."
            prompts = [prompt + long_suffix for prompt in base_prompts[:count]]
        
        return prompts

    def run_benchmark_scenario(self, framework, prompts: List[str], max_tokens: int, 
                             concurrent_requests: int, batch_size: int) -> BenchmarkResult:
        prompt_length = len(prompts[0].split())
        model_name = framework.model_name
        
        if concurrent_requests == 1:
            # Single request benchmark
            ttft_ms, tokens_per_second, total_time_ms = framework.benchmark_single_request(
                prompts[0], max_tokens, self.monitor
            )
            throughput_req_per_sec = 1000 / total_time_ms if total_time_ms > 0 else 0
            latencies = [total_time_ms]
            
        else:
            # Batch or concurrent requests
            if hasattr(framework, 'benchmark_batch') and len(prompts) > 1:
                ttft_ms, tokens_per_second, total_time_ms = framework.benchmark_batch(
                    prompts[:concurrent_requests], max_tokens, self.monitor
                )
            else:
                # Fallback to sequential requests
                start_time = time.perf_counter()
                ttft_times = []
                tps_values = []
                
                for i in range(concurrent_requests):
                    prompt = prompts[i % len(prompts)]
                    ttft, tps, _ = framework.benchmark_single_request(prompt, max_tokens, self.monitor)
                    ttft_times.append(ttft)
                    tps_values.append(tps)
                
                end_time = time.perf_counter()
                total_time_ms = (end_time - start_time) * 1000
                ttft_ms = statistics.mean(ttft_times)
                tokens_per_second = sum(tps_values)
            
            throughput_req_per_sec = concurrent_requests / (total_time_ms / 1000) if total_time_ms > 0 else 0
            latencies = [total_time_ms / concurrent_requests] * concurrent_requests
        
        # Calculate percentiles
        latency_p50_ms = statistics.median(latencies)
        latency_p95_ms = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0]
        latency_p99_ms = statistics.quantiles(latencies, n=100)[98] if len(latencies) > 1 else latencies[0]
        
        return BenchmarkResult(
            framework=framework.__class__.__name__.replace("Benchmark", ""),
            model=model_name,
            prompt_length=prompt_length,
            output_length=max_tokens,
            concurrent_requests=concurrent_requests,
            batch_size=batch_size,
            ttft_ms=ttft_ms,
            tokens_per_second=tokens_per_second,
            total_time_ms=total_time_ms,
            throughput_req_per_sec=throughput_req_per_sec,
            peak_memory_mb=self.monitor.get_peak_memory(),
            avg_cpu_percent=self.monitor.get_avg_cpu(),
            latency_p50_ms=latency_p50_ms,
            latency_p95_ms=latency_p95_ms,
            latency_p99_ms=latency_p99_ms,
            timestamp=datetime.now().isoformat()
        )

    def run_comprehensive_benchmark(self, frameworks: List[Any], test_scenarios: List[Dict]):
        print("Starting comprehensive benchmark...")
        
        for framework in frameworks:
            framework_name = framework.__class__.__name__.replace("Benchmark", "")
            print(f"\n=== Benchmarking {framework_name} ===")
            
            try:
                framework.setup()
                
                for scenario in test_scenarios:
                    print(f"Running scenario: {scenario}")
                    
                    prompts = self.generate_test_prompts(
                        scenario['prompt_length'], 
                        scenario['concurrent_requests']
                    )
                    
                    result = self.run_benchmark_scenario(
                        framework=framework,
                        prompts=prompts,
                        max_tokens=scenario['max_tokens'],
                        concurrent_requests=scenario['concurrent_requests'],
                        batch_size=scenario.get('batch_size', 1)
                    )
                    
                    self.results.append(result)
                    print(f"TTFT: {result.ttft_ms:.2f}ms, TPS: {result.tokens_per_second:.2f}, "
                          f"Throughput: {result.throughput_req_per_sec:.2f} req/s")
                
            except Exception as e:
                print(f"Error benchmarking {framework_name}: {e}")
                traceback.print_exc()
            finally:
                if hasattr(framework, 'cleanup'):
                    framework.cleanup()

    def save_results(self, filename: str = "benchmark_results.json"):
        results_dict = [asdict(result) for result in self.results]
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"Results saved to {filename}")

    def print_summary(self):
        if not self.results:
            print("No results to display")
            return
        
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        for framework in set(result.framework for result in self.results):
            framework_results = [r for r in self.results if r.framework == framework]
            
            print(f"\n{framework} Results:")
            print("-" * 50)
            
            avg_ttft = statistics.mean([r.ttft_ms for r in framework_results])
            avg_tps = statistics.mean([r.tokens_per_second for r in framework_results])
            avg_throughput = statistics.mean([r.throughput_req_per_sec for r in framework_results])
            avg_memory = statistics.mean([r.peak_memory_mb for r in framework_results])
            
            print(f"Average TTFT: {avg_ttft:.2f} ms")
            print(f"Average TPS: {avg_tps:.2f} tokens/sec")
            print(f"Average Throughput: {avg_throughput:.2f} req/sec")
            print(f"Average Peak Memory: {avg_memory:.2f} MB")


def print_configuration_info():
    """Print information about framework configurations for fair comparison."""
    print("\n" + "="*80)
    print("FRAMEWORK CONFIGURATION COMPARISON")
    print("="*80)
    print("vLLM Configuration:")
    print("  - Model precision: fp16 (vLLM default for optimal performance)")
    print("  - Temperature: 0.0 (greedy decoding)")
    print("  - Top-p: 1.0 (disabled nucleus sampling)")
    print("  - Max model length: 2048")
    print("  - Trust remote code: True")
    print()
    print("MAX Configuration:")
    print("  - Model precision: float32 (MAX default)")
    print("  - Temperature: DEFAULT (not configurable via CLI)")
    print("  - Top-p: DEFAULT (not configurable via CLI)")
    print("  - Max new tokens: Set per scenario")
    print("  - Trust remote code: True")
    print()
    print("NOTE: MAX does not expose temperature/top_p parameters via CLI.")
    print("vLLM has been configured for greedy decoding to maximize determinism.")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM vs Mojo/MAX frameworks")
    parser.add_argument("--model", default="allenai/OLMo-1B-hf", help="Model to benchmark")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file for results")
    parser.add_argument("--frameworks", nargs="+", choices=["vllm", "mojo"], 
                       default=["vllm", "mojo"], help="Frameworks to benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with fewer scenarios")
    
    args = parser.parse_args()
    
    # Print configuration information for transparency
    print_configuration_info()
    
    # Define test scenarios
    if args.quick:
        test_scenarios = [
            {"prompt_length": 64, "max_tokens": 128, "concurrent_requests": 1},
            {"prompt_length": 256, "max_tokens": 256, "concurrent_requests": 1},
            {"prompt_length": 64, "max_tokens": 128, "concurrent_requests": 5},
        ]
    else:
        test_scenarios = [
            # Single request scenarios
            {"prompt_length": 64, "max_tokens": 128, "concurrent_requests": 1},
            {"prompt_length": 256, "max_tokens": 256, "concurrent_requests": 1},
            {"prompt_length": 1024, "max_tokens": 512, "concurrent_requests": 1},
            
            # Concurrent request scenarios
            {"prompt_length": 64, "max_tokens": 128, "concurrent_requests": 5},
            {"prompt_length": 64, "max_tokens": 128, "concurrent_requests": 10},
            {"prompt_length": 256, "max_tokens": 256, "concurrent_requests": 5},
            {"prompt_length": 256, "max_tokens": 256, "concurrent_requests": 10},
        ]
    
    # Initialize frameworks
    frameworks = []
    
    if "vllm" in args.frameworks and VLLM_AVAILABLE:
        frameworks.append(VLLMBenchmark(args.model))
    elif "vllm" in args.frameworks:
        print("Warning: vLLM requested but not available")
    
    if "mojo" in args.frameworks and REQUESTS_AVAILABLE:
        frameworks.append(MojoBenchmark(args.model))
    elif "mojo" in args.frameworks:
        print("Warning: Mojo/MAX requested but requests library not available")
    
    if not frameworks:
        print("No frameworks available for benchmarking")
        sys.exit(1)
    
    # Run benchmarks
    runner = BenchmarkRunner()
    
    # Validate framework consistency if benchmarking multiple frameworks
    if len(frameworks) > 1:
        if not runner.validate_framework_consistency(frameworks):
            print("WARNING: Framework configurations may not be equivalent!")
            print("Results may not be directly comparable. Proceeding anyway...\n")
        else:
            print("✓ Framework validation passed. Configurations appear comparable.\n")
    
    runner.run_comprehensive_benchmark(frameworks, test_scenarios)
    runner.save_results(args.output)
    runner.print_summary()


if __name__ == "__main__":
    main()