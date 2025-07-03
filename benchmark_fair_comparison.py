#!/usr/bin/env python3
"""
Fair vLLM vs MAX Benchmarking using Official MAX benchmark_serving.py
This script automates server management and uses the official benchmarking methodology.
"""

import subprocess
import time
import json
import os
import sys
import argparse
import signal
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedServerManager:
    """Manages vLLM and MAX servers with optimal configurations"""
    
    def __init__(self):
        self.vllm_process = None
        self.max_process = None
        self.vllm_port = 8001
        self.max_port = 8002
        
    def start_vllm_server(self, model: str) -> bool:
        """Start vLLM server with official benchmarking optimizations"""
        logger.info("Starting optimized vLLM server...")
        
        # Kill any existing vLLM processes
        subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"], 
                      capture_output=True)
        time.sleep(2)
        
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--port", str(self.vllm_port),
            "--host", "0.0.0.0",
            "--trust-remote-code",
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", "0.9",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens", "8192",
            "--max-num-seqs", "256",
            "--max-model-len", "8192",
            "--disable-log-stats",
            "--tensor-parallel-size", "1"
        ]
        
        env = os.environ.copy()
        env.update({
            "VLLM_USE_V1": "1",
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",  
            "VLLM_FLASH_ATTN_FORCE_ENABLE": "1",
            "CUDA_VISIBLE_DEVICES": "0"
        })
        
        # Activate vLLM environment
        if os.path.exists("vllm_optimized_env/bin/activate"):
            env["PATH"] = f"{os.getcwd()}/vllm_optimized_env/bin:{env['PATH']}"
        
        self.vllm_process = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        # Wait for server to be ready
        return self._wait_for_server(f"http://localhost:{self.vllm_port}/health", "vLLM", 120)
    
    def start_max_server(self, model: str) -> bool:
        """Start MAX server with optimal configuration"""
        logger.info("Starting optimized MAX server...")
        
        # Kill any existing MAX processes
        subprocess.run(["pkill", "-f", "max serve"], capture_output=True)
        time.sleep(2)
        
        env = os.environ.copy()
        env.update({
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_TOKEN": os.environ.get("HF_TOKEN", "")
        })
        
        if not env["HF_TOKEN"]:
            logger.error("HF_TOKEN not set. MAX server may fail to access gated models.")
            return False
        
        # Check if we have a pixi environment
        max_dir = "max_optimized"
        if os.path.exists(max_dir):
            cmd = ["pixi", "run", "max", "serve", 
                  f"--model-path={model}",
                  f"--port={self.max_port}",
                  "--host=0.0.0.0"]
            cwd = max_dir
        else:
            # Fallback to direct max command
            cmd = ["max", "serve",
                  f"--model-path={model}", 
                  f"--port={self.max_port}",
                  "--host=0.0.0.0"]
            cwd = None
        
        self.max_process = subprocess.Popen(
            cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        # Wait for server to be ready (MAX takes longer to load)
        return self._wait_for_server(f"http://localhost:{self.max_port}/health", "MAX", 180)
    
    def _wait_for_server(self, health_url: str, name: str, timeout: int) -> bool:
        """Wait for server to become healthy"""
        logger.info(f"Waiting for {name} server to be ready...")
        
        for i in range(timeout):
            try:
                response = requests.get(health_url, timeout=1)
                if response.status_code == 200:
                    logger.info(f"✅ {name} server is ready!")
                    return True
            except:
                pass
            
            if i % 10 == 0:
                logger.info(f"Waiting for {name}... ({i}/{timeout}s)")
            time.sleep(1)
        
        logger.error(f"❌ {name} server failed to start after {timeout} seconds")
        return False
    
    def stop_servers(self):
        """Stop both servers"""
        if self.vllm_process:
            logger.info("Stopping vLLM server...")
            self.vllm_process.terminate()
            try:
                self.vllm_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.vllm_process.kill()
        
        if self.max_process:
            logger.info("Stopping MAX server...")
            self.max_process.terminate()
            try:
                self.max_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.max_process.kill()
        
        # Cleanup any remaining processes
        subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"], 
                      capture_output=True)
        subprocess.run(["pkill", "-f", "max serve"], capture_output=True)

class FairBenchmarkRunner:
    """Runs fair comparison benchmarks using official MAX benchmark tool"""
    
    def __init__(self, benchmark_script_path: str = "benchmark_serving_official.py"):
        self.benchmark_script = benchmark_script_path
        self.server_manager = OptimizedServerManager()
        
        if not os.path.exists(self.benchmark_script):
            raise FileNotFoundError(f"Benchmark script not found: {self.benchmark_script}")
    
    def run_benchmark(self, backend: str, model: str, host: str, port: int, 
                     output_file: str, **kwargs) -> Optional[Dict]:
        """Run benchmark using official benchmark script"""
        
        cmd = [
            "python", self.benchmark_script,
            "--backend", backend,
            "--host", host,
            "--port", str(port),
            "--model", model
        ]
        
        # Add optional parameters
        defaults = {
            "num-prompts": kwargs.get("num_prompts", 100),
            "dataset-name": kwargs.get("dataset_name", "sharegpt"),
            "request-rate": kwargs.get("request_rate", 2.0),
            "save-result": kwargs.get("save_result", True),
            "result-filename": output_file
        }
        
        for key, value in defaults.items():
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{key}")
                else:
                    cmd.extend([f"--{key}", str(value)])
        
        logger.info(f"Running benchmark: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info(f"✅ {backend} benchmark completed successfully")
                
                # Try to parse the result
                if os.path.exists(output_file):
                    with open(output_file, 'r') as f:
                        return json.load(f)
                else:
                    # Parse from stdout if file not created
                    try:
                        lines = result.stdout.strip().split('\n')
                        for line in reversed(lines):
                            if line.startswith('{') and line.endswith('}'):
                                return json.loads(line)
                    except:
                        pass
                
                return {"success": True, "output": result.stdout}
            else:
                logger.error(f"❌ {backend} benchmark failed:")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {backend} benchmark timed out")
            return None
        except Exception as e:
            logger.error(f"❌ {backend} benchmark error: {e}")
            return None
    
    def run_fair_comparison(self, model: str, scenarios: List[Dict]) -> Dict:
        """Run complete fair comparison between vLLM and MAX"""
        
        results = {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "vllm_results": [],
            "max_results": [],
            "comparison": {}
        }
        
        try:
            # Test vLLM
            logger.info("="*60)
            logger.info("BENCHMARKING vLLM")
            logger.info("="*60)
            
            if not self.server_manager.start_vllm_server(model):
                raise Exception("Failed to start vLLM server")
            
            for i, scenario in enumerate(scenarios):
                output_file = f"vllm_result_{i}.json"
                result = self.run_benchmark(
                    backend="vllm",
                    model=model,
                    host="localhost", 
                    port=self.server_manager.vllm_port,
                    output_file=output_file,
                    **scenario
                )
                if result:
                    result["scenario"] = scenario
                    results["vllm_results"].append(result)
            
            self.server_manager.stop_servers()
            time.sleep(5)  # Allow cleanup
            
            # Test MAX
            logger.info("="*60)
            logger.info("BENCHMARKING MAX")
            logger.info("="*60)
            
            if not self.server_manager.start_max_server(model):
                raise Exception("Failed to start MAX server")
            
            for i, scenario in enumerate(scenarios):
                output_file = f"max_result_{i}.json"
                result = self.run_benchmark(
                    backend="modular",
                    model=model,
                    host="localhost",
                    port=self.server_manager.max_port, 
                    output_file=output_file,
                    **scenario
                )
                if result:
                    result["scenario"] = scenario
                    results["max_results"].append(result)
            
            # Generate comparison
            results["comparison"] = self._compare_results(
                results["vllm_results"], results["max_results"]
            )
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            results["error"] = str(e)
        finally:
            self.server_manager.stop_servers()
        
        return results
    
    def _compare_results(self, vllm_results: List[Dict], max_results: List[Dict]) -> Dict:
        """Compare vLLM and MAX results"""
        if not vllm_results or not max_results:
            return {"error": "Missing results for comparison"}
        
        comparison = {
            "scenarios_compared": min(len(vllm_results), len(max_results)),
            "metrics": {}
        }
        
        # Extract key metrics for comparison
        for i in range(comparison["scenarios_compared"]):
            vllm = vllm_results[i]
            max_res = max_results[i]
            
            scenario_name = f"scenario_{i}"
            comparison["metrics"][scenario_name] = {
                "vllm": self._extract_metrics(vllm),
                "max": self._extract_metrics(max_res),
            }
        
        return comparison
    
    def _extract_metrics(self, result: Dict) -> Dict:
        """Extract key performance metrics from benchmark result"""
        # The official benchmark script returns these metrics
        metrics = {}
        
        for key in ["request_throughput", "input_token_throughput", "output_token_throughput",
                   "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
                   "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms"]:
            metrics[key] = result.get(key, 0.0)
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description="Fair vLLM vs MAX Benchmark")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct", 
                       help="Model to benchmark")
    parser.add_argument("--output", default="fair_comparison_results.json",
                       help="Output file for results")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmark with fewer scenarios")
    parser.add_argument("--num-prompts", type=int, default=100,
                       help="Number of prompts per scenario")
    
    args = parser.parse_args()
    
    # Define benchmark scenarios
    if args.quick:
        scenarios = [
            {"dataset_name": "sharegpt", "num_prompts": 50, "request_rate": 1.0},
            {"dataset_name": "sharegpt", "num_prompts": 50, "request_rate": 2.0},
        ]
    else:
        scenarios = [
            {"dataset_name": "sharegpt", "num_prompts": args.num_prompts, "request_rate": 0.5},
            {"dataset_name": "sharegpt", "num_prompts": args.num_prompts, "request_rate": 1.0},
            {"dataset_name": "sharegpt", "num_prompts": args.num_prompts, "request_rate": 2.0},
            {"dataset_name": "sharegpt", "num_prompts": args.num_prompts, "request_rate": 4.0},
            {"dataset_name": "sonnet", "num_prompts": args.num_prompts, "request_rate": 2.0},
        ]
    
    logger.info("🚀 Starting Fair vLLM vs MAX Benchmark")
    logger.info(f"Model: {args.model}")
    logger.info(f"Scenarios: {len(scenarios)}")
    logger.info(f"Using official MAX benchmark methodology")
    
    runner = FairBenchmarkRunner()
    
    # Setup signal handler for cleanup
    def signal_handler(sig, frame):
        logger.info("Interrupt received, cleaning up...")
        runner.server_manager.stop_servers()
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        results = runner.run_fair_comparison(args.model, scenarios)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Benchmark complete! Results saved to {args.output}")
        
        # Print summary
        if "comparison" in results and "metrics" in results["comparison"]:
            logger.info("\n📊 BENCHMARK SUMMARY")
            logger.info("="*40)
            
            for scenario, metrics in results["comparison"]["metrics"].items():
                vllm_throughput = metrics["vllm"].get("request_throughput", 0)
                max_throughput = metrics["max"].get("request_throughput", 0)
                vllm_ttft = metrics["vllm"].get("mean_ttft_ms", 0)
                max_ttft = metrics["max"].get("mean_ttft_ms", 0)
                
                logger.info(f"\n{scenario}:")
                logger.info(f"  Request Throughput - vLLM: {vllm_throughput:.2f}, MAX: {max_throughput:.2f}")
                logger.info(f"  Mean TTFT - vLLM: {vllm_ttft:.2f}ms, MAX: {max_ttft:.2f}ms")
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()