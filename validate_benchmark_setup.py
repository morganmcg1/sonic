#!/usr/bin/env python3
"""
Validation Suite for Fair vLLM vs MAX Benchmarking
Ensures both frameworks are configured optimally and consistently.
"""

import os
import sys
import json
import time
import requests
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    component: str
    test_name: str
    passed: bool
    message: str
    details: Optional[Dict] = None

class BenchmarkValidator:
    """Comprehensive validation for fair benchmarking setup"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.vllm_port = 8001
        self.max_port = 8002
    
    def run_all_validations(self) -> bool:
        """Run complete validation suite"""
        logger.info("🔍 Starting Benchmark Setup Validation")
        logger.info("=" * 60)
        
        self.validate_environment()
        self.validate_dependencies()
        self.validate_model_access()
        self.validate_benchmark_tools()
        self.validate_server_configs()
        
        # Print summary
        self.print_validation_summary()
        
        # Return overall success
        return all(result.passed for result in self.results)
    
    def validate_environment(self):
        """Validate environment variables and system setup"""
        logger.info("📋 Validating Environment Setup")
        
        # Check HF_TOKEN
        hf_token = os.environ.get("HF_TOKEN")
        self.results.append(ValidationResult(
            component="Environment",
            test_name="HF_TOKEN Set",
            passed=bool(hf_token),
            message="HF_TOKEN is required for accessing gated models" if not hf_token else "HF_TOKEN configured",
            details={"token_length": len(hf_token) if hf_token else 0}
        ))
        
        # Check CUDA availability
        cuda_available = self._check_cuda()
        self.results.append(ValidationResult(
            component="Environment", 
            test_name="CUDA Available",
            passed=cuda_available,
            message="CUDA/GPU not detected - performance will be limited" if not cuda_available else "CUDA detected",
            details={"cuda_available": cuda_available}
        ))
        
        # Check vLLM optimization environment variables
        vllm_env_vars = {
            "VLLM_USE_V1": "1",
            "VLLM_ATTENTION_BACKEND": ["FLASHINFER", "FLASH_ATTN"],
            "VLLM_FLASH_ATTN_FORCE_ENABLE": "1"
        }
        
        for var, expected in vllm_env_vars.items():
            current = os.environ.get(var)
            if isinstance(expected, list):
                passed = current in expected
                expected_str = " or ".join(f"'{v}'" for v in expected)
                message = f"Correctly set to '{current}'" if passed else f"Expected {expected_str}, got '{current}'"
            else:
                passed = current == expected
                message = f"Correctly set to '{current}'" if passed else f"Expected '{expected}', got '{current}'"
            
            self.results.append(ValidationResult(
                component="Environment",
                test_name=f"vLLM {var}",
                passed=passed,
                message=message,
                details={"expected": expected, "actual": current}
            ))
        
        # Check MAX optimization environment variables
        max_env_vars = {
            "HF_HUB_ENABLE_HF_TRANSFER": "1"
        }
        
        for var, expected in max_env_vars.items():
            current = os.environ.get(var)
            passed = current == expected
            self.results.append(ValidationResult(
                component="Environment",
                test_name=f"MAX {var}",
                passed=passed,
                message=f"Expected '{expected}', got '{current}'" if not passed else f"Correctly set to '{expected}'",
                details={"expected": expected, "actual": current}
            ))
    
    def validate_dependencies(self):
        """Validate required dependencies and installations"""
        logger.info("📦 Validating Dependencies")
        
        # Check Python packages
        required_packages = [
            "vllm", "transformers", "torch", "aiohttp", "requests", 
            "numpy", "tqdm", "huggingface_hub"
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                self.results.append(ValidationResult(
                    component="Dependencies",
                    test_name=f"Python package: {package}",
                    passed=True,
                    message=f"{package} installed"
                ))
            except ImportError:
                self.results.append(ValidationResult(
                    component="Dependencies", 
                    test_name=f"Python package: {package}",
                    passed=False,
                    message=f"{package} not installed"
                ))
        
        # Check Flash Attention
        try:
            import flash_attn
            self.results.append(ValidationResult(
                component="Dependencies",
                test_name="Flash Attention",
                passed=True,
                message="Flash Attention available for optimal performance"
            ))
        except ImportError:
            self.results.append(ValidationResult(
                component="Dependencies",
                test_name="Flash Attention", 
                passed=False,
                message="Flash Attention not available - will use fallback"
            ))
        
        # Check MAX installation
        max_available = self._check_max_installation()
        self.results.append(ValidationResult(
            component="Dependencies",
            test_name="MAX Installation",
            passed=max_available,
            message="MAX framework installed and accessible" if max_available else "MAX not accessible"
        ))
        
        # Check pixi availability
        pixi_available = self._check_command("pixi")
        self.results.append(ValidationResult(
            component="Dependencies",
            test_name="Pixi Package Manager",
            passed=pixi_available,
            message="Pixi available for MAX environment management" if pixi_available else "Pixi not found"
        ))
    
    def validate_model_access(self):
        """Validate access to the benchmark model"""
        logger.info("🤖 Validating Model Access")
        
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        
        # Check model access via Hugging Face
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Test tokenization
            test_text = "Hello, world!"
            tokens = tokenizer(test_text)
            
            self.results.append(ValidationResult(
                component="Model Access",
                test_name="Model Tokenizer",
                passed=True,
                message=f"Successfully loaded {model_name} tokenizer",
                details={"test_tokens": len(tokens["input_ids"])}
            ))
        except Exception as e:
            self.results.append(ValidationResult(
                component="Model Access",
                test_name="Model Tokenizer",
                passed=False,
                message=f"Failed to load {model_name}: {str(e)}"
            ))
    
    def validate_benchmark_tools(self):
        """Validate benchmark tools and scripts"""
        logger.info("🛠️ Validating Benchmark Tools")
        
        # Check official benchmark script
        benchmark_script = "benchmark_serving_official.py"
        script_exists = os.path.exists(benchmark_script)
        self.results.append(ValidationResult(
            component="Benchmark Tools",
            test_name="Official Benchmark Script",
            passed=script_exists,
            message=f"{benchmark_script} found" if script_exists else f"{benchmark_script} missing"
        ))
        
        if script_exists:
            # Validate script can be imported/executed
            try:
                result = subprocess.run([sys.executable, benchmark_script, "--help"], 
                                      capture_output=True, text=True, timeout=10)
                help_works = result.returncode == 0
                self.results.append(ValidationResult(
                    component="Benchmark Tools",
                    test_name="Benchmark Script Execution", 
                    passed=help_works,
                    message="Benchmark script can be executed" if help_works else "Benchmark script has issues"
                ))
            except Exception as e:
                self.results.append(ValidationResult(
                    component="Benchmark Tools",
                    test_name="Benchmark Script Execution",
                    passed=False,
                    message=f"Error testing benchmark script: {str(e)}"
                ))
        
        # Check improved benchmark runner
        fair_script = "benchmark_fair_comparison.py"
        fair_exists = os.path.exists(fair_script)
        self.results.append(ValidationResult(
            component="Benchmark Tools",
            test_name="Fair Comparison Script",
            passed=fair_exists,
            message=f"{fair_script} found" if fair_exists else f"{fair_script} missing"
        ))
    
    def validate_server_configs(self):
        """Validate server startup configurations"""
        logger.info("⚙️ Validating Server Configurations")
        
        # Check vLLM startup script
        vllm_script = "start_vllm_optimized.sh"
        vllm_exists = os.path.exists(vllm_script)
        self.results.append(ValidationResult(
            component="Server Config",
            test_name="vLLM Startup Script",
            passed=vllm_exists,
            message=f"{vllm_script} found" if vllm_exists else f"{vllm_script} missing"
        ))
        
        # Check MAX startup script
        max_script = "start_max_optimized.sh"
        max_exists = os.path.exists(max_script)
        self.results.append(ValidationResult(
            component="Server Config",
            test_name="MAX Startup Script", 
            passed=max_exists,
            message=f"{max_script} found" if max_exists else f"{max_script} missing"
        ))
        
        # Check virtual environments
        vllm_env = "vllm_optimized_env"
        vllm_env_exists = os.path.exists(vllm_env)
        self.results.append(ValidationResult(
            component="Server Config",
            test_name="vLLM Environment",
            passed=vllm_env_exists,
            message=f"{vllm_env} environment found" if vllm_env_exists else f"{vllm_env} missing"
        ))
        
        max_env = "max_optimized"
        max_env_exists = os.path.exists(max_env)
        self.results.append(ValidationResult(
            component="Server Config", 
            test_name="MAX Environment",
            passed=max_env_exists,
            message=f"{max_env} environment found" if max_env_exists else f"{max_env} missing"
        ))
    
    def validate_server_responses(self, test_servers: bool = False):
        """Optionally validate server responses if servers are running"""
        if not test_servers:
            return
            
        logger.info("🔗 Testing Server Responses")
        
        # Test vLLM server
        vllm_healthy = self._test_server_health(f"http://localhost:{self.vllm_port}/health", "vLLM")
        if vllm_healthy:
            vllm_inference = self._test_server_inference(
                f"http://localhost:{self.vllm_port}/v1/completions", "vLLM"
            )
        else:
            vllm_inference = False
            
        # Test MAX server  
        max_healthy = self._test_server_health(f"http://localhost:{self.max_port}/health", "MAX")
        if max_healthy:
            max_inference = self._test_server_inference(
                f"http://localhost:{self.max_port}/v1/completions", "MAX"
            )
        else:
            max_inference = False
        
        self.results.extend([
            ValidationResult("Server Health", "vLLM Health Check", vllm_healthy, 
                           "vLLM server responding" if vllm_healthy else "vLLM server not accessible"),
            ValidationResult("Server Health", "vLLM Inference", vllm_inference,
                           "vLLM inference working" if vllm_inference else "vLLM inference failed"),
            ValidationResult("Server Health", "MAX Health Check", max_healthy,
                           "MAX server responding" if max_healthy else "MAX server not accessible"), 
            ValidationResult("Server Health", "MAX Inference", max_inference,
                           "MAX inference working" if max_inference else "MAX inference failed")
        ])
    
    def print_validation_summary(self):
        """Print detailed validation summary"""
        logger.info("\n📊 VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        # Group results by component
        components = {}
        for result in self.results:
            if result.component not in components:
                components[result.component] = []
            components[result.component].append(result)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        
        for component, tests in components.items():
            logger.info(f"\n{component}:")
            for test in tests:
                status = "✅" if test.passed else "❌"
                logger.info(f"  {status} {test.test_name}: {test.message}")
        
        logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All validations passed! Ready for fair benchmarking.")
        else:
            logger.info("⚠️  Some validations failed. Address issues before benchmarking.")
            
        return passed_tests == total_tests
    
    def get_setup_recommendations(self) -> List[str]:
        """Get recommendations for fixing validation issues"""
        recommendations = []
        
        for result in self.results:
            if not result.passed:
                if "HF_TOKEN" in result.test_name:
                    recommendations.append("Set HF_TOKEN: export HF_TOKEN='your_token_here'")
                elif "VLLM_" in result.test_name:
                    recommendations.append(f"Set {result.test_name.split()[-1]}: export {result.test_name.split()[-1]}={result.details.get('expected', '1')}")
                elif "MAX" in result.test_name and "HF_HUB" in result.test_name:
                    recommendations.append("Set HF_HUB_ENABLE_HF_TRANSFER: export HF_HUB_ENABLE_HF_TRANSFER=1")
                elif "Flash Attention" in result.test_name:
                    recommendations.append("Install Flash Attention: pip install flash-attn --no-build-isolation")
                elif "package" in result.test_name:
                    package = result.test_name.split(": ")[-1]
                    recommendations.append(f"Install missing package: pip install {package}")
                elif "Script" in result.test_name:
                    recommendations.append("Run setup script: ./setup_improved_benchmark.sh")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _check_max_installation(self) -> bool:
        """Check if MAX is properly installed"""
        # Try pixi run max first
        try:
            if os.path.exists("max_optimized"):
                result = subprocess.run(["pixi", "run", "max", "--version"], 
                                      cwd="max_optimized", capture_output=True, timeout=10)
                if result.returncode == 0:
                    return True
        except:
            pass
        
        # Try direct max command
        try:
            result = subprocess.run(["max", "--version"], capture_output=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    def _check_command(self, command: str) -> bool:
        """Check if a command is available"""
        try:
            result = subprocess.run([command, "--version"], capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def _test_server_health(self, health_url: str, name: str) -> bool:
        """Test if server health endpoint responds"""
        try:
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _test_server_inference(self, api_url: str, name: str) -> bool:
        """Test server inference capability"""
        try:
            payload = {
                "model": "meta-llama/Meta-Llama-3-8B-Instruct",
                "prompt": "What is 2+2?",
                "max_tokens": 5,
                "temperature": 0.0
            }
            
            response = requests.post(api_url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return "choices" in result and len(result["choices"]) > 0
            return False
        except:
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate benchmark setup")
    parser.add_argument("--test-servers", action="store_true",
                       help="Test server responses (requires servers to be running)")
    parser.add_argument("--output", help="Save validation results to JSON file")
    
    args = parser.parse_args()
    
    validator = BenchmarkValidator()
    
    # Run core validations
    all_passed = validator.run_all_validations()
    
    # Optionally test servers
    if args.test_servers:
        validator.validate_server_responses(test_servers=True)
        all_passed = validator.print_validation_summary()
    
    # Save results if requested
    if args.output:
        results_data = {
            "timestamp": time.time(),
            "all_passed": all_passed,
            "results": [
                {
                    "component": r.component,
                    "test_name": r.test_name, 
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details
                } for r in validator.results
            ],
            "recommendations": validator.get_setup_recommendations()
        }
        
        with open(args.output, 'w') as f:
            json.dump(results_data, f, indent=2)
        logger.info(f"Validation results saved to {args.output}")
    
    # Print recommendations if there are issues
    if not all_passed:
        recommendations = validator.get_setup_recommendations()
        if recommendations:
            logger.info("\n🔧 SETUP RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"{i}. {rec}")
    
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()