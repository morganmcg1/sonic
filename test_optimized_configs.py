#!/usr/bin/env python3
"""
Test script to validate optimized vLLM and MAX configurations before full benchmarking.
"""

import requests
import time
import subprocess
import os
import sys
import json

def test_server_health(base_url, server_name, timeout=10):
    """Test if server is healthy and responding"""
    try:
        response = requests.get(f"{base_url}/health", timeout=timeout)
        if response.status_code == 200:
            print(f"✅ {server_name} server is healthy")
            return True
        else:
            print(f"❌ {server_name} server health check failed: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ {server_name} server health check failed: {e}")
        return False

def test_inference_request(base_url, server_name):
    """Test a simple inference request"""
    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Answer briefly."}
        ],
        "max_tokens": 10,
        "temperature": 0.0,
        "stream": False,
        "stop": ["<|end_of_text|>", "<|eot_id|>"]
    }
    
    try:
        start_time = time.perf_counter()
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=30
        )
        end_time = time.perf_counter()
        
        if response.status_code == 200:
            result = response.json()
            latency_ms = (end_time - start_time) * 1000
            
            print(f"✅ {server_name} inference successful:")
            print(f"   - Latency: {latency_ms:.2f}ms")
            
            if 'usage' in result:
                usage = result['usage']
                print(f"   - Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
                print(f"   - Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            
            if 'choices' in result and result['choices']:
                content = result['choices'][0]['message']['content']
                print(f"   - Response: {content[:50]}...")
            
            return True
        else:
            print(f"❌ {server_name} inference failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except requests.RequestException as e:
        print(f"❌ {server_name} inference failed: {e}")
        return False

def check_vllm_optimizations():
    """Check if vLLM optimizations are properly configured"""
    print("\n🔍 Checking vLLM optimization environment variables:")
    
    optimizations = {
        "VLLM_USE_V1": "v1 engine enabled",
        "VLLM_ATTENTION_BACKEND": "Flash Attention backend",
        "VLLM_FLASH_ATTN_FORCE_ENABLE": "Flash Attention forced"
    }
    
    all_set = True
    for env_var, description in optimizations.items():
        value = os.environ.get(env_var)
        if value:
            print(f"   ✅ {env_var}={value} ({description})")
        else:
            print(f"   ❌ {env_var} not set ({description})")
            all_set = False
    
    return all_set

def check_max_optimizations():
    """Check if MAX optimizations are properly configured"""
    print("\n🔍 Checking MAX optimization environment variables:")
    
    optimizations = {
        "HF_HUB_ENABLE_HF_TRANSFER": "HF transfer acceleration",
        "HF_TOKEN": "Hugging Face authentication"
    }
    
    all_set = True
    for env_var, description in optimizations.items():
        value = os.environ.get(env_var)
        if value:
            if env_var == "HF_TOKEN":
                print(f"   ✅ {env_var}=***masked*** ({description})")
            else:
                print(f"   ✅ {env_var}={value} ({description})")
        else:
            print(f"   ❌ {env_var} not set ({description})")
            all_set = False
    
    return all_set

def main():
    print("🚀 Testing Optimized vLLM and MAX Configurations")
    print("=" * 60)
    
    # Check environment configurations
    vllm_env_ok = check_vllm_optimizations()
    max_env_ok = check_max_optimizations()
    
    if not vllm_env_ok:
        print("\n⚠️  vLLM environment variables not optimally configured")
        print("Run these commands before starting vLLM server:")
        print("export VLLM_USE_V1=1")
        print("export VLLM_ATTENTION_BACKEND=FLASHINFER")
        print("export VLLM_FLASH_ATTN_FORCE_ENABLE=1")
    
    if not max_env_ok:
        print("\n⚠️  MAX environment variables not optimally configured")
        print("Run these commands before starting MAX server:")
        print("export HF_HUB_ENABLE_HF_TRANSFER=1")
        print("export HF_TOKEN=your_token_here")
    
    # Test server connectivity
    print("\n🔗 Testing Server Connectivity")
    print("-" * 30)
    
    vllm_healthy = test_server_health("http://localhost:8001", "vLLM")
    max_healthy = test_server_health("http://localhost:8002", "MAX")
    
    if not vllm_healthy:
        print("\n❌ vLLM server not accessible. Start it with:")
        print("python benchmark_servers.py --start-servers --frameworks vllm")
        
    if not max_healthy:
        print("\n❌ MAX server not accessible. Start it with:")
        print("python benchmark_servers.py --start-servers --frameworks max")
    
    if not (vllm_healthy and max_healthy):
        print("\n⚠️  Cannot proceed with inference tests - servers not ready")
        return False
    
    # Test inference requests
    print("\n🧠 Testing Inference Requests")
    print("-" * 30)
    
    vllm_inference_ok = test_inference_request("http://localhost:8001", "vLLM")
    max_inference_ok = test_inference_request("http://localhost:8002", "MAX")
    
    # Summary
    print("\n📊 Configuration Test Summary")
    print("=" * 40)
    
    print(f"vLLM Environment: {'✅ Optimal' if vllm_env_ok else '❌ Needs Setup'}")
    print(f"vLLM Health: {'✅ Healthy' if vllm_healthy else '❌ Not Running'}")
    print(f"vLLM Inference: {'✅ Working' if vllm_inference_ok else '❌ Failed'}")
    
    print(f"MAX Environment: {'✅ Optimal' if max_env_ok else '❌ Needs Setup'}")
    print(f"MAX Health: {'✅ Healthy' if max_healthy else '❌ Not Running'}")
    print(f"MAX Inference: {'✅ Working' if max_inference_ok else '❌ Failed'}")
    
    all_good = all([vllm_env_ok, vllm_healthy, vllm_inference_ok, 
                   max_env_ok, max_healthy, max_inference_ok])
    
    if all_good:
        print("\n🎉 All optimizations configured correctly! Ready for benchmarking.")
        print("Run: python benchmark_servers.py --extended")
    else:
        print("\n⚠️  Some optimizations need attention before benchmarking.")
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)