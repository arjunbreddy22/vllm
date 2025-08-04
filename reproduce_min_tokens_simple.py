#!/usr/bin/env python3
"""
Simple reproduction script for min_tokens bug analysis.

🚨 CRITICAL: This bug is V1-SPECIFIC! We need to test V1 engine, not V0.

This script focuses on testing the SamplingParams logic and reproducing
the issue described in the bug report without requiring full vLLM LLM infrastructure.
"""

import sys
import os
# Since we're now inside the vllm directory, add the parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

# 🚨 ENABLE V1 ENGINE
os.environ['VLLM_USE_V1'] = '1'
print("🚨 V1 MODE ENABLED: VLLM_USE_V1=1")

def test_sampling_params_creation():
    """Test that SamplingParams can be created with the bug scenario parameters"""
    print("="*60)
    print("TESTING SAMPLING PARAMS CREATION")
    print("="*60)
    
    try:
        from vllm.sampling_params import SamplingParams
        
        # Exact configuration from bug report
        sampling_params = SamplingParams(
            n=1, 
            presence_penalty=0.0, 
            frequency_penalty=0.0, 
            repetition_penalty=1.0, 
            temperature=0.0, 
            top_p=1.0, 
            top_k=0, 
            min_p=0.0, 
            seed=None, 
            stop=['\n'],  # Stop on newline
            stop_token_ids=[], 
            bad_words=[], 
            include_stop_str_in_output=True, 
            ignore_eos=False, 
            max_tokens=200, 
            min_tokens=200,  # Should generate at least 200 tokens
            logprobs=None, 
            prompt_logprobs=None, 
            skip_special_tokens=True, 
            spaces_between_special_tokens=True, 
            truncate_prompt_tokens=None, 
            guided_decoding=None, 
            extra_args=None
        )
        
        print("✅ SamplingParams created successfully!")
        print(f"   min_tokens: {sampling_params.min_tokens}")
        print(f"   max_tokens: {sampling_params.max_tokens}")
        print(f"   stop: {sampling_params.stop}")
        print(f"   include_stop_str_in_output: {sampling_params.include_stop_str_in_output}")
        print(f"   output_text_buffer_length: {sampling_params.output_text_buffer_length}")
        
        # Test edge cases
        print("\n" + "="*40)
        print("TESTING EDGE CASES")
        print("="*40)
        
        test_cases = [
            {
                "name": "min_tokens=10, max_tokens=50, stop=['\n']",
                "min_tokens": 10,
                "max_tokens": 50,
                "stop": ['\n']
            },
            {
                "name": "min_tokens=5, max_tokens=20, stop=['.']", 
                "min_tokens": 5,
                "max_tokens": 20,
                "stop": ['.']
            },
            {
                "name": "min_tokens=0, max_tokens=10, stop=['\n']",
                "min_tokens": 0,
                "max_tokens": 10,
                "stop": ['\n']
            },
            {
                "name": "min_tokens=20, max_tokens=20, stop=None (exact equality)",
                "min_tokens": 20,
                "max_tokens": 20,
                "stop": None
            }
        ]
        
        for case in test_cases:
            try:
                params = SamplingParams(
                    temperature=0.0,
                    min_tokens=case['min_tokens'],
                    max_tokens=case['max_tokens'],
                    stop=case['stop'],
                    include_stop_str_in_output=True
                )
                print(f"✅ {case['name']} - OK")
                print(f"   buffer_length: {params.output_text_buffer_length}")
            except Exception as e:
                print(f"❌ {case['name']} - ERROR: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create SamplingParams: {e}")
        import traceback
        traceback.print_exc()
        return False

def collect_environment_info():
    """Collect basic environment information"""
    print("="*60)
    print("ENVIRONMENT INFORMATION")
    print("="*60)
    
    try:
        from vllm.collect_env import get_pretty_env_info
        env_info = get_pretty_env_info()
        print(env_info)
    except Exception as e:
        print(f"Could not get vLLM environment info: {e}")
        # Fallback to basic info
        import platform
        try:
            import torch
            print(f"Python version: {platform.python_version()}")
            print(f"Platform: {platform.platform()}")
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA devices: {torch.cuda.device_count()}")
        except Exception as torch_error:
            print(f"PyTorch info unavailable: {torch_error}")
            print(f"Python version: {platform.python_version()}")
            print(f"Platform: {platform.platform()}")
    
    print("="*60)
    print()

def analyze_min_tokens_logic():
    """Analyze the min_tokens implementation in the codebase"""
    print("="*60)
    print("ANALYZING MIN_TOKENS LOGIC")
    print("="*60)
    
    try:
        # Let's examine where min_tokens logic is implemented
        from vllm.sampling_params import SamplingParams
        
        # Check if there are any references to min_tokens in key components
        print("SamplingParams min_tokens field details:")
        
        # Create instance to see defaults
        default_params = SamplingParams()
        print(f"Default min_tokens: {default_params.min_tokens}")
        print(f"Default max_tokens: {default_params.max_tokens}")
        print(f"Default stop: {default_params.stop}")
        
        # Look for min_tokens validation logic
        bug_params = SamplingParams(min_tokens=200, max_tokens=200, stop=['\n'])
        print(f"\nBug scenario params:")
        print(f"min_tokens: {bug_params.min_tokens}")
        print(f"max_tokens: {bug_params.max_tokens}")
        print(f"stop: {bug_params.stop}")
        
        # Check the validation method
        print(f"\nValidation passed: SamplingParams accepts min_tokens=max_tokens")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing min_tokens logic: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("MIN_TOKENS BUG REPRODUCTION SCRIPT")
    print("="*60)
    print("This script analyzes the min_tokens bug without requiring")
    print("full LLM model loading (which needs CUDA compilation).")
    print()
    
    # Step 1: Collect environment info
    collect_environment_info()
    
    # Step 2: Test SamplingParams creation
    if not test_sampling_params_creation():
        print("❌ Cannot proceed - SamplingParams creation failed")
        return
    
    # Step 3: Analyze min_tokens logic
    analyze_min_tokens_logic()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("✅ SamplingParams accepts the bug scenario configuration")
    print("✅ min_tokens=200, max_tokens=200, stop=['\n'] is valid")
    print("⚠️  Need to test actual generation to see the bug in action")
    print("⚠️  The bug likely occurs during token generation, not parameter validation")
    print()
    print("NEXT STEPS:")
    print("1. Look at the generation logic to see where min_tokens is checked")
    print("2. Find where stop sequences are processed") 
    print("3. Identify the order of operations that causes early stopping")
    print("="*60)

if __name__ == "__main__":
    main()