#!/usr/bin/env python3
"""
Reproduction script for min_tokens bug with stop sequences.

Bug: When both min_tokens and stop are specified, vLLM stops generation 
as soon as a stop sequence is encountered, even if min_tokens threshold 
has not been reached.

Expected: The model should continue generating until BOTH conditions are met:
1. At least min_tokens tokens have been generated
2. A stop sequence is encountered
"""

import sys
import os
# Since we're now inside the vllm directory, add the parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from vllm import LLM, SamplingParams

def collect_environment_info():
    """Collect environment information for debugging"""
    print("="*60)
    print("ENVIRONMENT INFORMATION")
    print("="*60)
    
    try:
        from vllm.collect_env import get_pretty_env_info
        env_info = get_pretty_env_info()
        print(env_info)
    except Exception as e:
        print(f"Failed to collect environment info: {e}")
        # Fallback to basic info
        import platform
        import torch
        print(f"Python version: {platform.python_version()}")
        print(f"Platform: {platform.platform()}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
    
    print("="*60)
    print()

def reproduce_bug():
    """Reproduce the exact scenario from the bug report"""
    print("REPRODUCING MIN_TOKENS BUG")
    print("="*60)
    
    # Exact configuration from bug report
    prompts = [
        "test/",
    ]
    
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
    
    print(f"Prompt: {prompts[0]!r}")
    print(f"Sampling params:")
    print(f"  min_tokens: {sampling_params.min_tokens}")
    print(f"  max_tokens: {sampling_params.max_tokens}")
    print(f"  stop: {sampling_params.stop}")
    print(f"  include_stop_str_in_output: {sampling_params.include_stop_str_in_output}")
    print()
    
    try:
        # Try to use the same model from bug report
        model_name = "ibm-granite/granite-3.3-8b-instruct"
        print(f"Loading model: {model_name}")
        
        llm = LLM(model=model_name)
        
        print("Generating outputs...")
        outputs = llm.generate(prompts, sampling_params)
        
        print("\nRESULTS:")
        print("="*40)
        
        for output in outputs:
            print(f"Prompt: {output.prompt!r}")
            for completion in output.outputs:
                print(f"Generated text: {completion.text!r}")
                print(f"Generated token count: {len(completion.token_ids)}")
                print(f"Stop reason: {completion.stop_reason}")
                print(f"Token IDs: {completion.token_ids}")
                
        # Analyze the bug
        print("\nBUG ANALYSIS:")
        print("="*40)
        for output in outputs:
            for completion in output.outputs:
                token_count = len(completion.token_ids)
                expected_min = sampling_params.min_tokens
                
                print(f"Expected minimum tokens: {expected_min}")
                print(f"Actual tokens generated: {token_count}")
                print(f"Stop reason: {completion.stop_reason}")
                
                if completion.stop_reason == "stop" and token_count < expected_min:
                    print("🚨 BUG CONFIRMED: Stopped early due to stop sequence, ignoring min_tokens!")
                    print(f"   Should have generated at least {expected_min} tokens")
                    print(f"   But only generated {token_count} tokens")
                elif completion.stop_reason == "stop" and token_count >= expected_min:
                    print("✅ WORKING CORRECTLY: Generated enough tokens before stopping")
                elif completion.stop_reason == "length":
                    print("⚠️  STOPPED DUE TO MAX_TOKENS: May mask the min_tokens issue")
                else:
                    print(f"? UNEXPECTED STOP REASON: {completion.stop_reason}")
                    
    except Exception as e:
        print(f"Error during reproduction: {e}")
        import traceback
        traceback.print_exc()
        
        # Try with a smaller/local model if available
        print("\nTrying with smaller model...")
        try:
            # Fallback to a smaller model that might be available
            fallback_models = [
                "microsoft/DialoGPT-medium",
                "gpt2", 
                "facebook/opt-125m"
            ]
            
            for model_name in fallback_models:
                try:
                    print(f"Trying model: {model_name}")
                    llm = LLM(model=model_name)
                    outputs = llm.generate(prompts, sampling_params)
                    
                    print(f"\nFALLBACK RESULTS ({model_name}):")
                    print("="*50)
                    for output in outputs:
                        for completion in output.outputs:
                            print(f"Generated text: {completion.text!r}")
                            print(f"Generated token count: {len(completion.token_ids)}")
                            print(f"Stop reason: {completion.stop_reason}")
                    break
                except Exception as fallback_error:
                    print(f"Fallback model {model_name} failed: {fallback_error}")
                    continue
            else:
                print("All fallback models failed")
                
        except Exception as fallback_error:
            print(f"Fallback attempt failed: {fallback_error}")

def test_additional_scenarios():
    """Test additional scenarios to understand the bug better"""
    print("\n" + "="*60)
    print("ADDITIONAL TEST SCENARIOS")
    print("="*60)
    
    test_cases = [
        {
            "name": "Case 1: min_tokens=10, max_tokens=50, stop=['\n']",
            "prompt": "Write a short story about a cat",
            "min_tokens": 10,
            "max_tokens": 50,
            "stop": ['\n']
        },
        {
            "name": "Case 2: min_tokens=5, max_tokens=20, stop=['.']", 
            "prompt": "List numbers: 1, 2, 3",
            "min_tokens": 5,
            "max_tokens": 20,
            "stop": ['.']
        },
        {
            "name": "Case 3: min_tokens=0, max_tokens=10, stop=['\n'] (baseline)",
            "prompt": "test/",
            "min_tokens": 0,
            "max_tokens": 10,
            "stop": ['\n']
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n{case['name']}")
        print("-" * len(case['name']))
        
        sampling_params = SamplingParams(
            temperature=0.0,
            min_tokens=case['min_tokens'],
            max_tokens=case['max_tokens'],
            stop=case['stop'],
            include_stop_str_in_output=True
        )
        
        print(f"Prompt: {case['prompt']!r}")
        print(f"min_tokens={case['min_tokens']}, max_tokens={case['max_tokens']}, stop={case['stop']}")
        
        try:
            # Use a simple model for testing
            llm = LLM(model="gpt2")  # Try simplest model first
            outputs = llm.generate([case['prompt']], sampling_params)
            
            for output in outputs:
                for completion in output.outputs:
                    token_count = len(completion.token_ids)
                    print(f"Result: {completion.text!r}")
                    print(f"Tokens: {token_count}, Stop reason: {completion.stop_reason}")
                    
                    if completion.stop_reason == "stop" and token_count < case['min_tokens']:
                        print("🚨 BUG: Early stop!")
                    else:
                        print("✅ Behavior as expected")
                        
        except Exception as e:
            print(f"Test case failed: {e}")

if __name__ == "__main__":
    print("MIN_TOKENS BUG REPRODUCTION SCRIPT")
    print("="*60)
    print("This script reproduces the bug where min_tokens is ignored")
    print("when stop sequences are encountered.")
    print()
    
    # Step 1: Collect environment info
    collect_environment_info()
    
    # Step 2: Reproduce the exact bug scenario
    reproduce_bug()
    
    # Step 3: Test additional scenarios
    # test_additional_scenarios()  # Uncomment if needed
    
    print("\n" + "="*60)
    print("REPRODUCTION COMPLETE")
    print("="*60)