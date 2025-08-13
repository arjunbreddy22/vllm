# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
import numpy as np
import torch
from typing import List

from vllm.sampling_params import SamplingParams
from vllm.utils import FlexibleArgumentParser, is_pin_memory_available
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch


def _create_sampling_params():
    return SamplingParams(
        temperature=1.0,
        top_p=0.9,
        top_k=10,
        frequency_penalty=0.1,
        presence_penalty=0.2,
        repetition_penalty=1.1,
    )


def _construct_request_with_tokens(req_id_suffix: int, num_tokens: int, vocab_size: int):
    """Create a request with a specific number of tokens."""
    # Split between prompt and output tokens
    num_prompt_tokens = max(1, num_tokens // 2)
    num_output_tokens = num_tokens - num_prompt_tokens
    
    prompt_token_ids = [
        np.random.randint(0, vocab_size) for _ in range(num_prompt_tokens)
    ]
    output_token_ids = [
        np.random.randint(0, vocab_size) for _ in range(num_output_tokens)
    ]
    
    return CachedRequestState(
        req_id=f"req_id_{req_id_suffix}",
        prompt_token_ids=prompt_token_ids,
        sampling_params=_create_sampling_params(),
        pooling_params=None,
        mm_inputs=[],
        mm_positions=[],
        block_ids=([], ),
        generator=None,
        num_computed_tokens=len(output_token_ids),
        output_token_ids=output_token_ids,
    )


def benchmark_swap_performance(
    device: str,
    batch_size: int,
    max_model_len: int,
    request_sizes: List[int],
    vocab_size: int = 1024,
    num_trials: int = 100
):
    """Benchmark tensor swap performance with different request sizes."""
    
    input_batch = InputBatch(
        max_num_reqs=batch_size,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len * batch_size,
        device=torch.device(device),
        pin_memory=is_pin_memory_available(),
        vocab_size=vocab_size,
        block_sizes=[1],
    )
    
    # Add requests with specified sizes
    for req_index in range(batch_size):
        num_tokens = request_sizes[req_index % len(request_sizes)]
        req = _construct_request_with_tokens(req_index, num_tokens, vocab_size)
        assigned_req_index = input_batch.add_request(req)
        assert assigned_req_index == req_index
    
    # Warm up
    for _ in range(10):
        input_batch.swap_states(0, 1)
    
    # Benchmark swap operations
    swap_times = []
    for _ in range(num_trials):
        i1, i2 = 0, 1  # Always swap first two requests
        
        start_time = time.perf_counter()
        input_batch.swap_states(i1, i2)
        end_time = time.perf_counter()
        
        swap_times.append((end_time - start_time) * 1_000_000)  # Convert to microseconds
        
        # Swap back to restore original state
        input_batch.swap_states(i1, i2)
    
    return swap_times


def main(args):
    print("=== vLLM Tensor Swap Performance Benchmark ===")
    print(f"Device: {args.device}")
    print(f"Max model length: {args.max_model_len}")
    print(f"Trials per scenario: {args.num_trials}")
    print()
    
    scenarios = [
        ("Short requests (100-500 tokens)", [100, 200, 300, 400, 500]),
        ("Medium requests (1K-5K tokens)", [1000, 2000, 3000, 4000, 5000]),
        ("Long requests (10K-30K tokens)", [10000, 15000, 20000, 25000, 30000]),
        ("Mixed: short + long", [100, 500, 15000, 25000]),
        ("Mixed: all sizes", [100, 1000, 5000, 10000, 20000, 30000]),
    ]
    
    results = []
    
    for scenario_name, request_sizes in scenarios:
        # Skip scenarios where tokens exceed max_model_len
        if max(request_sizes) > args.max_model_len:
            print(f"⏭️  Skipping {scenario_name} (exceeds max_model_len={args.max_model_len})")
            continue
            
        print(f"🧪 Testing {scenario_name}...")
        
        batch_size = max(len(request_sizes), 8)  # Ensure we have enough requests
        swap_times = benchmark_swap_performance(
            device=args.device,
            batch_size=batch_size,
            max_model_len=args.max_model_len,
            request_sizes=request_sizes,
            num_trials=args.num_trials
        )
        
        avg_time = np.mean(swap_times)
        std_time = np.std(swap_times)
        min_time = np.min(swap_times)
        max_time = np.max(swap_times)
        
        # Calculate data copied (rough estimate)
        max_tokens_copied = max(request_sizes) * 2  # Copy + assign operations
        total_possible = args.max_model_len * 2  # What current implementation copies
        efficiency_gain = total_possible / max_tokens_copied if max_tokens_copied > 0 else 1
        
        print(f"   Average: {avg_time:.1f} ± {std_time:.1f} μs")
        print(f"   Range: {min_time:.1f} - {max_time:.1f} μs")
        print(f"   Potential efficiency gain: {efficiency_gain:.1f}x")
        print()
        
        results.append({
            'scenario': scenario_name,
            'avg_time': avg_time,
            'request_sizes': request_sizes,
            'efficiency_gain': efficiency_gain
        })
    
    print("=== Summary ===")
    for result in results:
        print(f"{result['scenario']}: {result['avg_time']:.1f} μs "
              f"(potential {result['efficiency_gain']:.1f}x speedup)")
    
    print("\n💡 The optimization opportunity:")
    print("   Current implementation copies entire tensor rows (max_model_len)")
    print("   Optimized implementation would copy only valid tokens")
    print("   Expected improvement: 10-50x reduction in memory bandwidth")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark tensor swap performance in GPU input batch"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0", 
        help="Device to run benchmark on"
    )
    parser.add_argument(
        "--max-model-len", 
        type=int, 
        default=32768, 
        help="Maximum model length (affects tensor size)"
    )
    parser.add_argument(
        "--num-trials", 
        type=int, 
        default=100, 
        help="Number of trials per scenario"
    )
    
    args = parser.parse_args()
    main(args)