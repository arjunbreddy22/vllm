# vLLM Development Guide

## Architecture Overview

vLLM has two main architectures:

### v1 Architecture (Primary Focus)
- **Location**: `vllm/v1/` directory
- **Status**: Enabled by default, actively developed
- **Key Features**: 
  - Simplified, modular codebase
  - Unified scheduler for prompt and output tokens
  - Near-zero CPU overhead
  - Built-in optimizations (prefix caching, chunked prefill, etc.)
  - Uses torch.compile with piecewise cudagraph

### v0 Architecture (Legacy)
- **Status**: Being deprecated (see RFC #18571)
- **Location**: Main `vllm/` directory (excluding `vllm/v1/`)
- **Disable v1 with**: `VLLM_USE_V1=0` environment variable

## Development Focus

**Work primarily with v1 architecture** as v0 will be deprecated soon. The v1 architecture provides:

- Better performance, especially for long context scenarios
- Cleaner, more maintainable code structure
- Modern optimizations enabled by default
- Simplified core systems (scheduler, KV cache manager, worker, sampler)

## Key v1 Components

- **Core**: `vllm/v1/core/` - Scheduling, KV cache management
- **Engine**: `vllm/v1/engine/` - Main engine logic
- **Worker**: `vllm/v1/worker/` - GPU/TPU workers
- **Attention**: `vllm/v1/attention/` - Attention backends
- **Sampling**: `vllm/v1/sample/` - Token sampling logic

## Testing

- v1 tests: `tests/v1/`
- v0 tests: Other test directories

When contributing, focus on v1 implementations and ensure compatibility with the unified scheduler design.

## Current Optimization Target: Tensor Swap Performance

### The Problem

**Context**: In `vllm/v1/worker/gpu_input_batch.py`, there's a `swap_states()` function that swaps two requests in a batch during scheduling/reordering.

**The Issue**:
```python
# This is what they WANT to do (but it's unsafe):
# self.token_ids_cpu[i1, ...], self.token_ids_cpu[i2, ...] = \
#     self.token_ids_cpu[i2, ...], self.token_ids_cpu[i1, ...]

# So they do this instead (safe but inefficient):
tmp = self.token_ids_cpu[i1, ...].copy()  # ← COPIES ENTIRE ROW
self.token_ids_cpu[i1, ...] = self.token_ids_cpu[i2, ...]
self.token_ids_cpu[i2, ...] = tmp
```

**Why is tuple swap "unsafe"?**

Actually, `token_ids_cpu` is a NumPy array created from a PyTorch tensor:
```python
self.token_ids_cpu_tensor = torch.zeros((max_num_reqs, max_model_len), device="cpu")
self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()  # NumPy array sharing memory
```

**The original author's concern**: The NumPy array shares memory with the PyTorch tensor, which *could* theoretically cause issues with concurrent access or memory management conflicts between PyTorch and NumPy.

**However**: For normal NumPy arrays, tuple swapping like `a[i], a[j] = a[j], a[i]` is typically safe and commonly used. The original author may have been **overly cautious** about the PyTorch-NumPy memory sharing.

**Potential PR discussion point**: This raises an interesting question - is the tuple swap actually unsafe, or could this be simplified further? The current `.copy()` approach is definitely safe, but the "unsafe" designation might warrant investigation in a future optimization.

### The Inefficiency

**Current behavior**:
- `token_ids_cpu` shape: `[batch_size, max_model_len]`
- `max_model_len` could be 32K, 64K, or even larger
- When swapping requests, copies THE ENTIRE row: `token_ids_cpu[i1, ...]`
- Most requests are much shorter (hundreds to thousands of tokens)

**Example**:
- max_model_len = 32,768 tokens
- Actual request lengths = 500 and 800 tokens
- Current: Copies 32,768 tokens (131KB at 4 bytes/token)
- Needed: Copy only ~800 tokens (3.2KB)
- **Waste: 40x more memory bandwidth than necessary**

### Our Optimization Strategy

**Core insight**: Only copy the "valid indices" - the actual tokens, not the padding.

**Solution 1 (Simple & Safe)**:
```python
# Get actual token counts for each request
num_tokens_i1 = self.num_tokens[i1]  # e.g., 500 tokens
num_tokens_i2 = self.num_tokens[i2]  # e.g., 800 tokens
max_tokens = max(num_tokens_i1, num_tokens_i2)  # 800

# Only copy valid token range
tmp = self.token_ids_cpu[i1, :max_tokens].copy()  # Copy 800 tokens, not 32K
self.token_ids_cpu[i1, :max_tokens] = self.token_ids_cpu[i2, :max_tokens]
self.token_ids_cpu[i2, :max_tokens] = tmp
```

**Why this works**:
- Preserves all actual token data
- Maintains safety (still uses .copy())
- Ignores padding tokens (they're unused anyway)
- Dramatically reduces memory bandwidth

### Expected Performance Impact

**Memory bandwidth reduction**:
- Typical case: 10-50x less data copied
- Long sequences: 2-5x less data copied
- Short sequences: 50-100x less data copied

**When this matters**:
- During batch reordering (happens frequently)
- High-throughput scenarios with many requests
- Long context models where max_model_len is very large

**Measurable metrics**:
- Swap latency (microseconds)
- Memory bandwidth usage
- Overall batching throughput

### Potential Concerns & Validation

**Correctness risks**:
1. **num_tokens accuracy**: What if `self.num_tokens[i]` is wrong?
2. **Padding semantics**: Do any consumers expect specific padding values?
3. **Tensor indexing**: Edge cases with empty sequences or max-length sequences

**Validation approach**:
1. **Benchmark current performance** first
2. **Add assertions** to verify num_tokens accuracy
3. **Test edge cases**: empty sequences, max-length sequences, mismatched lengths
4. **Regression tests** to ensure no behavioral changes

### Why This Is Resume-Worthy

1. **Systems performance**: Demonstrates understanding of memory optimization
2. **Real impact**: Affects hot path in production inference
3. **Measurable**: Clear before/after performance numbers
4. **Production-ready**: Simple, safe optimization with clear benefits
5. **Open source contribution**: Shows ability to read, understand, and improve complex codebases

This is exactly the kind of optimization that shows both technical depth and practical impact - perfect for demonstrating performance engineering skills!

## Benchmark Implementation

Created `benchmarks/overheads/benchmark_tensor_swap.py` to measure current performance before optimization:

### Key Features
- **Leverages existing test infrastructure** - Uses same `InputBatch` and `CachedRequestState` setup from tests
- **Tests realistic scenarios**:
  - Short requests (100-500 tokens)
  - Medium requests (1K-5K tokens) 
  - Long requests (10K-30K tokens)
  - Mixed scenarios (short + long)
- **Measures actual performance** - Times `swap_states()` with microsecond precision
- **Calculates potential improvement** - Shows theoretical efficiency gains
- **Configurable** - Command line args for device, max_model_len, trials

### Usage
```bash
python benchmarks/overheads/benchmark_tensor_swap.py --max-model-len 32768
```

### What It Measures
- **Baseline timing** before optimization
- **Performance across different request sizes** 
- **Theoretical speedup potential** (e.g., "40x improvement possible")
- **Concrete numbers for PR description**

The benchmark creates requests with specific token counts, times the swap operation, and calculates how much improvement we could get by only copying valid tokens instead of entire rows.

This will provide the "before" numbers to compare against after implementing the optimization, giving us concrete performance data for the PR submission.

## PR Strategy and Discussion Points

### Primary Optimization
Our main contribution: **Only copy valid token indices instead of entire tensor rows**

### Additional Insight to Mention
**The "unsafe" tuple swap question**: In our analysis, we discovered that the original "unsafe" designation for tuple swapping might be overly cautious:

- `token_ids_cpu` is a NumPy array (not a PyTorch tensor as initially assumed)
- Standard NumPy tuple swapping is typically safe: `a[i], a[j] = a[j], a[i]`
- The concern appears to be PyTorch-NumPy memory sharing, but this may not actually pose risks

**PR value**: Mentioning this shows:
1. **Deep code analysis** - We understand the implementation details beyond just the surface optimization
2. **Potential for future improvement** - Could investigate eliminating `.copy()` entirely in a future PR
3. **Technical rigor** - We question assumptions and validate the actual necessity of current "safe" approaches

**Recommended PR tone**: "While implementing the current optimization, we noticed the tuple swap 'unsafe' designation might be overly cautious. For this PR, we maintain the safe `.copy()` approach while dramatically improving performance. The tuple swap question could be worth investigating in future optimizations."

This demonstrates both **immediate practical value** (our optimization) and **longer-term technical insight** (potential further improvements).