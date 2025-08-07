# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚨 CRITICAL: This is a FORK of vLLM for Bug Fix & PR

**THIS REPOSITORY IS A FORK** of the main vLLM project specifically created to fix a critical bug and submit a pull request.

### 🎯 PRIMARY OBJECTIVE: Fix V1 min_tokens Bug

**Bug Description**: The V1 engine incorrectly processes stop sequences even when `min_tokens` hasn't been reached, causing premature generation termination.

**GitHub Issues Being Fixed**:
- #21950: Verify and add CI coverage for min_tokens
- #21987: Stop sequences ignored when min_tokens specified  
- #22014: Calvin's fix for the bug

**Expected vs Observed Behavior**:
- **Expected**: Generate tokens until BOTH conditions met: (1) min_tokens reached AND (2) stop sequence found
- **Observed**: Generation stops immediately when stop sequence found, ignoring min_tokens

### 🔍 Root Cause Analysis (CONFIRMED)

**TWO SEPARATE ROOT CAUSES IDENTIFIED**:

1. **Detokenizer Issue** (`vllm/v1/engine/detokenizer.py:90`):
   - `BaseIncrementalDetokenizer.update()` missing min_tokens check before stop string detection
   - Affects stop **strings** (e.g., `stop=['\n']`)

2. **LogitsProcessor Issue** (`vllm/v1/sample/logits_processor.py:401`):
   - `MinTokensLogitsProcessor` may not be working correctly
   - Affects stop **token IDs** including EOS tokens

### 🎯 V1 Engine Architecture (Bug-Specific Context)

The V1 engine has **two separate mechanisms** for stop conditions:
- **Stop Token IDs**: Handled by `MinTokensLogitsProcessor` (sampling level) ⚠️ Potentially broken
- **Stop Strings**: Handled by `BaseIncrementalDetokenizer` (text level) ❌ Missing min_tokens check

**Key V1 vs V0 Difference**:
- **V0 Engine**: Uses `vllm/model_executor/layers/sampler.py` and `vllm/engine/output_processor/stop_checker.py`
- **V1 Engine**: Uses `vllm/v1/sample/logits_processor.py` with `MinTokensLogitsProcessor`

### 📁 Critical Files for This Bug Fix

1. **Bug Location #1**: `vllm/v1/engine/detokenizer.py:90`
   - `BaseIncrementalDetokenizer.update()` method
   - **NEEDS**: min_tokens check before stop string detection

2. **Bug Location #2**: `vllm/v1/sample/logits_processor.py:401`
   - `MinTokensLogitsProcessor` class
   - **NEEDS**: Investigation of why EOS tokens aren't being properly penalized

3. **Test Suite**: `vllm/tests/v1/test_min_tokens.py`
   - Comprehensive V1-specific test coverage
   - Addresses both identified root causes

### 🧪 Reproduction Scripts Created

- `reproduce_min_tokens_simple.py`: Tests SamplingParams validation without full LLM
- `reproduce_min_tokens_bug.py`: Full reproduction script
- `bug_analysis_step1.py`: Documents findings and analysis

### 🐳 Docker & GPU Requirements Issues

**CRITICAL DEPLOYMENT CHALLENGE**: 
- **Pre-built GPU Docker images** require specific GPU compute capabilities
- **CPU-only builds** can take 45+ minutes to compile (default timeout: 10 minutes)
- **Dependency conflicts** prevent direct local installation without containers

**Docker Build Strategy**:
```bash
# CRITICAL: Use D drive for builds (C drive insufficient space)
# 1. Clone repository to D drive first:
D:
git clone https://github.com/arjunbreddy22/vllm.git vllm-fork
cd vllm-fork

# 2. CPU build with extended timeout (45+ minutes)
docker build -f Dockerfile.cpu -t vllm-cpu-fork .

# 3. Alternative with explicit build args:
docker build -f Dockerfile.cpu -t vllm-cpu-fork --build-arg VLLM_TARGET_DEVICE=cpu .

# 4. After build completes, test the container:
docker run -it vllm-cpu-fork /bin/bash
```

**Environment Constraints**:
- GPU compute capability insufficient for pre-built images
- CPU compilation requires significant time and memory (45+ minutes)
- **STORAGE**: C drive insufficient (1.96GB free) - MUST use D drive (700GB free)
- Testing requires either working GPU setup or patience for CPU builds
- **Fork Location**: https://github.com/arjunbreddy22/vllm.git

## Project Overview

vLLM is a high-throughput and memory-efficient inference and serving engine for Large Language Models (LLMs). It provides state-of-the-art serving throughput with features like PagedAttention, continuous batching, and various optimizations for CUDA, ROCm, TPU, and other accelerators.

## Key Development Commands

### Installation and Setup
```bash
# Install from source (requires CUDA/ROCm/CPU target)
pip install -e .

# Install development dependencies
pip install -r requirements/dev.txt

# Install test dependencies  
pip install -r requirements/test.txt
```

### Testing

#### 🎯 V1-Specific Testing (Critical for Bug Fix)
```bash
# Enable V1 engine for testing
export VLLM_USE_V1=1

# Run the min_tokens bug test suite
pytest tests/v1/test_min_tokens.py -v

# Run specific min_tokens test scenarios
pytest tests/v1/test_min_tokens.py::test_min_tokens_comprehensive -v
pytest tests/v1/test_min_tokens.py::test_sampling_params_validation -v

# Test with xfail markers (known bugs)
pytest tests/v1/test_min_tokens.py -v --runxfail
```

#### General Testing
```bash
# Run basic tests
pytest tests/

# Run specific test categories
pytest tests/basic_correctness/
pytest tests/models/
pytest tests/kernels/

# Run tests with specific markers
pytest -m "not distributed"
pytest -m core_model

# Run single test file
pytest tests/test_sampling_params.py
```

### Code Quality and Formatting
```bash
# Install pre-commit hooks (replaces old format.sh)
pip install -r requirements/lint.txt
pre-commit install

# Manual linting (if needed)
ruff check vllm/
mypy vllm/
```

### Building

#### 🎯 V1 Engine Development
```bash
# Enable V1 engine for development
export VLLM_USE_V1=1

# Run reproduction scripts
cd vllm && python reproduce_min_tokens_simple.py
python reproduce_min_tokens_bug.py  # Requires CUDA
python bug_analysis_step1.py
```

#### General Building  
```bash
# Build with CMake (used by setup.py)
# The build system uses CMake with target device detection
# Set VLLM_TARGET_DEVICE=cuda|rocm|tpu|cpu|neuron|xpu

# Enable verbose build
CMAKE_VERBOSE_MAKEFILE=ON pip install -e .

# Use precompiled wheels (for CUDA builds)
VLLM_USE_PRECOMPILED=1 pip install -e .
```

## Architecture Overview

### Core Components

1. **LLMEngine** (`vllm/engine/llm_engine.py`)
   - Central inference engine handling request processing
   - Manages input processing, scheduling, model execution, and output processing
   - Supports both synchronous and asynchronous operation modes

2. **AsyncLLMEngine** (`vllm/engine/async_llm_engine.py`)
   - Asynchronous wrapper around LLMEngine
   - Used by the OpenAI-compatible API server
   - Handles concurrent requests and streaming outputs

3. **Worker Architecture** (`vllm/worker/`)
   - Workers are processes that run model inference (one per accelerator device)
   - Each worker has a ModelRunner responsible for loading and executing the model
   - Supports tensor, pipeline, data, and expert parallelism

4. **Model Executor** (`vllm/model_executor/`)
   - Contains model implementations and execution logic
   - Unified model interface using VllmConfig for all models
   - Supports 50+ popular open-source models with consistent constructor signatures

### Entry Points

- **Offline Inference**: `LLM` class (`vllm/entrypoints/llm.py`)
- **API Server**: `vllm serve` command (`vllm/entrypoints/cli/main.py`) 
- **OpenAI API**: Compatible server at `vllm/entrypoints/openai/api_server.py`

### Configuration System

- **VllmConfig** (`vllm/config.py`) - Central configuration object passed throughout the system
- All classes accept this unified config object for extensibility
- Enables adding new features without changing constructor signatures across the hierarchy

## Key Development Patterns

### Model Integration
- All models use unified constructor: `__init__(self, *, vllm_config: VllmConfig, prefix: str = "")`
- Models are registered in `vllm/model_executor/models/`
- Quantization and sharding happen during model initialization, not after

### Testing Strategy
- Mix of unit tests and end-to-end integration tests
- Model tests use markers: `core_model`, `distributed`, `cpu_model`, etc.
- Most tests require GPU access for realistic validation

### Custom Operations
- CUDA kernels in `csrc/` directory
- CMake-based build system with device-specific compilation
- Support for various quantization schemes (GPTQ, AWQ, FP8, etc.)

## Directory Structure Highlights

- `vllm/` - Main Python package
- `csrc/` - CUDA/ROCm kernels and C++ extensions  
- `tests/` - Test suite organized by functionality
- `docs/` - Documentation and design documents
- `benchmarks/` - Performance benchmarking tools
- `examples/` - Usage examples for different scenarios
- `requirements/` - Platform-specific dependency files

## Platform Support

The codebase supports multiple target devices set via `VLLM_TARGET_DEVICE`:
- `cuda` - NVIDIA GPUs (primary target)
- `rocm` - AMD GPUs  
- `tpu` - Google TPUs
- `cpu` - CPU-only inference
- `neuron` - AWS Neuron devices
- `xpu` - Intel XPU devices

## Performance Considerations

- Uses PagedAttention for efficient memory management
- Supports continuous batching of requests
- CUDA graph capture for optimized execution
- Multiple quantization formats supported
- Speculative decoding and chunked prefill optimizations