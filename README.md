# FloatLLM

A memory-aware LLM inference engine for edge devices.

FloatLLM is a inference engine that uses memory-mapping for GGUF models, keeps RAM usage bounded, and streams tensors through ggml backends instead of loading the entire model into memory at once into memory.

> Current Status: experimental (In early development stage)



## Why FloatLLM? 

Most inference engines assume the model can work perfectly in available RAM or VRAM. If the model is too large then the device could crash or the inference could fail.

But FloatLLM prioritizes memory safety and keeps RAM usage bounded while still allowing inference to run on large models even on constrained hardware.

FloatLLM takes a different approach:

- It `mmap`s the `.gguf` file instead of copying the whole model into memory.
- It keeps tensor handling zero-copy.
- It slices the model into chunks(layer).
- It contains safety checks to avoid crashes when memory pressure gets too high.

## What is currently working?

- GGUF model loading through memory mapping
- Backend selection and hardware detection
- Tokenizer loading from GGUF vocab metadata
- RAM and storage safety checks
- Chunking tensors into memory-budgeted blocks
- A CLI entry point for loading a model and running a prompt

## Parts that are still in progress

- Full transformer layer execution
- KV cache support
- Better tokenization behavior for all model families
- Session persistence
- Sampling controls like temperature, top-k, and top-p
- CLI flags and better CLI user interface

---

## Getting Started

### 1. Build FloatLLM
Clone this repository and install the required dependencies for your platform:
```bash
git clone https://github.com/suryanshRoy/FloatLLM.git
cd FloatLLM
```

Clone the required ggml library:
```bash
git clone https://github.com/ggml-org/ggml
```

> Make sure to have CMake installed on your system. You can download it from [here](https://cmake.org/download/).


```bash
cmake -B build -DGGML_DIR=/path/to/ggml
cmake --build build --config Release -j 4 --target runFloatLLM
```

Backend flags depend on your platform for GPU acceleration:

```bash
# NVIDIA CUDA
cmake -B build -DGGML_DIR=/path/to/ggml -DGGML_CUDA=ON

# Vulkan
cmake -B build -DGGML_DIR=/path/to/ggml -DGGML_VULKAN=ON

# OpenCL
cmake -B build -DGGML_DIR=/path/to/ggml -DGGML_OPENCL=ON

# SYCL
cmake -B build -DGGML_DIR=/path/to/ggml -DGGML_SYCL=ON

# Kompute
cmake -B build -DGGML_DIR=/path/to/ggml -DGGML_KOMPUTE=ON
```

### 2. Run a model

```bash
./runFloatLLM \
  --hardware auto \
  --model-path ./model.gguf \
  --prompt "<your prompt here>"
```

If your system is memory constrained, you can also tune the safety settings:

```bash
./runFloatLLM \
  --hardware auto \
  --model-path ./model.gguf \
  --prompt "<your prompt here>" \
  --crash-threshold 200 \
  --ram-limit 4 \
  --ram-buffer 0.20
```


In the current implementation:

- `FloatLoader` handles GGUF parsing and memory mapping.
- `FloatEngine` owns backend setup and tensor registration.
- `FloatTokenizer` reads vocabulary and performs encode/decode.
- `FloatUI` prints memory stats and other UI/UX.


## Goals

- Run locally, without cloud calls
- Keep RAM usage bounded and avoid crashes
- Be portable across CPU and GPU backends
- Give users control over settings

---

### This project is under MIT License. See LICENSE for more details.