# FloatLLM 🚀

**A bare-metal, hardware-agnostic Large Language Model (LLM) inference engine designed to run massive models on heavily memory-constrained edge devices, and to act as a safety feature for running LLMs locally.**

FloatLLM is built for a fundamental shift in local AI execution: **Dynamic Zero-Copy Memory Chunking**. 

## 🚀 The Architectural Shift
Originally, handling models larger than host RAM relied on static, layer-by-layer disk swapping. However, static swapping creates massive I/O bottlenecks. 

FloatLLM abandons static swapping. Instead, it utilizes OS-level hardware interrogation to calculate exact, real-time memory boundaries, slicing standard `.gguf` neural network weights into mathematically perfect execution blocks. By leveraging native `mmap` (memory-mapping), it creates a zero-copy hardware bridge, streaming gigabytes of tensor data from SSD to RAM at bare-metal speeds without ever triggering an Out-of-Memory (OOM) panic.

This allows massive architectures to execute natively on anything from an Apple Silicon Mac to a non-rooted Android device running terminal environments, completely offline.

---

## 🏗️ Project Architecture & Status

FloatLLM is being developed in these stages:

### ✅ Phase 1 (Hardware Router) - `floatllm_router.py`
The master entry point. The router dynamically interrogates the host machine's hardware, evaluating total RAM, free RAM, and SSD capacity. 
* **Hardware Agnostic:** Automatically routes compute workloads based on host detection.
* **Failsafe Math:** Calculates strict safety thresholds, ensuring a configurable buffer (default 20%) is always left free for the operating system and dynamic KV Cache context.
* **Absolute Control:** Allows users to manually force RAM limits to run multi-gigabyte models through ultra-tight memory constraints.

### ✅ Phase 2 (Memory Loader) - `floatllm_loader.py`
The physical memory mapper. 
* **Metadata Parsing:** Uses the official `gguf` library to scan the model header, discovering exact tensor byte offsets without loading the massive payload.
* **Dynamic Slicing:** Takes the safety limits and mathematically groups hundreds of tensors into safe execution blocks.
* **Zero-Copy Streaming:** Utilizes a read-only `mmap` bridge to swap execution chunks in and out of RAM at maximum SSD read speeds. 

### ✅ Phase 3 (Inference Engine) - floatllm_compute.cpp
The bare-metal execution layer utilizing `ggml`.
* **Hardware Binding:** Dynamically binds zero-copy Python memory maps to dedicated GPU cores (Metal, CUDA, Vulkan).
* **VRAM Detachment:** Securely detaches CPU memory pointers to prevent OS-level segmentation faults, allowing the GPU allocator to provision safe computational VRAM on the fly.

### ✅ Phase 4 (Tokenizer) - floatllm_tokenizer.py
The translation layer.
* **100% Offline Generation:** Dynamically reads the internal `tokenizer.ggml.tokens` array directly from the GGUF file. Zero API calls, zero internet dependency.
* **Dynamic Handling:** Automatically scales between 1B and 405B parameter models, supporting all standard tokenization architectures.

### Phase 5 (Generation loop & Transforemer Brain) - In Development
The output interface
* **Generation loop:** Pipeline integrated. Prompt integers are passed securely across the `ctypes` bridge, processed through the GPU, and streamed back horizontally to the user terminal in real-time.

---

## 🛠️ Usage 

FloatLLM relies on a custom C++ backend (Compute Bridge) to execute bare-metal matrix operations. Before running the router, you must compile the C++ compute bridge into a shared library natively on your machine using CMake.

### 1. Build the Compute Bridge

**For Apple Silicon (Metal/MPS):**
```bash
rm -rf build 
cmake -B build -DGGML_DIR=../ggml
cmake --build build --config Release -j 4
```
**For NVIDIA GPU (CUDA):**
```bash
rm -rf build
cmake -B build -DGGML_CUDA=ON -DGGML_DIR=../ggml
cmake --build build --config Release -j 4
```
**For Vulkan GPU:**
```bash
rm -rf build
cmake -B build -DGGML_VULKAN=ON -DGGML_DIR=../ggml
cmake --build build --config Release -j 4
```
**For OpenCL:**
```bash
rm -rf build
cmake -B build -DGGML_OPENCL=ON -DGGML_DIR=../ggml
cmake --build build --config Release -j 4
```
**For SYCL (Intel OneAPI):**
```bash
rm -rf build
cmake -B build -DGGML_SYCL=ON -DGGML_DIR=../ggml
cmake --build build --config Release -j 4
```
**For Kompute / DirectX:**
```bash
rm -rf build
cmake -B build -DGGML_KOMPUTE=ON -DGGML_DIR=../ggml
cmake --build build --config Release -j 4
```
**For CPU-Only / Native ARM:**
```bash
rm -rf build
cmake -B build -DGGML_DIR=../ggml
cmake --build build --config Release -j 4
``` 

### 2. Run the Engine
* Execute the router, pointing it to a local .gguf file:
```bash
python floatllm_router.py --model-path /path/to/your/model.gguf --prompt "What is the capital of France?"
```