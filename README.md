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

### ✅ The Brain (Hardware Router) - `floatllm_router.py`
The master entry point. The router dynamically interrogates the host machine's hardware, evaluating total RAM, free RAM, and SSD capacity. 
* **Hardware Agnostic:** Automatically routes compute workloads based on host detection.
* **Failsafe Math:** Calculates strict safety thresholds, ensuring a configurable buffer (default 20%) is always left free for the operating system and dynamic KV Cache context.
* **Absolute Control:** Allows users to manually force RAM limits to run multi-gigabyte models through ultra-tight memory constraints.

### ✅ The Hands (Memory Loader) - `floatllm_loader.py`
The physical memory mapper. 
* **Metadata Parsing:** Uses the official `gguf` library to scan the model header, discovering exact tensor byte offsets without loading the massive payload.
* **Dynamic Slicing:** Takes the safety limits from Phase 1 and mathematically groups hundreds of tensors into safe execution blocks.
* **Zero-Copy Streaming:** Utilizes a read-only `mmap` bridge to swap execution chunks in and out of RAM at maximum SSD read speeds. 

### 🚧 The Heart (Inference Engine) - *In Development*
Currently introducing the `ctypes` Compute Bridge, connecting the Python memory logic to a bare-metal C/C++ backend for high-speed matrix multiplication. It will also introduce the dynamic KV Cache Manager to securely page conversation history to the SSD during long contexts.

### ⏳ The Voice (Tokenizer & UI) - *Planned*
The final pipeline to convert raw tensor math back into human-readable text and stream it efficiently to the terminal interface.

---

## 🛠️ Usage 

FloatLLM relies on a custom C++ backend (Compute Bridge) to execute bare-metal matrix operations. Before running the router, you must compile the C++ compute bridge into a shared library natively on your machine using CMake.

### 1. Build the Compute Bridge

**For (Apple Silicon/MPS):**
```bash
rm -rf build 
cmake -B build
cmake --build build --config Release -j 4
```
**For (NVIDIA GPU / CUDA):**
```bash
rm -rf build
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j 4
```
**For (Vulkan GPU):**
```bash
rm -rf build
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j 4
```
