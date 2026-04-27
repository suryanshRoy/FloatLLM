# FloatLLM 🚀

**A metal engine, hardware-agnostic Large Language Model (LLM) inference engine designed to run massive models on heavily memory-constrained edge devices as well as to act like an safety feature to run LLM locally.**

FloatLLM is built for a fundamental shift in local AI execution: **Dynamic Zero-Copy Memory Chunking**. 

## 🚀 The Architectural Shift
Originally, handling models larger than host RAM relied on static, layer-by-layer disk swapping (similar to traditional AirLLM implementations). However, static swapping creates massive I/O bottlenecks. 

FloatLLM abandons static swapping. Instead, it utilizes OS-level hardware interrogation to calculate exact, real-time memory boundaries, slicing standard `.gguf` neural network weights into mathematically perfect execution blocks. By leveraging native `mmap` (memory-mapping), it creates a zero-copy hardware bridge, streaming gigabytes of tensor data from SSD to RAM at bare-metal speeds without ever triggering an Out-of-Memory (OOM) panic.

This allows massive architectures (like Meta Llama 3 or any model) to execute natively on anything from an Apple Silicon Mac to a non-rooted Android device running terminals, completely offline.

---

## 🏗️ Project Architecture & Status

FloatLLM is being developed in these steps:

### The Brain (Hardware Router) - `floatllm_router.py`
The master entry point. The router dynamically interrogates the host machine's hardware, evaluating total RAM, free RAM, and SSD capacity. 
* **Hardware Agnostic:** Automatically routes compute workloads to Apple MPS, Vulkan, native ARM, or CUDA based on host detection.
* **Failsafe Math:** Calculates strict safety thresholds, ensuring a configurable buffer (default 20%) is always left free for the operating system and dynamic KV Cache context.
* **Absolute Control:** Allows users to manually force RAM limits (e.g., `--ram-limit 1`) to run multi-gigabyte models through ultra-tight memory constraints.

### ✅ The Hands (Memory Loader) - `floatllm_loader.py`
The physical memory mapper. 
* **Metadata Parsing:** Uses the official `gguf` library to scan the model header, discovering exact tensor byte offsets without loading the massive payload.
* **Dynamic Slicing:** Takes the safety limits from Phase 1 and mathematically groups hundreds of tensors into safe execution blocks.
* **Zero-Copy Streaming:** Utilizes a read-only `mmap` bridge to swap execution chunks in and out of RAM at maximum SSD read speeds. 

### 🚧 The Heart (Inference Engine) - *In Development*
Currently transitioning into Phase 3. This phase will introduce the `ctypes` Compute Bridge, connecting the Python memory logic to a bare-metal C/C++ backend for high-speed matrix multiplication. It will also introduce the dynamic KV Cache Manager to securely page conversation history to the SSD during long contexts.

### ⏳ The Voice (Tokenizer & UI) - *Planned*
The final pipeline to convert raw tensor math back into human-readable text and stream it efficiently to the terminal interface.

---

## 🛠️ Usage 

Currently we can execute the mapping pipeline to watch the engine interrogate your hardware and slice your model.
FloatLLM relies on a custom C++ backend to execute bare-metal matrix operations. Before running the router, you must compile the C++ compute bridge into a shared library natively on your machine.

**For macOS:**
```bash
clang++ -shared -fPIC -o floatllm_compute.dylib floatllm_compute.cpp
```
**For Linux**
```bash
clang++ -shared -o floatllm_compute.dil floatllm_compute.cpp
```
**For Windows**
```bash
clang++ -shared -o floatllm_compute.dil floatllm_compute.cpp
```