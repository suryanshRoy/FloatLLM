# FloatLLM 🚀 (Work in Progress)

FloatLLM is a custom project I'm building to run Large Language Models (LLMs) completely offline on almost any device. Right now, running local AI usually requires expensive graphics cards with tons of memory. I want to fix that by making an engine that adjusts to whatever device it's running on, even if it doesn't have much RAM. By combining the speed of the **GGUF** format with the memory-saving tricks of **AirLLM**, this project allows you to run massive AI models even if you have very little RAM. 

## 🏗️ Core Architecture

The system is designed to be universally scalable and relies on three main pillars to give users total control and flexibility:

### 1. The Core: Moving to GGUF 
FloatLLM uses **GGUF** as its primary local format.
* **Plug-and-play:** All the model data is stored in one file.
* **Speed & Flexibility:** It allows for instant loading and runs natively across almost all hardware architectures and accelerators, including **ARM, MPS (Apple Silicon), CUDA, NPUs, XPUs, and standard CPUs/GPUs**.

### 2. The Engine: Hybrid Inference Strategy
We use a three-tier system so the engine can adapt to your specific device:
* **⚡ Turbo Mode:** Uses compressed (4-bit/6-bit) files for high-speed, offline chatting. Great for everyday use on laptops and phones.
* **🧠 Pro Mode:** Uses "Smart Quantization." It keeps the important parts of the model highly accurate while compressing the less important parts, giving you a perfect balance of speed and accuracy.
* **🐢 Ultra Mode (AirLLM-Style):** For when you need 100% accuracy. It reads the massive, uncompressed 16-bit model directly from your **SSD layer-by-layer**. It's slower, but it lets you run massive 70B models on a low-end device with just 4GB of RAM!

### 3. The "Freedom" Feature: Local Quantization
Usually, you have to download pre-shrunk models from the internet or do it on your high-end devices. FloatLLM changes that. Using AirLLM's layer-swapping logic, FloatLLM lets you take a massive 16-bit model and "shrink" it down directly on your low-end device without crashing your RAM. This means you stay completely offline and private, even while optimizing your models.

## 🙏 Acknowledgements
A massive shoutout to the **[airllm](https://github.com/lyogavin/airllm)** project. Their logic for layer-swapping is what makes many features possible!

## 🚀 Current Status
**v0.1 - Groundwork**
Currently laying down the foundations for the GGUF integration and the 3-tier engine strategy. 