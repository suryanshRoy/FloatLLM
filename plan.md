# FloatLLM — AI Assistant Context File

> This file is for AI assistants (Claude, Copilot, etc.) helping with FloatLLM.
> Read this before touching any code. It will save you from making wrong assumptions.

---

## What FloatLLM Actually Is

FloatLLM is a **bare-metal C++ LLM inference engine** built on top of `ggml` (the tensor library behind llama.cpp).

Its core innovation is **Dynamic Zero-Copy Memory Chunking**:
- It `mmap()`s the `.gguf` model file directly into virtual address space
- It never copies tensor data into a separate RAM buffer — the OS pages it in on demand
- It slices tensors into chunks that fit within a user-defined RAM budget
- It streams those chunks through the GPU/CPU compute backend one at a time

This allows models **larger than available RAM** to run on edge devices (phones, low-RAM laptops) without OOM crashes.

---

## Project Structure

```
FloatLLM/
├── src/
│   ├── FloatLLM.h          # All class declarations + includes
│   ├── FloatLLM.cpp        # main() — CLI parsing, orchestration
│   ├── FloatEngine.cpp     # ComputeEngine — backend init, tensor mapping, forward pass
│   ├── FloatLoader.cpp     # Loader — mmap, GGUF parsing, chunking, CLI args
│   ├── FloatTokenizer.cpp  # Tokenizer — BPE encode/decode from GGUF vocab
│   ├── FloatUI.cpp         # TerminalUI — RAM/storage stats, safety checks
│   └── FloatUI.h           # TerminalUI class declaration
├── CMakeLists.txt          # Hardware-agnostic build (no if(APPLE) guards for backends)
└── PLAN.md                 # This file
```

---

## Key Classes

### `ComputeEngine` (FloatEngine.cpp)
- `init(backend_name, total_tensors)` — loads all ggml backends, picks hardware, allocates graph context
- `map_tensor(name, type, ptr, ne0..ne3, chunk_id)` — registers a zero-copy tensor into `tensor_registry`
- `forward_pass(tokens, num_tokens)` — runs inference, returns next token ID
- `detect_hardware()` — static, returns backend string based on OS/hardware
- `shutdown()` — frees all ggml contexts, backend, allocator

### `Tokenizer` (FloatTokenizer.cpp)
- Reads vocab from GGUF metadata key `tokenizer.ggml.tokens`
- BPE-style greedy longest-match encode
- Ġ (U+0120) used as space prefix marker, same as GPT-2/Qwen convention
- `encode(text)` → `vector<int32_t>`
- `decode(token_ids)` → `string`

### `Loader` (FloatLoader.cpp)
- `mmap()`s the model file (POSIX) or `MapViewOfFile` (Windows)
- `parse_gguf_metadata()` → `vector<TensorInfo>` (name, type, offset, size, shape)
- `build_dynamic_chunks(tensors)` — greedy bin-packing into RAM-budget chunks
- `stream_all_chunks()` → calls `engine.map_tensor()` for each tensor via reference

### `TerminalUI` (FloatUI.cpp)
- `check_threshold(...)` — pre-flight safety check, prints dashboard, exits if unsafe
- `abort_if_overloaded()` — called every token generation step, exits if RAM > 95%
- `get_ram_stats_mb()` — Windows: `GlobalMemoryStatusEx`, Apple: `mach`, Linux: **currently broken** (returns `total * 0.5` hardcoded — needs `/proc/meminfo`)
- `get_storage_stats_gb()` — `statvfs` on POSIX, `GetDiskFreeSpaceEx` on Windows

---

## Critical Constraints — Do Not Violate

1. **CMakeLists.txt must stay hardware-agnostic.** No `if(APPLE)`, `if(WIN32)`, or `if(UNIX)` guards that force-enable backends. Only `GGML_CUDA`, `GGML_VULKAN`, etc. flags drive backend linking. The one exception already in the file (`if(APPLE) target_link_libraries(... ggml-metal)`) should eventually be removed and replaced with a cmake option flag.

2. **No global state.** The `Loader` holds a reference to `ComputeEngine` — it does not use globals or singletons. Keep it this way.

3. **Zero-copy must be preserved.** When adding new tensor operations, never `memcpy` tensor data into a new buffer. Always work with the mmap pointer directly.

4. **ggml 2025 API patterns.** This project uses the modern ggml API:
   - `ggml_backend_load_all()` for backend registry
   - `ggml_gallocr_new()` / `ggml_gallocr_alloc_graph()` for graph allocation
   - `ggml_backend_tensor_set()` / `ggml_backend_tensor_get()` for data transfer
   - `ggml_rope_ext()` for RoPE (not the deprecated `ggml_rope()`)
   - `ggml_soft_max_ext()` for attention softmax
   - `ggml-cpu.h` is separate from `ggml.h`

5. **Namespace:** Everything lives in `namespace floatllm`. Keep it that way.

6. **C++17.** `CMakeLists.txt` sets `CMAKE_CXX_STANDARD 17`. Use structured bindings, `if constexpr`, etc. freely.

---

## Current State of `forward_pass()` — The Big Gap

The current `forward_pass()` is **not a real transformer**. It only does:
```
token_ids → embedding lookup → matmul with output weight → argmax
```

It skips all transformer layers (`blk.N.*` tensors). This means it technically runs but produces garbage output for real models.

### What a real forward pass needs (Qwen2 / LLaMA architecture):

```
Input token IDs
    → token_embd.weight [vocab_size, hidden_dim]          # embedding lookup
    → for each layer N (blk.N.*):
        → blk.N.attn_norm.weight                          # RMSNorm
        → blk.N.attn_q.weight / attn_k.weight / attn_v.weight  # QKV projections
        → RoPE on Q and K (ggml_rope_ext)
        → Attention scores: Q @ K^T / sqrt(head_dim)
        → ggml_soft_max_ext (with causal mask)
        → scores @ V
        → blk.N.attn_output.weight                        # output projection
        → residual add
        → blk.N.ffn_norm.weight                           # RMSNorm
        → blk.N.ffn_gate.weight / ffn_up.weight / ffn_down.weight  # SwiGLU FFN
        → residual add
    → output_norm.weight                                  # final RMSNorm
    → output.weight                                       # logit projection
    → argmax (or sample) over vocab
```

Tensor names in GGUF follow the pattern above exactly. They are already loaded into `tensor_registry` — just look them up by name.

---

## Roadmap — Ordered by Priority

### Phase 1 — Fix Foundations (do these first, they're safety-critical or quick wins)

**1.1 Fix Linux RAM stats** (`FloatUI.cpp` → `get_ram_stats_mb()`)
- Current code falls through to `return {total, total * 0.5}` on Linux — a hardcoded 50% guess
- Fix: read `/proc/meminfo`, parse `MemTotal:` and `MemAvailable:` lines
- This is a safety feature — it must be accurate

**1.2 Add `--max-tokens` CLI flag** (`FloatLoader.cpp` → `parse_args`, `FloatLLM.h` → `CliOptions`, `FloatLLM.cpp` → generation loop)
- Currently hardcoded as `const int max_tokens_to_generate = 60` with a `REVIEW` comment
- Add `int max_tokens = 60` to `CliOptions`, parse `--max-tokens N` in `parse_args`, use it in the loop

**1.3 Add `check_system_status()`** (`FloatUI.h` + `FloatUI.cpp`)
- Mentioned in README checklist as done, but missing entirely
- Simple diagnostic: calls `get_ram_stats_mb()` + `get_storage_stats_gb()`, prints a health table
- Signature: `static void check_system_status()`

**1.4 Add `quantize_memory()` utility** (new file `src/FloatUtils.cpp` or add to `FloatUI.cpp`)
- Also listed in README as done, missing entirely
- Converts a buffer of F32 values to F16 using `ggml_fp32_to_fp16`
- Signature: `static void quantize_memory(float* src, ggml_fp16_t* dst, size_t count)`

---

### Phase 2 — Real Transformer Forward Pass (the core AI work)

**2.1 Read model hyperparameters from GGUF metadata**
- Keys to read: `llama.block_count`, `llama.embedding_length`, `llama.attention.head_count`, `llama.attention.head_count_kv`, `llama.rope.freq_base`, `llama.context_length`
- Store in a `ModelConfig` struct inside `ComputeEngine` or as a separate struct in `FloatLLM.h`
- These are needed before building the compute graph

**2.2 Implement RMSNorm helper**
- `ggml_rms_norm()` + `ggml_mul()` with the norm weight tensor
- Will be called at the start of every attention and FFN block

**2.3 Implement single attention layer**
- Look up `blk.N.attn_q.weight`, `blk.N.attn_k.weight`, `blk.N.attn_v.weight`, `blk.N.attn_output.weight`
- Project: `Q = x @ Wq`, `K = x @ Wk`, `V = x @ Wv`
- Reshape Q/K/V into `[head_dim, n_heads, seq_len]`
- Apply RoPE: `ggml_rope_ext(Q, ...)`, `ggml_rope_ext(K, ...)`
- Attention: `scores = Q @ K^T`, scale, `ggml_soft_max_ext`, `out = scores @ V`
- Project output: `out @ Wo`, add residual

**2.4 Implement single FFN layer (SwiGLU)**
- Look up `blk.N.ffn_gate.weight`, `blk.N.ffn_up.weight`, `blk.N.ffn_down.weight`
- `gate = x @ Wgate`, `up = x @ Wup`
- `hidden = ggml_silu(gate) * up` (element-wise)
- `out = hidden @ Wdown`, add residual

**2.5 Loop over all N layers**
- `n_layers` comes from `llama.block_count` (read in 2.1)
- For each layer: RMSNorm → Attention → residual → RMSNorm → FFN → residual

**2.6 Final norm + logit projection**
- Apply `output_norm.weight` RMSNorm to last hidden state
- Multiply by `output.weight` to get logits over vocab
- Argmax (current) or add temperature sampling

---

### Phase 3 — KV Cache (makes generation actually usable)

**3.1 Allocate KV cache tensors**
- One `k_cache` and `v_cache` per layer: shape `[head_dim, n_heads_kv, context_length]`
- Allocate once in `ComputeEngine::init()`, stored in a `vector<KVLayer>` struct

**3.2 Slice-and-update pattern**
- Each forward pass: slice the cache at position `current_pos`, write new K/V
- Use `ggml_view_*` to create views into the cache without copying

**3.3 Expose `current_pos` to the generation loop**
- The loop in `FloatLLM.cpp` must track token position and pass it to `forward_pass()`
- Change signature to `forward_pass(tokens, num_tokens, int pos)`

---

### Phase 4 — Session Persistence (nice to have)

**4.1 Save/load token history**
- On `--temp-chat false` (default), serialize `token_ids` to `~/.floatllm/<session_id>.bin`
- On startup, load previous tokens if session file exists
- This makes the "PERSISTENT" label in the dashboard actually true

---

## Known Bugs to Fix Along the Way

| Location | Bug | Fix |
|----------|-----|-----|
| `FloatUI.cpp` `get_ram_stats_mb()` | Linux returns `total * 0.5` hardcoded | Read `/proc/meminfo` |
| `FloatEngine.cpp` `forward_pass()` | 16MB graph allocation hardcoded comment says "great for 405B" — it's not, it's just metadata overhead | Scale with `n_layers * overhead` |
| `FloatEngine.cpp` `forward_pass()` | Tied-embedding echo prevention skips the top logit entirely — can cause bad outputs | Use proper repetition penalty instead |
| `CMakeLists.txt` | `if(APPLE) target_link_libraries(... ggml-metal)` is a platform guard | Replace with `-DGGML_METAL=ON` cmake option |

---

## How to Look Up Tensors

All tensors are in `tensor_registry` (an `unordered_map<string, ggml_tensor*>`).

For layer N, tensor names follow this pattern (Qwen2/LLaMA):
```
blk.0.attn_norm.weight
blk.0.attn_q.weight
blk.0.attn_k.weight
blk.0.attn_v.weight
blk.0.attn_output.weight
blk.0.ffn_norm.weight
blk.0.ffn_gate.weight
blk.0.ffn_up.weight
blk.0.ffn_down.weight
...
blk.31.ffn_down.weight   (for a 32-layer model)
token_embd.weight
output_norm.weight
output.weight
```

Helper to look up safely:
```cpp
ggml_tensor* get_tensor(const string& name) {
    auto it = tensor_registry.find(name);
    if (it == tensor_registry.end()) return nullptr;
    return it->second;
}
```

---

## What NOT to Do

- Do not use `llama.cpp` patterns that assume single-batch inference with a pre-built graph — FloatLLM builds its graph fresh each forward pass (by design, for chunk streaming)
- Do not add platform-specific `#ifdef` blocks for backends in CMakeLists — use ggml's own detection variables
- Do not `memcpy` tensor data — always use the mmap pointer directly
- Do not add Python bindings yet — get the C++ core working first
- Do not change the `namespace floatllm` scoping

---

## Quick Reference: ggml 2025 API Cheatsheet

```cpp
// Create tensor (metadata only, no allocation)
ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows, cols);

// RMSNorm
ggml_tensor* normed = ggml_rms_norm(ctx, x, 1e-5f);
ggml_tensor* scaled = ggml_mul(ctx, normed, weight);

// Matrix multiply: A @ B  (note: ggml uses column-major, so it's ggml_mul_mat(B, A) for row-major math)
ggml_tensor* out = ggml_mul_mat(ctx, weight, input);

// RoPE (use ext version)
ggml_tensor* q_roped = ggml_rope_ext(ctx, q, positions, nullptr,
    n_dims, rope_mode, 0, freq_base, freq_scale, 1.0f, 1.0f, 1.0f, 1.0f);

// Softmax with scale
ggml_tensor* attn = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);

// SiLU activation
ggml_tensor* activated = ggml_silu(ctx, gate);

// Element-wise multiply
ggml_tensor* gated = ggml_mul(ctx, activated, up);

// Upload data to backend
ggml_backend_tensor_set(tensor, data_ptr, offset_bytes, size_bytes);

// Download data from backend
ggml_backend_tensor_get(tensor, out_ptr, offset_bytes, size_bytes);

// Build and run graph
ggml_build_forward_expand(gf, output_tensor);
ggml_gallocr_alloc_graph(allocr, gf);
ggml_backend_graph_compute(backend, gf);
```