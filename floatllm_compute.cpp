#include <iostream>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

// Bug fix
#ifdef __APPLE__
extern "C" ggml_backend_t ggml_backend_metal_init(void);
#endif

// Map user flags to exact GGML backend name
std::string resolve_backend_name(const std::string& input_name) {
    std::string lower_name = input_name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

    if (lower_name == "cuda") return "CUDA";
    if (lower_name == "metal" || lower_name == "mps") return "Metal";
    if (lower_name == "vulkan") return "Vulkan";
    if (lower_name == "opencl") return "OpenCL";
    if (lower_name == "rocm" || lower_name == "hip") return "CUDA"; // ggml maps HIP to CUDA interface internally
    if (lower_name == "oneapi" || lower_name == "sycl") return "SYCL";
    if (lower_name == "directx" || lower_name == "kompute") return "Kompute";

    return input_name; // fallback
}

extern "C" {
    // pointer to hold math engine's state & memory map
    struct ggml_context* ctx;

    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocr = nullptr;

    // Tensor dictionary to hold model config
    std::unordered_map<std::string, struct ggml_tensor*> tensor_registry;

    // 1. Initialization socket
    void init_compute_engine(const char* backend_name, int total_tensors) {
        std::string raw_hw(backend_name);
        std::string target_hw = resolve_backend_name(raw_hw);

        std::cout << "[FloatLLM(C++)] Hardware Router active. Requested: [" << target_hw << "]" <<std::endl;

        // Scan all the compiled available drivers
        ggml_backend_load_all();
        
        // dynamically assign the physical hardware
        if (target_hw == "Metal") {
            #ifdef __APPLE__
            backend = ggml_backend_metal_init();
            #else
            std::cout << "[FloatLLM(C++)] Warning: Metal requested on non-Apple hardware." << std::endl;
            #endif
        }
        else if (raw_hw == "cpu" || raw_hw == "native_arm") {
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
        }
        else if (raw_hw == "best" || raw_hw == "auto") {
            backend = ggml_backend_init_best(); // for auto default behaviour
        }
        else {
            // Automatically handle Metal, Vulkan, CUDA, etc.
            backend = ggml_backend_init_by_name(target_hw.c_str(), NULL);
        }

        // If the requested GPU isn't available/installed
        if (backend == nullptr) {
            std::cout << "[FloatLLM(C++)] Target hardware" << target_hw << " unavailable. Falling back to CPU." << std::endl;
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
        }

        // --- Dynamically calculate allocation memory size
        size_t tensor_overhead = ggml_tensor_overhead();
        size_t slack_buffer = 4 * 1024 * 1024; // 4MB safety buffer for extreme large models
        size_t dynamic_mem_size = (total_tensors * tensor_overhead) + slack_buffer;

        std::cout << "[FloatLLM(C++)] Allocating dynamic context of: " << (dynamic_mem_size / 1024.0 / 1024.0) << "MB" << std::endl;

        // Initialize GGML & allocate RAM for "Compute Graph"
        struct ggml_init_params params = {
            /* .mem_size    = */ dynamic_mem_size,
            /* .mem_buffer  = */ NULL,
            /* .no_alloc    = */ true, // <-- ZERO-COPY
        };

        ctx = ggml_init(params);

        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

        if (ctx == NULL || backend == NULL || allocr == NULL) {
            std::cerr << "[FloatLLM(C++)] Engine initialization failed!" << std::endl;
        }
        else {
            std::cout << "[FloatLLM(C++)] Engine mapped & Harware locked to: " << ggml_backend_name(backend) << std::endl;
            std::cout << "[FloatLLM(C++)] Graph Allocator online." << std::endl;
        }
    }

    // 2. The execution socket with dynamic shapes
    void execute_tensor_chunk(const char* tensor_name, int tensor_type, void* raw_memory_pointer, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, int chunk_id) {

        //  Model architecture
        struct ggml_tensor* tensor = ggml_new_tensor_4d(ctx, (enum ggml_type)tensor_type, ne0, ne1, ne2,ne3);
        ggml_set_name(tensor, tensor_name); 

        tensor->data = raw_memory_pointer; // bind physical ram address

        std::string name_str(tensor_name);
        tensor_registry[name_str] = tensor;

        std::cout << "[FloatLLM C++] Mapped " << name_str
                  << "| Shape: [" << ne0 << ", " << ne1 << ", " << ne2 << ", " <<ne3 << "]"
                  << "| Target hardware: " << ggml_backend_name(backend) << std::endl;
        }

    // Return 32-bit integer (the next token)
    int32_t execute_forward_pass(int32_t* tokens, int num_tokens) {
        
        struct ggml_init_params params = {1024 * 1024 * 16, NULL, true};
        struct ggml_context * graph_ctx = ggml_init(params);
        struct ggml_cgraph * gf = ggml_new_graph(graph_ctx);

        struct ggml_tensor* prompt_tensor = ggml_new_tensor_1d(graph_ctx, GGML_TYPE_I32, num_tokens);
        ggml_set_name(prompt_tensor, "prompt_input");

        struct ggml_tensor* token_embd = tensor_registry["token_embd.weight"];
        struct ggml_tensor* output_weight = tensor_registry["output.weight"];

        if (!token_embd || !output_weight){
            std::cerr << "[FloatLLM(C++)] ERROR: Core tensors (token_embd or output) not found in model." << std::endl;
            ggml_free(graph_ctx);
            return 128009; 
        }

        // --- MATH PIPELINE ---
        struct ggml_tensor* current_embeddings = ggml_get_rows(graph_ctx, token_embd, prompt_tensor);
        
        
        struct ggml_tensor* logits = ggml_mul_mat(graph_ctx, output_weight, current_embeddings);
        ggml_build_forward_expand(gf, logits);

        // Save the Python RAM pointers locally 
        void* raw_embd_ptr = token_embd->data;
        void* raw_out_ptr = output_weight->data;

        // detach pointers so the GPU allocates true VRAM for them
        token_embd->data = nullptr;
        output_weight->data = nullptr;

        // Clear the buffers from the previous loop iteration
        token_embd->buffer = nullptr;
        output_weight->buffer = nullptr;

        ggml_gallocr_alloc_graph(allocr, gf); 
        
        // Securely upload the zero-copy Python data into the Hardware VRAM buffer
        ggml_backend_tensor_set(token_embd, raw_embd_ptr, 0, ggml_nbytes(token_embd));
        ggml_backend_tensor_set(output_weight, raw_out_ptr, 0, ggml_nbytes(output_weight));
        ggml_backend_tensor_set(prompt_tensor, tokens, 0, num_tokens * sizeof(int32_t));

        // Fire the GPU!
        ggml_backend_graph_compute(backend, gf);

        // Read the final results back to RAM to pick the winner
        int vocab_size = logits->ne[0];
        std::vector<float> logits_data(vocab_size);
        
        // Logits for the very LAST token in the sequence 
        ggml_backend_tensor_get(logits, logits_data.data(), (num_tokens - 1) * vocab_size * sizeof(float), vocab_size * sizeof(float));

        // Find the index of the highest probability 
        int32_t best_token = 0;
        float max_val = -1e9;
        for (int i = 0; i < vocab_size; i++) {
            if (logits_data[i] > max_val) {
                max_val = logits_data[i];
                best_token = i;
            }
        }

        // --- RESTORE THE PYTHON RAM POINTERS FOR THE NEXT GENERATION LOOP ---
        token_embd->data = raw_embd_ptr;
        output_weight->data = raw_out_ptr;

        ggml_free(graph_ctx);
        return best_token;
    }

    // 4. Shutdown to prevent Memory Leaks
    void shutdown_compute_engine() {
        std::cout << "[FloatLLM(C++)] Releasing hardware locks..." << std::endl;
        if (allocr) {
            ggml_gallocr_free(allocr);
            allocr = nullptr;
        }
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        if (backend) {
            ggml_backend_free(backend);
            backend = nullptr;
        }
        tensor_registry.clear();
        std::cout << "[FloatLLM(C++)] Engine shut down. VRAM/RAM cleared safely." << std::endl;
    }
}