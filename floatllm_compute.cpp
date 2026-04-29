#include <iostream>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

extern "C" {
    // pointer to hold math engine's state & memory map
    struct ggml_context* ctx;

    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocr = nullptr;

    // Tensor dictionary to hold model config
    std::unordered_map<std::string, struct ggml_tensor*> tensor_registry;

    // 1. Initialization socket
    void init_compute_engine(const char* backend_name, int total_tensors) {
        std::string target_hw(backend_name);
        std::cout << "[FloatLLM(C++)] Hardware Router active. Requested: [" << target_hw << "]" <<std::endl;

        // Scan all the available drivers
        ggml_backend_load_all();
        // dynamically assign the physical hardware
        if (target_hw == "cpu" || target_hw == "CPU") {
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
        }
        else {
            backend = ggml_backend_init_best();
        }

        // If the requested GPU isn't available/installed
        if (backend == nullptr) {
            std::cout << "[FloatLLM(C++)] Target GPU unavailable. Falling back to CPU." << std::endl;
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

    // 2. The execution socket with 4D shape
    void execute_tensor_chunk(const char* tensor_name, int tensor_type, void* raw_memory_pointer, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, int chunk_id) {

        
        // 4D model architecture
        struct ggml_tensor* tensor = ggml_new_tensor_4d(ctx, (enum ggml_type)tensor_type, ne0, ne1, ne2,ne3);
        ggml_set_name(tensor, tensor_name); 

        tensor->data = raw_memory_pointer; // bind physical ram address

        std::string name_str(tensor_name);
        tensor_registry[name_str] = tensor;

        std::cout << "[FloatLLM C++] Mapped " << name_str
                  << "| Shape: [" << ne0 << ", " << ne1 << ", " << ne2 << ", " <<ne3 << "]"
                  << "| Target hardware: " << ggml_backend_name(backend) << std::endl;
        }

    // 3. Math execution pipeline
    void execute_graph_test() {
        std::cout << "[FloatLLM(C++)] Assembling computation graph..." << std::endl;

        // tiny localized context for graph's instructions (1MB)
        struct ggml_init_params params = {
            /* .mem_size   = */ 1024 * 1024,
            /* .mem_buffer = */ NULL,
            /* .no_alloc.  = */ true,
        };
        struct ggml_context * graph_ctx = ggml_init(params);

        struct ggml_cgraph * gf = ggml_new_graph(graph_ctx);

        // --- Math Pipeline ---
        // Tell the Allocator to reserve VRAM on the GPU/CPU for the math operations
        ggml_gallocr_alloc_graph(allocr, gf);

        std::cout << "[FloatLLM(C++)] Graph memory reserved. Pushing data to compute cores..." << std::endl;
        
        ggml_backend_graph_compute(backend, gf);

        std::cout << "[Float(C++)] Matrix math complete. Target hardware stabilized." << std::endl;

        ggml_free(graph_ctx);
    }
}