#include "FloatLLM.h"

namespace floatllm {

ComputeEngine::~ComputeEngine() {
    shutdown();
}

string ComputeEngine::resolve_backend(const string& input_name) {
    string lower_name = input_name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

    if (lower_name == "cuda") return "CUDA";
    if (lower_name == "metal" || lower_name == "mps") return "Metal";
    if (lower_name == "vulkan") return "Vulkan";
    if (lower_name == "vulkan_kompute") return "Vulkan";
    if (lower_name == "opencl") return "OpenCL";
    if (lower_name == "npu_ascend") return "OpenCL";
    if (lower_name == "rocm" || lower_name == "hip") return "CUDA"; // ggml maps HIP to CUDA interface internally
    if (lower_name == "oneapi" || lower_name == "sycl") return "SYCL";
    if (lower_name == "xpu") return "SYCL";
    if (lower_name == "directx" || lower_name == "kompute") return "Kompute";

    return input_name; // fallback to CPU
}

//  engine init & shutdown 
void ComputeEngine::init(const string& backend_name, int total_tensors) {
    string raw_hw = backend_name;
    string target_hw = resolve_backend(raw_hw);

    cout << "Requested backend: [" << target_hw << "]" << endl;

    // Scan all the compiled available drivers
    ggml_backend_load_all();
    
    // assign the physical hardware
    if (target_hw == "Metal") {
        #ifdef __APPLE__
        backend = ggml_backend_metal_init();
        #else
        cout << YELLOW("Warning: Metal requested on non-Apple hardware.") << endl;
        #endif
    }
    else if (raw_hw == "cpu" || raw_hw == "native_arm") {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    }
    else if (raw_hw == "best" || raw_hw == "auto") {
        backend = ggml_backend_init_best(); // for automatic best hardware selection
    }
    else {
        // handle Metal, Vulkan, CUDA, etc.
        backend = ggml_backend_init_by_name(target_hw.c_str(), NULL);
    }

    // if the requested GPU isn't available/installed fall back to CPU
    if (backend == nullptr) {
        cout << PURPLE("Required hardware " << target_hw << " unavailable. Falling back to CPU.") << endl;
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    }

    // calculate allocation memory size
    size_t tensor_overhead = ggml_tensor_overhead();
    size_t slack_buffer = 64 * 1024 * 1024; // TODO: Manage this dynamically don't rely on this absolute value! Give user their freedom to choose!
        size_t dynamic_mem_size = (total_tensors * tensor_overhead) + slack_buffer;

    cout << PURPLE("Allocating: " << (dynamic_mem_size / 1024.0 / 1024.0) << "MB for tensors") << endl;

    // Initialize GGML & allocate RAM for compute graph
    struct ggml_init_params params = {
        /* .mem_size    = */ dynamic_mem_size,
        /* .mem_buffer  = */ NULL,
        /* .no_alloc    = */ true, // <-- ZERO-COPY
    };

    ctx = ggml_init(params);

    allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    if (ctx == NULL || backend == NULL || allocr == NULL) {
        cerr << RED("Engine initialization failed!") << endl;
    }
    else {
        cout << GREEN("Engine mapped & Harware locked to: ") << ggml_backend_name(backend) << endl;
    }
}

void ComputeEngine::shutdown() {
    if (!ctx && !backend && !allocr) return;

    cout << " Releasing hardware locks..." << endl;
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
    cout << " Engine shut down. VRAM/RAM cleared safely." << endl;
}

//  tensor operations 

void ComputeEngine::map_tensor(const char* tensor_name, int tensor_type, void* raw_memory_pointer,
                               int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, int chunk_id) {

    //  Model architecture
    struct ggml_tensor* tensor = ggml_new_tensor_4d(ctx, (enum ggml_type)tensor_type, ne0, ne1, ne2, ne3);
    ggml_set_name(tensor, tensor_name);

    tensor->data = raw_memory_pointer; // bind physical ram address

    string name_str(tensor_name);
    tensor_registry[name_str] = tensor;

    cout << "Mapped " << name_str
              << "| Shape: [" << ne0 << ", " << ne1 << ", " << ne2 << ", " << ne3 << "]"
              << "| Target hardware: " << ggml_backend_name(backend) << endl;
}

int32_t ComputeEngine::forward_pass(int32_t* tokens, int num_tokens) {

    struct ggml_init_params params = {1024 * 1024 * 16, NULL, true}; // TODO need to manage dynamically instead of fix 16Mb allocation
    struct ggml_context* graph_ctx = ggml_init(params);
    struct ggml_cgraph* gf = ggml_new_graph(graph_ctx);

    struct ggml_tensor* prompt_tensor = ggml_new_tensor_1d(graph_ctx, GGML_TYPE_I32, num_tokens);
    ggml_set_name(prompt_tensor, "prompt_input");

    struct ggml_tensor* token_embd = nullptr;
    struct ggml_tensor* output_weight = nullptr;

    // check if the core tensors are present in the registry
    if (tensor_registry.count("token_embd.weight")) {
        token_embd = tensor_registry["token_embd.weight"];
    }
    else if (tensor_registry.count("model.embed_tokens.weight")){
        token_embd = tensor_registry["model.embed_tokens.weight"];
    }

    //check if the output tensor is present in the registry
    if (tensor_registry.count("output.weight")) {
        output_weight = tensor_registry["output.weight"];
    }
    else if (tensor_registry.count("lm_head.weight")) {
        output_weight = tensor_registry["lm_head.weight"];
    }

    // if no core tensors found then fallback to scan the registry for the first occurance of the core tensors
    if (!token_embd || !output_weight) {
        for (auto const& [key, val] : tensor_registry) {
            // Skip attention and feed-forward intermediate layers entirely
            if (key.find("blk.") != string::npos || key.find("layer") != string::npos) continue;

            if (!token_embd && key.find("embed") != string::npos) {
                token_embd = val;
            }
            if (!output_weight && (key.find("output") != string::npos || key.find("head") != string::npos) && key.find("norm") == string::npos) {
                output_weight = val;
            }
        }
    }

    // fallback for tied embd (Only if the model is explicitly flagged as tied)
    if (!output_weight && token_embd && tied_embeddings) {
        output_weight = token_embd;
    }

    if (!token_embd || !output_weight) {
        cerr << RED("ERROR: Core tensors (token_embd or output) not found in model.") << endl;
        ggml_free(graph_ctx);
        return -1;
    }

        int64_t max_vocab_size = token_embd->ne[1];
    for (int i = 0; i < num_tokens; i++) {
        if (tokens[i] < 0 || tokens[i] >= max_vocab_size) {
            tokens[i] = 0; // Clamp to safe default to keep the engine alive
        }
    }

    // handle the math and logits
    struct ggml_tensor* current_embeddings = ggml_get_rows(graph_ctx, token_embd, prompt_tensor);

    struct ggml_tensor* logits = ggml_mul_mat(graph_ctx, output_weight, current_embeddings);
    ggml_build_forward_expand(gf, logits);

    // Save the RAM pointers locally
    void* raw_embd_ptr = token_embd->data;
    void* raw_out_ptr = output_weight->data;

    // detach pointers so the GPU allocates true VRAM for them
    token_embd->data = nullptr;
    output_weight->data = nullptr;

    // Clear the buffers from the previous loop iteration
    token_embd->buffer = nullptr;
    output_weight->buffer = nullptr;

    ggml_gallocr_alloc_graph(allocr, gf);

    // upload the zero-copy data into the Hardware VRAM buffer
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

    // Extract the last token we just fed into the network
    int32_t last_input_token = tokens[num_tokens - 1];

    for (int i = 0; i < vocab_size; i++) {
        // Prevent tied-embedding models from continuously echoing the exact same token
        if (i == last_input_token) continue;

        if (logits_data[i] > max_val) {
            max_val = logits_data[i];
            best_token = i;
        }
    }

    // Restore the RAM pointers for nexxt generation loop
    token_embd->data = raw_embd_ptr;
    output_weight->data = raw_out_ptr;

    ggml_free(graph_ctx);
    return best_token;
}

string ComputeEngine::detect_hardware() {
#if defined(__APPLE__)
    return "mps";
#elif defined(__ANDROID__)
    if (access("/system/lib64/libvulkan.so", F_OK) == 0 || access("/system/lib/libvulkan.so", F_OK) == 0) {
        return "vulkan_kompute";
    }
    return "native_arm";
#elif defined(__linux__)
    if (access("/usr/bin/vulkaninfo", X_OK) == 0 || access("/usr/local/bin/vulkaninfo", X_OK) == 0 ||
        access("/system/lib64/libvulkan.so", F_OK) == 0 || access("/system/lib/libvulkan.so", F_OK) == 0) {
        return "vulkan_kompute";
    }
#if defined(__aarch64__) || defined(__arm__)
    return "native_arm";
#endif
    return "cpu";
#else
    return "cpu";
#endif
}

} // namespace floatllm
