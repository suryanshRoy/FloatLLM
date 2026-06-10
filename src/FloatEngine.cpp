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
void ComputeEngine::init(const string& backend_name, int total_tensors, double slack_buffer_mb) {
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

    // If the requested GPU isn't available/installed
    if (backend == nullptr) {
        cout << PURPLE("Required hardware" << target_hw << " unavailable. Falling back to CPU.") << endl;
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    }

    backend_is_cpu = ggml_backend_is_cpu(backend);

    // calculate allocation memory size (slack is user-tunable via --slack-buffer-mb)
    size_t tensor_overhead = ggml_tensor_overhead();
    size_t slack_buffer = static_cast<size_t>(slack_buffer_mb * 1024.0 * 1024.0);
    size_t dynamic_mem_size = (total_tensors * tensor_overhead) + slack_buffer;

    cout << PURPLE("Allocating: " << (dynamic_mem_size / 1024.0 / 1024.0) << "MB for tensors") << endl;

    // Initialize GGML & allocate RAM for "Compute Graph"
    // no_alloc keeps weight tensors metadata-only: CPU runs zero-copy straight
    // from the mmap, GPUs get a single one-time upload in finalize_weights()
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

void ComputeEngine::set_hparams(const ModelHParams& hp) {
    hparams = hp;
}

void ComputeEngine::shutdown() {
    if (!ctx && !backend && !allocr) return;

    cout << " Releasing hardware locks..." << endl;
    if (kv_buffer) {
        ggml_backend_buffer_free(kv_buffer);
        kv_buffer = nullptr;
    }
    if (kv_ctx) {
        ggml_free(kv_ctx);
        kv_ctx = nullptr;
    }
    k_cache.clear();
    v_cache.clear();
    if (weight_buffer) {
        ggml_backend_buffer_free(weight_buffer);
        weight_buffer = nullptr;
    }
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
    raw_data_registry.clear();
    weights_finalized = false;
    cout << " Engine shut down. VRAM/RAM cleared safely." << endl;
}

//  tensor operations 

void ComputeEngine::map_tensor(const char* tensor_name, int tensor_type, void* raw_memory_pointer,
                               int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, int chunk_id) {

    //  Model architecture
    struct ggml_tensor* tensor = ggml_new_tensor_4d(ctx, (enum ggml_type)tensor_type, ne0, ne1, ne2, ne3);
    ggml_set_name(tensor, tensor_name);

    string name_str(tensor_name);
    raw_data_registry[name_str] = raw_memory_pointer;

    if (backend_is_cpu) {
        tensor->data = raw_memory_pointer; // zero-copy: compute directly out of the mmap
    }

    tensor_registry[name_str] = tensor;

    cout << "Mapped " << name_str
              << "| Shape: [" << ne0 << ", " << ne1 << ", " << ne2 << ", " << ne3 << "]"
              << "| Target hardware: " << ggml_backend_name(backend) << endl;
}

struct ggml_tensor* ComputeEngine::get_weight(const string& name) const {
    const auto it = tensor_registry.find(name);
    return it == tensor_registry.end() ? nullptr : it->second;
}

// upload the weights to the backend exactly once (instead of re-uploading on
// every single token like before), then carve out the persistent KV cache.
// On CPU the mmap pointers are used directly - true zero-copy.
bool ComputeEngine::finalize_weights() {
    if (weights_finalized) return true;

    if (!backend_is_cpu) {
        // one-time bulk upload: allocate all weight tensors in backend memory
        weight_buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!weight_buffer) {
            // VRAM fallback: GPU could not hold the weights, drop to CPU zero-copy
            cout << YELLOW("Backend memory exhausted while uploading weights. Falling back to CPU zero-copy.") << endl;
            ggml_backend_free(backend);
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
            backend_is_cpu = true;
            if (allocr) ggml_gallocr_free(allocr);
            allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            for (auto& [name, tensor] : tensor_registry) {
                tensor->data = raw_data_registry[name];
            }
        } else {
            for (auto& [name, tensor] : tensor_registry) {
                ggml_backend_tensor_set(tensor, raw_data_registry[name], 0, ggml_nbytes(tensor));
            }
            cout << GREEN("Weights uploaded to backend memory once (no per-token re-upload).") << endl;
        }
    }

    // allocate the persistent KV cache
    if (hparams.valid()) {
        const int64_t n_embd_kv = hparams.n_embd_kv();
        const int64_t n_ctx = hparams.n_ctx;

        struct ggml_init_params kv_params = {
            /* .mem_size    = */ 2 * hparams.n_layer * ggml_tensor_overhead(),
            /* .mem_buffer  = */ NULL,
            /* .no_alloc    = */ true,
        };
        kv_ctx = ggml_init(kv_params);

        k_cache.resize(hparams.n_layer);
        v_cache.resize(hparams.n_layer);
        for (int l = 0; l < hparams.n_layer; ++l) {
            k_cache[l] = ggml_new_tensor_1d(kv_ctx, GGML_TYPE_F16, n_embd_kv * n_ctx);
            v_cache[l] = ggml_new_tensor_1d(kv_ctx, GGML_TYPE_F16, n_embd_kv * n_ctx);
            ggml_format_name(k_cache[l], "cache_k_l%d", l);
            ggml_format_name(v_cache[l], "cache_v_l%d", l);
        }

        kv_buffer = ggml_backend_alloc_ctx_tensors(kv_ctx, backend);
        if (!kv_buffer) {
            cerr << RED("ERROR: failed to allocate the KV cache!") << endl;
            return false;
        }
        ggml_backend_buffer_clear(kv_buffer, 0);

        const double kv_mb = ggml_backend_buffer_get_size(kv_buffer) / (1024.0 * 1024.0);
        cout << "\033[38;2;153;51;255mKV cache online: " << hparams.n_layer << " layers x " << n_ctx
             << " ctx slots (" << kv_mb << " MB)\033[0m" << endl;
    } else {
        cout << YELLOW("No transformer hyper-parameters found. Running in embedding-only fallback mode.") << endl;
    }

    weights_finalized = true;
    return true;
}

// builds the full transformer stack: RMSNorm -> QKV -> RoPE -> KV cache ->
// causal attention -> output proj -> residual -> SwiGLU FFN -> residual
struct ggml_tensor* ComputeEngine::build_transformer_graph(struct ggml_context* gctx, struct ggml_cgraph* gf,
                                                           struct ggml_tensor* inp_tokens, struct ggml_tensor* inp_pos,
                                                           struct ggml_tensor* kq_mask, int num_tokens, int n_past) {
    const int n_kv = n_past + num_tokens;
    const int64_t head_dim = hparams.head_dim();
    const int64_t n_head = hparams.n_head;
    const int64_t n_head_kv = hparams.n_head_kv;
    const int64_t n_embd = hparams.n_embd;
    const int64_t n_embd_kv = hparams.n_embd_kv();
    const int64_t n_ctx = hparams.n_ctx;
    const int n_rot = hparams.n_rot > 0 ? hparams.n_rot : static_cast<int>(head_dim);
    const float kq_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    const size_t kv_esize = ggml_type_size(GGML_TYPE_F16);

    struct ggml_tensor* token_embd = get_weight("token_embd.weight");
    struct ggml_tensor* output_norm = get_weight("output_norm.weight");
    struct ggml_tensor* output_w = get_weight("output.weight");
    if (!output_w) output_w = token_embd; // tied embeddings

    if (!token_embd || !output_norm) return nullptr;

    struct ggml_tensor* cur = ggml_get_rows(gctx, token_embd, inp_tokens);

    for (int l = 0; l < hparams.n_layer; ++l) {
        const string p = "blk." + std::to_string(l) + ".";
        struct ggml_tensor* attn_norm_w = get_weight(p + "attn_norm.weight");
        struct ggml_tensor* wq = get_weight(p + "attn_q.weight");
        struct ggml_tensor* wk = get_weight(p + "attn_k.weight");
        struct ggml_tensor* wv = get_weight(p + "attn_v.weight");
        struct ggml_tensor* wo = get_weight(p + "attn_output.weight");
        struct ggml_tensor* ffn_norm_w = get_weight(p + "ffn_norm.weight");
        struct ggml_tensor* w_gate = get_weight(p + "ffn_gate.weight");
        struct ggml_tensor* w_up = get_weight(p + "ffn_up.weight");
        struct ggml_tensor* w_down = get_weight(p + "ffn_down.weight");

        if (!attn_norm_w || !wq || !wk || !wv || !wo || !ffn_norm_w || !w_gate || !w_up || !w_down) {
            cerr << RED("ERROR: missing weights for transformer layer ") << l << endl;
            return nullptr;
        }

        // optional QKV biases (Qwen2 family uses them, Llama does not)
        struct ggml_tensor* bq = get_weight(p + "attn_q.bias");
        struct ggml_tensor* bk = get_weight(p + "attn_k.bias");
        struct ggml_tensor* bv = get_weight(p + "attn_v.bias");

        struct ggml_tensor* residual = cur;

        // pre-attention RMSNorm
        cur = ggml_rms_norm(gctx, cur, hparams.f_norm_rms_eps);
        cur = ggml_mul(gctx, cur, attn_norm_w);

        // QKV projections
        struct ggml_tensor* Qcur = ggml_mul_mat(gctx, wq, cur);
        struct ggml_tensor* Kcur = ggml_mul_mat(gctx, wk, cur);
        struct ggml_tensor* Vcur = ggml_mul_mat(gctx, wv, cur);
        if (bq) Qcur = ggml_add(gctx, Qcur, bq);
        if (bk) Kcur = ggml_add(gctx, Kcur, bk);
        if (bv) Vcur = ggml_add(gctx, Vcur, bv);

        // split into heads + RoPE (NEOX style rotary position embedding)
        Qcur = ggml_reshape_3d(gctx, Qcur, head_dim, n_head, num_tokens);
        Kcur = ggml_reshape_3d(gctx, Kcur, head_dim, n_head_kv, num_tokens);

        Qcur = ggml_rope_ext(gctx, Qcur, inp_pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX,
                             hparams.n_ctx_train, hparams.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        Kcur = ggml_rope_ext(gctx, Kcur, inp_pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX,
                             hparams.n_ctx_train, hparams.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // append the new K rows into the persistent cache at position n_past
        struct ggml_tensor* k_view = ggml_view_1d(gctx, k_cache[l], num_tokens * n_embd_kv,
                                                  n_past * n_embd_kv * kv_esize);
        ggml_build_forward_expand(gf, ggml_cpy(gctx, Kcur, k_view));

        // V is stored transposed so attention can read it with friendly strides
        struct ggml_tensor* v_t = ggml_transpose(gctx, ggml_reshape_2d(gctx, Vcur, n_embd_kv, num_tokens));
        struct ggml_tensor* v_view = ggml_view_2d(gctx, v_cache[l], num_tokens, n_embd_kv,
                                                  n_ctx * kv_esize, n_past * kv_esize);
        ggml_build_forward_expand(gf, ggml_cpy(gctx, v_t, v_view));

        // attention over everything in the cache (past + current tokens)
        struct ggml_tensor* K = ggml_view_3d(gctx, k_cache[l], head_dim, n_kv, n_head_kv,
                                             n_embd_kv * kv_esize, head_dim * kv_esize, 0);
        struct ggml_tensor* V = ggml_view_3d(gctx, v_cache[l], n_kv, head_dim, n_head_kv,
                                             n_ctx * kv_esize, n_ctx * head_dim * kv_esize, 0);

        struct ggml_tensor* Q = ggml_permute(gctx, Qcur, 0, 2, 1, 3); // [head_dim, num_tokens, n_head]
        struct ggml_tensor* KQ = ggml_mul_mat(gctx, K, Q);            // GQA broadcast over kv heads
        KQ = ggml_soft_max_ext(gctx, KQ, kq_mask, kq_scale, 0.0f);    // scaled + causal-masked softmax

        struct ggml_tensor* KQV = ggml_mul_mat(gctx, V, KQ);                  // [head_dim, num_tokens, n_head]
        KQV = ggml_permute(gctx, KQV, 0, 2, 1, 3);                            // [head_dim, n_head, num_tokens]
        cur = ggml_cont_2d(gctx, KQV, n_embd, num_tokens);                    // merge heads

        cur = ggml_mul_mat(gctx, wo, cur);
        cur = ggml_add(gctx, cur, residual);

        // feed-forward block (SwiGLU)
        residual = cur;
        cur = ggml_rms_norm(gctx, cur, hparams.f_norm_rms_eps);
        cur = ggml_mul(gctx, cur, ffn_norm_w);

        struct ggml_tensor* gate = ggml_silu(gctx, ggml_mul_mat(gctx, w_gate, cur));
        struct ggml_tensor* up = ggml_mul_mat(gctx, w_up, cur);
        cur = ggml_mul_mat(gctx, w_down, ggml_mul(gctx, gate, up));
        cur = ggml_add(gctx, cur, residual);
    }

    // final norm + LM head
    cur = ggml_rms_norm(gctx, cur, hparams.f_norm_rms_eps);
    cur = ggml_mul(gctx, cur, output_norm);
    return ggml_mul_mat(gctx, output_w, cur);
}

// the actual inference - returns next token id
int32_t ComputeEngine::forward_pass(const int32_t* tokens, int num_tokens, int n_past) {
    if (!weights_finalized && !finalize_weights()) {
        return -1;
    }

    const bool full_model = hparams.valid() && !k_cache.empty();
    const int n_kv = n_past + num_tokens;

    if (full_model && n_kv > hparams.n_ctx) {
        cerr << RED("ERROR: context window exhausted: ") << n_kv << " > " << hparams.n_ctx << endl;
        return -1;
    }

    const size_t graph_mem = 2 * ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    struct ggml_init_params params = {graph_mem, NULL, true};
    struct ggml_context* graph_ctx = ggml_init(params);
    struct ggml_cgraph* gf = ggml_new_graph(graph_ctx);

    struct ggml_tensor* prompt_tensor = ggml_new_tensor_1d(graph_ctx, GGML_TYPE_I32, num_tokens);
    ggml_set_name(prompt_tensor, "prompt_input");

    struct ggml_tensor* token_embd = get_weight("token_embd.weight");
    if (!token_embd) token_embd = get_weight("model.embed_tokens.weight");
    if (!token_embd) {
        cerr << RED("ERROR: token embedding tensor not found in model.") << endl;
        ggml_free(graph_ctx);
        return -1;
    }

    const int64_t max_vocab_size = token_embd->ne[1];
    std::vector<int32_t> safe_tokens(tokens, tokens + num_tokens);
    for (auto& t : safe_tokens) {
        if (t < 0 || t >= max_vocab_size) t = 0; // clamp to keep the engine alive
    }

    struct ggml_tensor* logits = nullptr;
    struct ggml_tensor* inp_pos = nullptr;
    struct ggml_tensor* kq_mask = nullptr;

    if (full_model) {
        inp_pos = ggml_new_tensor_1d(graph_ctx, GGML_TYPE_I32, num_tokens);
        ggml_set_name(inp_pos, "inp_pos");
        kq_mask = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, n_kv, num_tokens);
        ggml_set_name(kq_mask, "kq_mask");

        logits = build_transformer_graph(graph_ctx, gf, prompt_tensor, inp_pos, kq_mask, num_tokens, n_past);
    }

    if (!logits) {
        // fallback: embedding -> output head only (model missing transformer weights)
        struct ggml_tensor* output_w = get_weight("output.weight");
        if (!output_w) output_w = get_weight("lm_head.weight");
        if (!output_w) output_w = token_embd;
        struct ggml_tensor* embeddings = ggml_get_rows(graph_ctx, token_embd, prompt_tensor);
        logits = ggml_mul_mat(graph_ctx, output_w, embeddings);
        inp_pos = nullptr;
        kq_mask = nullptr;
    }

    ggml_build_forward_expand(gf, logits);

    ggml_gallocr_alloc_graph(allocr, gf);

    // upload only the tiny per-step inputs (tokens, positions, mask)
    ggml_backend_tensor_set(prompt_tensor, safe_tokens.data(), 0, num_tokens * sizeof(int32_t));

    if (inp_pos) {
        std::vector<int32_t> positions(num_tokens);
        for (int i = 0; i < num_tokens; ++i) positions[i] = n_past + i;
        ggml_backend_tensor_set(inp_pos, positions.data(), 0, num_tokens * sizeof(int32_t));
    }
    if (kq_mask) {
        // causal mask: token i may only attend to cache positions <= n_past + i
        std::vector<float> mask(static_cast<size_t>(n_kv) * num_tokens, 0.0f);
        for (int i = 0; i < num_tokens; ++i) {
            for (int j = n_past + i + 1; j < n_kv; ++j) {
                mask[static_cast<size_t>(i) * n_kv + j] = -INFINITY;
            }
        }
        ggml_backend_tensor_set(kq_mask, mask.data(), 0, mask.size() * sizeof(float));
    }

    // Fire the GPU!
    ggml_backend_graph_compute(backend, gf);

    // Read the final results back to RAM to pick the winner
    const int vocab_size = logits->ne[0];
    std::vector<float> logits_data(vocab_size);

    // Logits for the very LAST token in the sequence
    ggml_backend_tensor_get(logits, logits_data.data(), (num_tokens - 1) * static_cast<size_t>(vocab_size) * sizeof(float), vocab_size * sizeof(float));

    // greedy sampling: index of the highest logit
    int32_t best_token = 0;
    float max_val = -1e9;
    for (int i = 0; i < vocab_size; i++) {
        if (logits_data[i] > max_val) {
            max_val = logits_data[i];
            best_token = i;
        }
    }

    ggml_free(graph_ctx);
    return best_token;
}

//  static: pre-flight safety checks 

double ComputeEngine::check_threshold(double current_ram_mb, double crash_threshold_mb, double model_size_mb,
                                      double total_storage_gb, double free_storage_gb, double used_ram_mb,
                                      double total_ram_mb, double override_storage_gb, const char* session_id,
                                      int temp_chat, double ram_limit_gb, double ram_buffer) {
    
    // ATTENTION!
    auto fmt2 = [](double value) { // format double to string with 2 decimal places
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << value;
        return oss.str();
    };

    const string session = session_id ? session_id : "default";
    const double model_size_gb = model_size_mb / 1024.0;
    double trusted_free_gb = free_storage_gb;

    if (override_storage_gb >= 0.0) {
        trusted_free_gb = override_storage_gb;
        cout << "\n Overriding UNIX limits. Trusting the input value of "
                  << fmt2(trusted_free_gb) << " GB.\n";
        if (total_storage_gb > 0.0 && trusted_free_gb > total_storage_gb) {
            cerr << RED("ERROR: Override (" << trusted_free_gb
                      << " GB) exceeds total disk size (" << fmt2(total_storage_gb)
                      << " GB). Aborting the process.\n");
            std::exit(1);
        }
    }
    else {
#ifdef __APPLE__
        if (free_storage_gb > 0.0 && model_size_gb > free_storage_gb) {
            cout << YELLOW("\n⚠️ UNIX sees " << fmt2(free_storage_gb)
                      << " GB. Model needs " << fmt2(model_size_gb) << " GB.\n");
            cout << YELLOW("macOS hides Purgeable space. If you have enough space in System Settings, run with: [--override-storage YOUR_GB]\n");
        }
#endif
    }

    if (trusted_free_gb > 0.0 && model_size_gb > trusted_free_gb) {
        cerr << RED("\n--------------------------------------------------------------------------------\n");
        cerr << RED("ERROR: Model requires " << fmt2(model_size_gb)
                  << "GB, but only " << fmt2(trusted_free_gb) << " GB is free.\n");
        cerr << RED("Halting to prevent storage corruption.\n");
        cerr << RED("--------------------------------------------------------------------------------\n\n");
        std::exit(1);
    }

    if (current_ram_mb <= crash_threshold_mb) {
        cerr << RED("\n--------------------------------------------------------------------------------\n");
        cerr << RED("OOM Failsafe triggered to stop crashing/freezing of device.\n");
        cerr << RED("ERROR: Free RAM (" << fmt2(current_ram_mb)
                  << "MB) hit the crash threshold (" << fmt2(crash_threshold_mb) << " MB).\n");
        cerr << RED("Target Model Size: " << fmt2(model_size_mb) << " MB\n");
        if (used_ram_mb >= 0.0) {
            cerr << PURPLE("FloatLLM Consumed: " << fmt2(used_ram_mb) << " MB (Max Peak)\n");
        }
        cerr << PURPLE("Halting execution gracefully.\n");
        cerr << PURPLE("Adjust [--crash-threshold] or increase [--ram-limit] for more headroom.\n");
        cerr << PURPLE("--------------------------------------------------------------------------------\n\n");
        std::exit(1);
    }

    const double safe_ram_mb = std::max(1.0, (current_ram_mb * (1.0 - ram_buffer)) - crash_threshold_mb);
    const double allowed_ram_mb = (ram_limit_gb > 0.0)
        ? std::min(safe_ram_mb, ram_limit_gb * 1024.0)
        : safe_ram_mb;

    // print the dashboard so user knows whats going on
    cout << PURPLE("\n--- Pre-Flight Memory Dashboard ---\n");
    if (total_ram_mb > 0.0) {
        cout << PURPLE("Total Ram       : " << fmt2(total_ram_mb) << "MB\n");
        cout << PURPLE("Used RAM        : " << fmt2(total_ram_mb - current_ram_mb) << "MB\n");
    }
    cout << PURPLE("Free Ram        : " << fmt2(current_ram_mb) << "MB\n");
    cout << PURPLE("Allowed RAM (Chunk)  : " << fmt2(allowed_ram_mb) << "MB (Buffer: " << (ram_buffer * 100.0) << "%)\n");
    if (trusted_free_gb > 0.0) {
        cout << PURPLE("Free Storage    : " << fmt2(trusted_free_gb) << "GB " << (override_storage_gb >= 0.0 ? "(OVERRIDEN)" : "") << "\n");
    }
    cout << PURPLE("Target Model Size    : " << fmt2(model_size_mb) << "MB\n");
    cout << PURPLE("Kill threshold       : " << fmt2(crash_threshold_mb) << "MB\n");
    cout << PURPLE("--- Session Info \n");
    cout << PURPLE("Session ID           : [" << session << "]\n");
    cout << PURPLE("Context Saving       : " << (temp_chat ? "Temporary (Delete on Exit)" : "PERSISTENT (Saved to SSD)") << "\n\n");

    return allowed_ram_mb;
}

//  static: system info utilities 

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

std::pair<double, double> ComputeEngine::get_ram_stats_mb() {
#ifdef _WIN32
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    if (GlobalMemoryStatusEx(&statex)) {
        double total_mb = static_cast<double>(statex.ullTotalPhys) / (1024.0 * 1024.0);
        double free_mb = static_cast<double>(statex.ullAvailPhys) / (1024.0 * 1024.0);
        return {total_mb, free_mb};
    }
    return {0.0, 0.0};
#elif defined(__APPLE__)
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm_stat), &count) == KERN_SUCCESS) {
        const double page_size = static_cast<double>(sysconf(_SC_PAGESIZE));
        const double total = static_cast<double>(sysconf(_SC_PHYS_PAGES)) * page_size / (1024.0 * 1024.0);
        const double free = static_cast<double>(vm_stat.free_count + vm_stat.inactive_count) * page_size / (1024.0 * 1024.0);
        return {total, free};
    }
#endif
    // fallback for other linux distros
    const long page_size = sysconf(_SC_PAGESIZE);
    const long phys_pages = sysconf(_SC_PHYS_PAGES);
    const double total = (page_size > 0 && phys_pages > 0)
        ? (static_cast<double>(page_size) * static_cast<double>(phys_pages)) / (1024.0 * 1024.0)
        : 0.0;
    const double free = total * 0.5; // rough estimate when we cant get exact numbers
    return {total, free};
}

std::pair<double, double> ComputeEngine::get_storage_stats_gb() {
#ifdef _WIN32
    ULARGE_INTEGER freeBytesAvailable, totalNumberOfBytes, totalNumberOfFreeBytes;
    if (GetDiskFreeSpaceExA(".", &freeBytesAvailable, &totalNumberOfBytes, &totalNumberOfFreeBytes)) {
        double total_gb = static_cast<double>(totalNumberOfBytes.QuadPart) / (1024.0 * 1024.0 * 1024.0);
        double free_gb = static_cast<double>(freeBytesAvailable.QuadPart) / (1024.0 * 1024.0 * 1024.0);
        return {total_gb, free_gb};
    }
    return {0.0, 0.0};
#else
    const char* home = std::getenv("HOME");
    const char* root = home ? home : ".";
    struct statvfs fs {};
    if (statvfs(root, &fs) != 0) {
        return {0.0, 0.0};
    }

    const double total_bytes = static_cast<double>(fs.f_blocks) * static_cast<double>(fs.f_frsize);
    const double free_bytes = static_cast<double>(fs.f_bavail) * static_cast<double>(fs.f_frsize);
    return {
        total_bytes / (1024.0 * 1024.0 * 1024.0),
        free_bytes / (1024.0 * 1024.0 * 1024.0)
    };
#endif
}

size_t ComputeEngine::file_size_bytes(const string& path) {
    struct stat st {};
    if (stat(path.c_str(), &st) != 0) {
        throw std::runtime_error("failed to stat model file: " + path);
    }
    return static_cast<size_t>(st.st_size);
}

} // namespace floatllm
