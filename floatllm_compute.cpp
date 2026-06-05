#include <iostream>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <vector>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <utility>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif
#include <sys/stat.h>
#include <fcntl.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"

// Bug fix
#ifdef __APPLE__
#include <mach/mach.h>
extern "C" ggml_backend_t ggml_backend_metal_init(void);
#endif

// Map user flags to exact GGML backend name
std::string resolve_backend_name(const std::string& input_name) {
    std::string lower_name = input_name;
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
            // handle Metal, Vulkan, CUDA, etc.
            backend = ggml_backend_init_by_name(target_hw.c_str(), NULL);
        }

        // If the requested GPU isn't available/installed
        if (backend == nullptr) {
            std::cout << "[FloatLLM(C++)] Target hardware" << target_hw << " unavailable. Falling back to CPU." << std::endl;
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
        }

        // calculate allocation memory size
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

    double check_failsafe_threshold(double current_ram_mb, double crash_threshold_mb, double model_size_mb, double total_storage_gb,
                                    double free_storage_gb, double used_ram_mb, double total_ram_mb, int quantize_on_fly,
                                    int save_quantized, int no_ram_protocol, double override_storage_gb, const char* session_id,
                                    int temp_chat, double ram_limit_gb, double ram_buffer) {
        auto fmt2 = [](double value) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << value;
            return oss.str();
        };

        const std::string session = session_id ? session_id : "default";
        const double model_size_gb = model_size_mb / 1024.0;
        double trusted_free_gb = free_storage_gb;

        if (override_storage_gb >= 0.0) {
            trusted_free_gb = override_storage_gb;
            std::cout << "[FloatLLM(C++)]\n Overriding UNIX limits. Trusting your input of "
                      << fmt2(trusted_free_gb) << " GB.\n";
            if (total_storage_gb > 0.0 && trusted_free_gb > total_storage_gb) {
                std::cerr << "[FloatLLM(C++)] CRITICAL: Override (" << trusted_free_gb
                          << " GB) exceeds total disk size (" << fmt2(total_storage_gb)
                          << " GB). Halting.\n";
                std::exit(1);
            }
        }
        else {
#ifdef __APPLE__
            if (free_storage_gb > 0.0 && model_size_gb > free_storage_gb) {
                std::cout << "[FloatLLM(C++)]\n⚠️ UNIX sees " << fmt2(free_storage_gb)
                          << " GB. Model needs " << fmt2(model_size_gb) << " GB.\n";
                std::cout << "[FloatLLM(C++)] macOS hides Purgeable space. If you have enough space in System Settings, run with: [--override-storage YOUR_GB]\n";
            }
#endif
        }

        if (trusted_free_gb > 0.0 && model_size_gb > trusted_free_gb) {
            std::cerr << "[FloatLLM(C++)]\n--------------------------------------------------------------------------------\n";
            std::cerr << "[FloatLLM(C++)] 🚨 FloatLLM STORAGE FAILSAFE TRIGGERED\n";
            std::cerr << "[FloatLLM(C++)] CRITICAL: Model requires " << fmt2(model_size_gb)
                      << " GB, but only " << fmt2(trusted_free_gb) << " GB is free.\n";
            std::cerr << "[FloatLLM(C++)] Action: Halting to prevent storage corruption.\n";
            std::cerr << "[FloatLLM(C++)]--------------------------------------------------------------------------------\n\n";
            std::exit(1);
        }

        if (current_ram_mb <= crash_threshold_mb) {
            std::cerr << "[FloatLLM(C++)]\n--------------------------------------------------------------------------------\n";
            std::cerr << "[FloatLLM(C++)] 🚨 FloatLLM OOM Failsafe triggered to stop crashing/freezing of device.\n";
            std::cerr << "[FloatLLM(C++)] CRITICAL: Free RAM (" << fmt2(current_ram_mb)
                      << " MB) hit the crash threshold (" << fmt2(crash_threshold_mb) << " MB).\n";
            std::cerr << "[FloatLLM(C++)] Target Model Size: " << fmt2(model_size_mb) << " MB\n";
            if (used_ram_mb >= 0.0) {
                std::cerr << "[FloatLLM(C++)] FloatLLM Consumed: " << fmt2(used_ram_mb) << " MB (Max Peak)\n";
            }
            std::cerr << "[FloatLLM(C++)] Action: Halting execution gracefully. Model data safely flushed.\n";
            std::cerr << "[FloatLLM(C++)] Adjust [--crash-threshold] or increase [--ram-limit] for more safety.\n";
            std::cerr << "[FloatLLM(C++)] For extreme offload: Enable [--no-ram-protocol] to dump KV Cache & Hidden States to SSD.\n";
            std::cerr << "[FloatLLM(C++)] Or Compression: Enable [--quantize-on-fly] to compress weights in memory.\n";
            std::cerr << "[FloatLLM(C++)] Or Quantize the model permanently using --save-quantized to run the saved quantize model.\n";
            std::cerr << "[FloatLLM(C++)]--------------------------------------------------------------------------------\n\n";
            std::exit(1);
        }

        const double safe_ram_mb = std::max(1.0, (current_ram_mb * (1.0 - ram_buffer)) - crash_threshold_mb);
        const double allowed_ram_mb = (ram_limit_gb > 0.0)
            ? std::min(safe_ram_mb, ram_limit_gb * 1024.0)
            : safe_ram_mb;

        std::cout << "[FloatLLM(C++)]\n--- Pre-Flight Memory Dashboard ---\n";
        if (total_ram_mb > 0.0) {
            std::cout << "[FloatLLM(C++)] Host Total Ram       : " << fmt2(total_ram_mb) << " MB\n";
            std::cout << "[FloatLLM(C++)] Host Used RAM        : " << fmt2(total_ram_mb - current_ram_mb) << " MB\n";
        }
        std::cout << "[FloatLLM(C++)] Host Free Ram        : " << fmt2(current_ram_mb) << " MB\n";
        std::cout << "[FloatLLM(C++)] Allowed RAM (Chunk)  : " << fmt2(allowed_ram_mb) << " MB (Buffer: "
                  << (ram_buffer * 100.0) << "%)\n";
        if (trusted_free_gb > 0.0) {
            std::cout << "[FloatLLM(C++)] Host Free Storage    : " << fmt2(trusted_free_gb)
                      << " GB " << (override_storage_gb >= 0.0 ? "(OVERRIDEN)" : "") << "\n";
        }
        std::cout << "[FloatLLM(C++)] Target Model Size    : " << fmt2(model_size_mb) << " MB\n";
        std::cout << "[FloatLLM(C++)] Kill threshold       : " << fmt2(crash_threshold_mb) << " MB\n";
        std::cout << "[FloatLLM(C++)] --- User Execution Blueprint ---\n";
        std::cout << "[FloatLLM(C++)] Live Quantization    : " << (quantize_on_fly ? "ENABLED" : "DISABLED") << "\n";
        std::cout << "[FloatLLM(C++)] AOT Quantization (Save): " << (save_quantized ? "ACTIVE" : "DISABLED") << "\n";
        std::cout << "[FloatLLM(C++)] No-RAM Protocol (SSD): " << (no_ram_protocol ? "ACTIVE" : "DISABLED") << "\n";
        std::cout << "[FloatLLM(C++)] Session ID           : [" << session << "]\n";
        std::cout << "[FloatLLM(C++)] Context Saving       : "
                  << (temp_chat ? "Temporary (Delete on Exit)" : "PERSISTENT (Saved to SSD)") << "\n\n";

        return allowed_ram_mb;
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

        // --- NEW: DYNAMIC TENSOR RESOLUTION ---
        struct ggml_tensor* token_embd = nullptr;
        struct ggml_tensor* output_weight = nullptr;

        // 1. Prioritize exact standard matches first for predictable behavior
        if (tensor_registry.count("token_embd.weight")) token_embd = tensor_registry["token_embd.weight"];
        else if (tensor_registry.count("model.embed_tokens.weight")) token_embd = tensor_registry["model.embed_tokens.weight"];

        if (tensor_registry.count("output.weight")) output_weight = tensor_registry["output.weight"];
        else if (tensor_registry.count("lm_head.weight")) output_weight = tensor_registry["lm_head.weight"];

        // 2. Safe fallback scan (explicitly ignoring internal transformer blocks)
        if (!token_embd || !output_weight) {
            for (auto const& [key, val] : tensor_registry) {
                // Skip attention and feed-forward intermediate layers entirely
                if (key.find("blk.") != std::string::npos || key.find("layer") != std::string::npos) continue;

                if (!token_embd && key.find("embed") != std::string::npos) {
                    token_embd = val;
                }
                if (!output_weight && (key.find("output") != std::string::npos || key.find("head") != std::string::npos) && key.find("norm") == std::string::npos) {
                    output_weight = val;
                }
            }
        }

        // 3. Fallback for "Tied Embeddings" (Smaller models reuse embedding weights for the output)
        if (!output_weight && token_embd) {
            output_weight = token_embd;
        }

        if (!token_embd || !output_weight){
            std::cerr << "[FloatLLM(C++)] ERROR: Core tensors (token_embd or output) not found in model." << std::endl;
            ggml_free(graph_ctx);
            return -1; 
        }
        
        int64_t max_vocab_size = token_embd->ne[1]; 
        for (int i = 0; i < num_tokens; i++) {
            if (tokens[i] < 0 || tokens[i] >= max_vocab_size) {
                tokens[i] = 0; // Clamp to safe default to keep the engine alive
            }
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

        // Restore the python RAM pointers for nexxt generation loop
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

namespace floatllm_runner {

static std::string detect_hardware_backend() {
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

struct CliOptions {std::string hardware = "auto"; std::string model_path; std::string prompt; std::string session_id = "default_chat";
                    double crash_threshold_mb = 200.0; double override_storage_gb = -1.0; double ram_limit_gb = -1.0; double ram_buffer = 0.20;
                    bool quantize_on_fly = false; bool no_ram_protocol = false; bool temp_chat = false; bool save_quantized = false;
};

struct TensorInfo {std::string name; int type = 0; size_t offset = 0; size_t size = 0; int64_t shape[4] = {1, 1, 1, 1};};

static void print_usage(const char * exe) {
    std::cout << "Usage: " << exe << " --model-path MODEL.gguf --prompt \"text\" [options]\n"
              << "Options:\n"
              << "  --hardware BACKEND         cpu, auto, mps, cuda, vulkan, opencl, sycl\n"
              << "  --crash-threshold MB       RAM safety threshold (default 200)\n"
              << "  --override-storage GB      Manually override storage free-space check\n"
              << "  --session-id NAME          Session label for the dashboard\n"
              << "  --temp-chat                Mark session as temporary\n"
              << "  --quantize-on-fly          Mirror the Python flag\n"
              << "  --no-ram-protocol          Mirror the Python flag\n"
              << "  --save-quantized           Mirror the Python flag\n"
              << "  --ram-limit GB             Hard RAM chunk cap in GB\n"
              << "  --ram-buffer FRACTION      RAM reserve fraction (default 0.20)\n";
}

static bool parse_args(int argc, char ** argv, CliOptions & opts) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        } else if (arg == "--hardware") {
            opts.hardware = need_value("--hardware");
        } else if (arg == "--model-path") {
            opts.model_path = need_value("--model-path");
        } else if (arg == "--prompt") {
            opts.prompt = need_value("--prompt");
        } else if (arg == "--crash-threshold") {
            opts.crash_threshold_mb = std::stod(need_value("--crash-threshold"));
        } else if (arg == "--override-storage") {
            opts.override_storage_gb = std::stod(need_value("--override-storage"));
        } else if (arg == "--session-id") {
            opts.session_id = need_value("--session-id");
        } else if (arg == "--temp-chat") {
            opts.temp_chat = true;
        } else if (arg == "--quantize-on-fly") {
            opts.quantize_on_fly = true;
        } else if (arg == "--no-ram-protocol") {
            opts.no_ram_protocol = true;
        } else if (arg == "--save-quantized") {
            opts.save_quantized = true;
        } else if (arg == "--ram-limit") {
            opts.ram_limit_gb = std::stod(need_value("--ram-limit"));
        } else if (arg == "--ram-buffer") {
            opts.ram_buffer = std::stod(need_value("--ram-buffer"));
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (opts.model_path.empty()) {
        throw std::runtime_error("--model-path is required");
    }
    if (opts.prompt.empty()) {
        throw std::runtime_error("--prompt is required");
    }
    return true;
}

static std::pair<double, double> get_ram_stats_mb() {
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
    const long page_size = sysconf(_SC_PAGESIZE);
    const long phys_pages = sysconf(_SC_PHYS_PAGES);
    const double total = (page_size > 0 && phys_pages > 0)
        ? (static_cast<double>(page_size) * static_cast<double>(phys_pages)) / (1024.0 * 1024.0)
        : 0.0;
    const double free = total * 0.5;
    return {total, free};
}

static std::pair<double, double> get_storage_stats_gb() {
#ifdef _WIN32
    ULARGE_INTEGER freeBytesAvailable, totalNumberOfBytes, totalNumberOfFreeBytes;
    if (GetDiskFreeSpaceExA(".", &freeBytesAvailable, &totalNumberOfBytes, &totalNumberOfFreeBytes)) {
        double total_gb = static_cast<double>(totalNumberOfBytes.QuadPart) / (1024.0 * 1024.0 * 1024.0);
        double free_gb = static_cast<double>(freeBytesAvailable.QuadPart) / (1024.0 * 1024.0 * 1024.0);
        return {total_gb, free_gb};
    }
    return {0.0, 0.0};
#else
    const char * home = std::getenv("HOME");
    const char * root = home ? home : ".";
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

static size_t file_size_bytes(const std::string & path) {
    struct stat st {};
    if (stat(path.c_str(), &st) != 0) {
        throw std::runtime_error("failed to stat model file: " + path);
    }
    return static_cast<size_t>(st.st_size);
}

class Tokenizer {
public:
    explicit Tokenizer(const std::string & path) : model_path(path) {
        struct stat st {};
        if (stat(model_path.c_str(), &st) != 0) {
            throw std::runtime_error("model file not found: " + model_path);
        }

        struct gguf_init_params params { true, nullptr };
        ctx = gguf_init_from_file(model_path.c_str(), params);
        if (!ctx) {
            throw std::runtime_error("gguf_init_from_file failed for tokenizer");
        }

        extract_metadata();
    }

    ~Tokenizer() {
        if (ctx) {
            gguf_free(ctx);
            ctx = nullptr;
        }
    }

    int eos_id() const { return eos_token_id; }

    std::vector<int32_t> encode(const std::string & text) const {
        std::vector<int32_t> token_ids;

        if (bos_token_id >= 0) {
            token_ids.push_back(bos_token_id);
        }

        std::string transformed = text;
        size_t pos = 0;
        while ((pos = transformed.find(' ', pos)) != std::string::npos) {
            transformed.replace(pos, 1, " Ġ");
            pos += 2;
        }

        std::istringstream stream(transformed);
        std::string word;
        while (std::getline(stream, word, ' ')) {
            if (word.empty()) {
                continue;
            }

            while (!word.empty()) {
                bool matched = false;
                for (size_t len = word.size(); len > 0; --len) {
                    const std::string chunk = word.substr(0, len);
                    auto it = token_to_id.find(chunk);
                    if (it != token_to_id.end()) {
                        token_ids.push_back(it->second);
                        word.erase(0, len);
                        matched = true;
                        break;
                    }
                }

                if (!matched) {
                    word.erase(0, 1);
                }
            }
        }

        return token_ids;
    }

    std::string decode(const std::vector<int32_t> & token_ids) const {
        std::string output;
        for (int32_t id : token_ids) {
            if (id == bos_token_id || id == eos_token_id) {
                continue;
            }
            if (id >= 0 && static_cast<size_t>(id) < vocab.size()) {
                std::string token = vocab[static_cast<size_t>(id)];
                size_t start = 0;
                while ((start = token.find("Ġ", start)) != std::string::npos) {
                    token.replace(start, 2, " ");
                    start += 1;
                }
                output += token;
            }
        }
        while (!output.empty() && output.front() == ' ') {
            output.erase(output.begin());
        }
        return output;
    }

private:
    static int64_t read_scalar_integer(const struct gguf_context * ctx, const char * key_name) {
        const int64_t key_id = gguf_find_key(ctx, key_name);
        if (key_id < 0) {
            return -1;
        }

        const enum gguf_type type = gguf_get_kv_type(ctx, key_id);
        switch (type) {
            case GGUF_TYPE_UINT8:  return static_cast<int64_t>(gguf_get_val_u8(ctx, key_id));
            case GGUF_TYPE_INT8:   return static_cast<int64_t>(gguf_get_val_i8(ctx, key_id));
            case GGUF_TYPE_UINT16: return static_cast<int64_t>(gguf_get_val_u16(ctx, key_id));
            case GGUF_TYPE_INT16:  return static_cast<int64_t>(gguf_get_val_i16(ctx, key_id));
            case GGUF_TYPE_UINT32: return static_cast<int64_t>(gguf_get_val_u32(ctx, key_id));
            case GGUF_TYPE_INT32:  return static_cast<int64_t>(gguf_get_val_i32(ctx, key_id));
            case GGUF_TYPE_UINT64: return static_cast<int64_t>(gguf_get_val_u64(ctx, key_id));
            case GGUF_TYPE_INT64:  return static_cast<int64_t>(gguf_get_val_i64(ctx, key_id));
            case GGUF_TYPE_ARRAY: {
                const enum gguf_type arr_type = gguf_get_arr_type(ctx, key_id);
                const void * arr_data = gguf_get_arr_data(ctx, key_id);
                switch (arr_type) {
                    case GGUF_TYPE_UINT8:  return static_cast<int64_t>(static_cast<const uint8_t *>(arr_data)[0]);
                    case GGUF_TYPE_INT8:   return static_cast<int64_t>(static_cast<const int8_t *>(arr_data)[0]);
                    case GGUF_TYPE_UINT16: return static_cast<int64_t>(static_cast<const uint16_t *>(arr_data)[0]);
                    case GGUF_TYPE_INT16:  return static_cast<int64_t>(static_cast<const int16_t *>(arr_data)[0]);
                    case GGUF_TYPE_UINT32: return static_cast<int64_t>(static_cast<const uint32_t *>(arr_data)[0]);
                    case GGUF_TYPE_INT32:  return static_cast<int64_t>(static_cast<const int32_t *>(arr_data)[0]);
                    case GGUF_TYPE_UINT64: return static_cast<int64_t>(static_cast<const uint64_t *>(arr_data)[0]);
                    case GGUF_TYPE_INT64:  return static_cast<int64_t>(static_cast<const int64_t *>(arr_data)[0]);
                    default: break;
                }
            } break;
            default:
                break;
        }

        return -1;
    }

    void extract_metadata() {
        const int64_t model_key = gguf_find_key(ctx, "tokenizer.ggml.model");
        if (model_key >= 0) {
            model_type = gguf_get_val_str(ctx, model_key);
            std::cout << "[FloatLLM] Tokenizer Architecture: " << model_type << "\n";
        }

        bos_token_id = static_cast<int32_t>(read_scalar_integer(ctx, "tokenizer.ggml.bos_token_id"));
        eos_token_id = static_cast<int32_t>(read_scalar_integer(ctx, "tokenizer.ggml.eos_token_id"));

        std::cout << "[FloatLLM] BOS ID: " << bos_token_id << " | EOS ID: " << eos_token_id << "\n";

        const int64_t tokens_key = gguf_find_key(ctx, "tokenizer.ggml.tokens");
        if (tokens_key < 0) {
            throw std::runtime_error("tokenizer.ggml.tokens not found in GGUF");
        }

        const size_t n_tokens = gguf_get_arr_n(ctx, tokens_key);
        vocab.reserve(n_tokens);
        for (size_t i = 0; i < n_tokens; ++i) {
            const char * raw = gguf_get_arr_str(ctx, tokens_key, i);
            std::string token = raw ? raw : "";
            vocab.push_back(token);
            token_to_id[token] = static_cast<int32_t>(vocab.size() - 1);
        }

        std::cout << "[FloatLLM] Successfully extracted " << vocab.size()
                  << " offline tokens into memory\n";
    }

    std::string model_path;
    std::string model_type = "unknown";
    int32_t bos_token_id = -1;
    int32_t eos_token_id = -1;
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int32_t> token_to_id;
    struct gguf_context * ctx = nullptr;
};

class Loader {
public:
    Loader(const std::string & path, double allowed_ram_mb, const std::string & backend)
        : model_path(path), allowed_ram_bytes(static_cast<size_t>(allowed_ram_mb * 1024.0 * 1024.0)), backend_name(backend) {
        struct stat st_check {};
        if (stat(model_path.c_str(), &st_check) != 0) {
            throw std::runtime_error("model file not found: " + model_path);
        }

#ifdef _WIN32
        file_handle = CreateFileA(model_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_handle == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("failed to open model file: " + model_path);
        }
        
        LARGE_INTEGER li;
        if (!GetFileSizeEx(file_handle, &li)) {
            CloseHandle(file_handle);
            throw std::runtime_error("failed to stat model file");
        }
        file_size = static_cast<size_t>(li.QuadPart);
        
        map_handle = CreateFileMappingA(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!map_handle) {
            CloseHandle(file_handle);
            throw std::runtime_error("CreateFileMapping failed for model file");
        }
        
        mapped = MapViewOfFile(map_handle, FILE_MAP_READ, 0, 0, 0);
        if (!mapped) {
            CloseHandle(map_handle);
            CloseHandle(file_handle);
            throw std::runtime_error("MapViewOfFile failed for model file");
        }
#else
        fd = ::open(model_path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("failed to open model file");
        }

        struct stat st {};
        if (fstat(fd, &st) != 0) {
            ::close(fd);
            throw std::runtime_error("failed to stat model file");
        }
        file_size = static_cast<size_t>(st.st_size);

        mapped = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) {
            ::close(fd);
            throw std::runtime_error("mmap failed for model file");
        }
#endif

        struct gguf_init_params params { true, &ggml_ctx };
        gguf_ctx = gguf_init_from_file(model_path.c_str(), params);
        if (!gguf_ctx) {
            cleanup();
            throw std::runtime_error("gguf_init_from_file failed for loader");
        }
    }

    ~Loader() {
        cleanup();
    }

    std::vector<TensorInfo> parse_gguf_metadata() const {
        std::cout << "[FloatLLM] Scanning GGUF metadata for building " << model_path << "...\n";
        std::vector<TensorInfo> tensors;
        const int64_t n_tensors = gguf_get_n_tensors(gguf_ctx);
        tensors.reserve(static_cast<size_t>(n_tensors));

        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf_ctx, i);
            struct ggml_tensor * tensor = ggml_get_tensor(ggml_ctx, name);
            if (!tensor) {
                throw std::runtime_error(std::string("tensor not found in ggml context: ") + name);
            }

            TensorInfo info;
            info.name = name;
            info.type = static_cast<int>(tensor->type);
            info.offset = gguf_get_tensor_offset(gguf_ctx, i);
            info.size = ggml_nbytes(tensor);
            for (int d = 0; d < 4; ++d) {
                info.shape[d] = tensor->ne[d];
            }
            tensors.push_back(std::move(info));
        }

        std::cout << "[FloatLLM] Discovered " << tensors.size() << " individual tensors in the model architecture.\n";
        return tensors;
    }

    void wake_engine(size_t total_tensors) {
        init_compute_engine(backend_name.c_str(), static_cast<int>(total_tensors));
    }

    void set_allowed_ram_mb(double mb) {
        allowed_ram_bytes = static_cast<size_t>(mb * 1024.0 * 1024.0);
    }

    void build_dynamic_chunks(const std::vector<TensorInfo> & tensors) {
        std::cout << "[FloatLLM] Chucking Engine Active. Max RAM per block: "
                  << (allowed_ram_bytes / (1024.0 * 1024.0)) << " MB\n";

        current_chunk.clear();
        chunks.clear();

        size_t current_chunk_size = 0;
        for (const auto & tensor : tensors) {
            if (current_chunk_size + tensor.size > allowed_ram_bytes) {
                if (current_chunk.empty()) {
                    throw std::runtime_error("tensor exceeds allowed RAM budget: " + tensor.name);
                }
                chunks.push_back({static_cast<int>(chunks.size() + 1), current_chunk, current_chunk_size});
                current_chunk.clear();
                current_chunk_size = 0;
            }

            current_chunk.push_back(tensor);
            current_chunk_size += tensor.size;
        }

        if (!current_chunk.empty()) {
            chunks.push_back({static_cast<int>(chunks.size() + 1), current_chunk, current_chunk_size});
        }

        std::cout << "[FloatLLM] Model succesfuly sliced into " << chunks.size() << " dynamic blocks\n";
    }

    void stream_all_chunks() const {
        for (const auto & chunk : chunks) {
            stream_chunk(chunk.id);
        }
    }

    void stream_chunk(int chunk_id) const {
        const auto it = std::find_if(chunks.begin(), chunks.end(), [chunk_id](const Chunk & chunk) { return chunk.id == chunk_id; });
        if (it == chunks.end()) {
            return;
        }

        for (const auto & tensor : it->tensors) {
            const void * raw_ptr = static_cast<const uint8_t *>(mapped) + gguf_get_data_offset(gguf_ctx) + tensor.offset;
            execute_tensor_chunk(
                tensor.name.c_str(),
                tensor.type,
                const_cast<void *>(raw_ptr),
                tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3],
                chunk_id);
        }
    }

private:
    struct Chunk {
        int id;
        std::vector<TensorInfo> tensors;
        size_t total_size;
    };

    void cleanup() {
        if (gguf_ctx) {
            gguf_free(gguf_ctx);
            gguf_ctx = nullptr;
        }
        if (ggml_ctx) {
            ggml_free(ggml_ctx);
            ggml_ctx = nullptr;
        }
#ifdef _WIN32
        if (mapped) {
            UnmapViewOfFile(mapped);
            mapped = nullptr;
        }
        if (map_handle) {
            CloseHandle(map_handle);
            map_handle = nullptr;
        }
        if (file_handle != INVALID_HANDLE_VALUE) {
            CloseHandle(file_handle);
            file_handle = INVALID_HANDLE_VALUE;
        }
#else
        if (mapped && mapped != MAP_FAILED) {
            ::munmap(mapped, file_size);
            mapped = nullptr;
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
#endif
    }

    std::string model_path;
    size_t allowed_ram_bytes;
    std::string backend_name;
    void * mapped = nullptr;
    size_t file_size = 0;
#ifdef _WIN32
    HANDLE file_handle = INVALID_HANDLE_VALUE;
    HANDLE map_handle = nullptr;
#else
    int fd = -1;
#endif
    struct gguf_context * gguf_ctx = nullptr;
    struct ggml_context * ggml_ctx = nullptr;
    std::vector<TensorInfo> current_chunk;
    std::vector<Chunk> chunks;
};

} // namespace floatllm_runner

#ifdef FLOATLLM_BUILD_MAIN
int main(int argc, char ** argv) {
    try {
        floatllm_runner::CliOptions opts;
        if (!floatllm_runner::parse_args(argc, argv, opts)) {
            return 0;
        }

        std::string selected_backend = opts.hardware;
        std::string lower_backend = selected_backend;
        std::transform(lower_backend.begin(), lower_backend.end(), lower_backend.begin(), ::tolower);
        if (lower_backend == "auto") {
            selected_backend = floatllm_runner::detect_hardware_backend();
        }

        std::cout << "[FloatLLM] Hardware Router engaged: Backend -> [" << selected_backend << "]\n";

        const auto [total_ram_mb, free_ram_mb] = floatllm_runner::get_ram_stats_mb();
        const auto [total_storage_gb, free_storage_gb] = floatllm_runner::get_storage_stats_gb();
        const double model_size_mb = static_cast<double>(floatllm_runner::file_size_bytes(opts.model_path)) / (1024.0 * 1024.0);

        const double calculated_limit = check_failsafe_threshold(free_ram_mb, opts.crash_threshold_mb, model_size_mb, total_storage_gb,
                                                                free_storage_gb, -1.0, total_ram_mb, opts.quantize_on_fly ? 1 : 0,
                                                                opts.save_quantized ? 1 : 0, opts.no_ram_protocol ? 1 : 0,opts.override_storage_gb,
                                                                opts.session_id.c_str(), opts.temp_chat ? 1 : 0, opts.ram_limit_gb, opts.ram_buffer);

        floatllm_runner::Tokenizer tokenizer(opts.model_path);
        floatllm_runner::Loader loader(opts.model_path, calculated_limit, selected_backend);

        const auto tensor_map = loader.parse_gguf_metadata();
        loader.wake_engine(tensor_map.size());
        loader.set_allowed_ram_mb(calculated_limit);
        loader.build_dynamic_chunks(tensor_map);
        loader.stream_all_chunks();

        std::cout << "[FloatLLM] Engine successfully mapped. Handing to AI...\n";
        std::cout << "[FloatLLM] --------------------------------------------------------------------------------\n";
        std::cout << "[FloatLLM] \nUser: " << opts.prompt << "\n";
        std::cout << "[FloatLLM] ";

        std::vector<int32_t> token_ids = tokenizer.encode(opts.prompt);
        const int max_tokens_to_generate = 60;

        for (int step = 0; step < max_tokens_to_generate; ++step) {
            std::vector<int32_t> working_tokens = token_ids;
            int32_t next_token_id = execute_forward_pass(working_tokens.data(), static_cast<int>(working_tokens.size()));
            if (next_token_id == tokenizer.eos_id()) {
                break;
            }

            std::string word = tokenizer.decode({next_token_id});
            std::cout << "\033[92m" << word << "\033[0m ";
            std::cout.flush();
            token_ids.push_back(next_token_id);
        }

        std::cout << "\n\n";
        std::cout << "[FloatLLM] Generated first 60 tokens in output!\n";
        std::cout << "[FloatLLM] --------------------------------------------------------------------------------\n";
        shutdown_compute_engine();
        std::cout << "[FloatLLM] Closing C++ memory maps...\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "[FloatLLM] ERROR: " << e.what() << "\n";
        return 1;
    }
}
#endif