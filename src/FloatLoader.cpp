#include "FloatLLM.h"

namespace floatllm {

//  CLI stuff 

void print_usage(const char* exe) {
    cout << GREEN("Usage: " << exe << " --model-path MODEL.gguf --prompt \"text\" [options]\n"
              << "Options:\n"
              << "  --hardware BACKEND         cpu, auto, mps, cuda, vulkan, opencl, sycl\n"
              << "  --crash-threshold MB       RAM safety threshold (default 200)\n"
              << "  --override-storage GB      Manually override storage free-space check\n"
              << "  --override-ram-mb MB          Manually override RAM free-space check\n"
              << "  --session-id NAME          Session label for the dashboard\n"
              << "  --temp-chat                Mark session as temporary\n"
              << "  --ram-limit GB             Hard RAM chunk cap in GB\n"
              << "  --ram-buffer FRACTION      RAM reserve fraction (default 0.20)\n");
}

bool parse_args(int argc, char** argv, CliOptions& opts) {
    for (int i = 1; i < argc; ++i) {
        const string arg = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error(string("missing value for ") + name);
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
        } else if (arg == "--override-ram-mb") {
            opts.override_ram_mb = std::stod(need_value("--override-ram-mb"));
        } else if (arg == "--session-id") {
            opts.session_id = need_value("--session-id");
        } else if (arg == "--temp-chat") {
            opts.temp_chat = true;
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

//  Loader class 

Loader::Loader(const string& path, double allowed_ram_mb, ComputeEngine& eng)
    : model_path(path), allowed_ram_bytes(static_cast<size_t>(allowed_ram_mb * 1024.0 * 1024.0)), engine(eng) {
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

Loader::~Loader() {
    cleanup();
}

std::vector<TensorInfo> Loader::parse_gguf_metadata() const {
    cout << "[FloatLLM] Scanning GGUF metadata for building " << model_path << "...\n";
    std::vector<TensorInfo> tensors;
    const int64_t n_tensors = gguf_get_n_tensors(gguf_ctx);
    tensors.reserve(static_cast<size_t>(n_tensors));

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor* tensor = ggml_get_tensor(ggml_ctx, name);
        if (!tensor) {
            throw std::runtime_error(string("tensor not found in ggml context: ") + name);
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

    cout << "[FloatLLM] Discovered " << tensors.size() << " individual tensors in the model architecture.\n";
    return tensors;
}

void Loader::set_allowed_ram_mb(double mb) {
    allowed_ram_bytes = static_cast<size_t>(mb * 1024.0 * 1024.0);
}

void Loader::build_dynamic_chunks(const std::vector<TensorInfo>& tensors) {
    cout << "[FloatLLM] Chucking Engine Active. Max RAM per block: "
              << (allowed_ram_bytes / (1024.0 * 1024.0)) << " MB\n";

    current_chunk.clear();
    chunks.clear();

    size_t current_chunk_size = 0;
    for (const auto& tensor : tensors) {
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

    cout << "[FloatLLM] Model succesfuly sliced into " << chunks.size() << " dynamic blocks\n";
}

void Loader::stream_all_chunks() const {
    for (const auto& chunk : chunks) {
        stream_chunk(chunk.id);
    }
}

// streams tensor data directly into the engine via reference
void Loader::stream_chunk(int chunk_id) const {
    const auto it = std::find_if(chunks.begin(), chunks.end(), [chunk_id](const Chunk& chunk) { return chunk.id == chunk_id; });
    if (it == chunks.end()) {
        return;
    }

    for (const auto& tensor : it->tensors) {
        const void* raw_ptr = static_cast<const uint8_t*>(mapped) + gguf_get_data_offset(gguf_ctx) + tensor.offset;
        // calls engine method directly through reference - no globals involved
        engine.map_tensor(
            tensor.name.c_str(),
            tensor.type,
            const_cast<void*>(raw_ptr),
            tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3],
            chunk_id);
    }
}

void Loader::cleanup() {
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

} // namespace floatllm
