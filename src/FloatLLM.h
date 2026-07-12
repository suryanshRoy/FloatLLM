#pragma once

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
#include "FloatUI.h"

#ifdef __APPLE__
#include <mach/mach.h>
extern "C" ggml_backend_t ggml_backend_metal_init(void);
#endif

using std::cout;
using std::cerr;
using std::endl;
using std::string;

#define YELLOW(text) "\033[1;33m" text "\033[0m"  // for warnings
#define GREEN(text) "\033[1;32m" text "\033[0m"   // for successful execution of code
#define RED(text) "\033[1;31m" text "\033[0m"      // for errors
#define PURPLE(text) "\033[38;2;153;51;255m" text "\033[0m" // for informative messages

namespace floatllm {

class ComputeEngine {
public:
    ComputeEngine() = default;
    ~ComputeEngine();

    // fire up the backend and alllocate graph memory
    void init(const string& backend_name, int total_tensors);

    void set_tied_embd(bool tied) {
        tied_embeddings = tied;
    }

    // cleanup everything - vram, ram, the whole thing
    void shutdown();

    // map a single tensor into the compute graph
    void map_tensor(const char* name, int type, void* data,
                    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, int chunk_id);

    // run forward pass - returns the next token id
    int32_t forward_pass(int32_t* tokens, int num_tokens);

    //  static utilities (dont need an engine instance) 

    // auto detect best GPU/CPU backend for the current OS
    static string detect_hardware();

private:
    // resolves user-freindly backend names to ggml ones
    static string resolve_backend(const string& input_name);

    struct ggml_context* ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocr = nullptr;
    std::unordered_map<string, struct ggml_tensor*> tensor_registry;
    bool tied_embeddings = false;
};


// handle BPE style tokenization
class Tokenizer {
public:
    explicit Tokenizer(const string& path);
    ~Tokenizer();

    int eos_id() const { return eos_token_id; }

    // text -> token ids
    std::vector<int32_t> encode(const string& text) const;

    // token ids -> readable text
    string decode(const std::vector<int32_t>& token_ids) const;

private:
    static int64_t read_scalar_integer(const struct gguf_context* ctx, const char* key_name);
    void extract_metadata();

    string model_path;
    string model_type = "unknown";
    int32_t bos_token_id = -1;
    int32_t eos_token_id = -1;
    std::vector<string> vocab;
    std::unordered_map<string, int32_t> token_to_id;
    struct gguf_context* ctx = nullptr;
};


// TensorInfo & CliOptions - simple data structs
struct TensorInfo {
    string name;
    int type = 0;
    size_t offset = 0;
    size_t size = 0;
    int64_t shape[4] = {1, 1, 1, 1};
};

struct CliOptions {
    string hardware = "auto";
    string model_path;
    string prompt;
    string session_id = "default_chat";
    double crash_threshold_mb = 200.0;
    double override_storage_gb = -1.0;
    double override_ram_mb = -1.0;
    double ram_limit_gb = -1.0;
    double ram_buffer = 0.20;
    bool temp_chat = false;
    int max_tokens = 60;
    bool quantize_memory = false;
};


class Loader {
public:
    Loader(const string& path, double allowed_ram_mb, ComputeEngine& engine);
    ~Loader();

    std::vector<TensorInfo> parse_gguf_metadata() const;
    void set_allowed_ram_mb(double mb);
    void build_dynamic_chunks(const std::vector<TensorInfo>& tensors);
    void stream_all_chunks() const;
    void stream_chunk(int chunk_id) const;

private:
    struct Chunk {
        int id;
        std::vector<TensorInfo> tensors;
        size_t total_size;
    };

    void cleanup();

    string model_path;
    size_t allowed_ram_bytes;
    ComputeEngine& engine; // reference to engine, not global state
    void* mapped = nullptr;
    size_t file_size = 0;
#ifdef _WIN32
    HANDLE file_handle = INVALID_HANDLE_VALUE;
    HANDLE map_handle = nullptr;
#else
    int fd = -1;
#endif
    struct gguf_context* gguf_ctx = nullptr;
    struct ggml_context* ggml_ctx = nullptr;
    std::vector<TensorInfo> current_chunk;
    std::vector<Chunk> chunks;
};

//  CLI helpers 
void print_usage(const char* exe);
bool parse_args(int argc, char** argv, CliOptions& opts);
void lineSep();
inline void linesep() { lineSep(); }

} // namespace floatllm
using floatllm::lineSep;