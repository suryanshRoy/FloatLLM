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
#include <cmath>
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
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "gguf.h"

#ifdef __APPLE__
#include <mach/mach.h>
extern "C" ggml_backend_t ggml_backend_metal_init(void);
#endif

// NOTE I use these convensions for my ease
using std::cout;
using std::cerr;
using std::endl;
using std::string;

#define YELLOW(text) "\033[1;33m" text "\033[0m"  // for warnings
#define GREEN(text) "\033[1;32m" text "\033[0m"   // for successful execution of code
#define RED(text) "\033[1;31m" text "\033[0m"      // for errors
#define PURPLE(text) "\033[38;2;153;51;255m" text "\033[0m" // for informative messages

namespace floatllm {

// model hyper-parameters pulled from the GGUF metadata
struct ModelHParams {
    string arch = "unknown";
    int32_t n_layer = 0;
    int32_t n_head = 0;
    int32_t n_head_kv = 0;
    int32_t n_embd = 0;
    int32_t n_ff = 0;
    int32_t n_rot = 0;          // rope dimension count
    int32_t n_ctx_train = 0;    // max context the model was trained on
    int32_t n_ctx = 0;          // runtime context window (kv cache slots)
    float f_norm_rms_eps = 1e-5f;
    float rope_freq_base = 10000.0f;

    int32_t head_dim() const { return n_head > 0 ? n_embd / n_head : 0; }
    int32_t n_embd_kv() const { return head_dim() * n_head_kv; }
    bool valid() const { return n_layer > 0 && n_head > 0 && n_embd > 0; }
};

class ComputeEngine {
public:
    ComputeEngine() = default;
    ~ComputeEngine();

    // fire up the backend and alllocate graph memory
    void init(const string& backend_name, int total_tensors, double slack_buffer_mb = 64.0);

    // give the engine the model architecture info (layers, heads, rope, ...)
    void set_hparams(const ModelHParams& hp);

    // move weights into backend memory once (GPU) or keep zero-copy mmap (CPU),
    // then allocate the persistent KV cache
    bool finalize_weights();

    // cleanup everything - vram, ram, the whole thing
    void shutdown();

    // map a single tensor into the compute graph
    void map_tensor(const char* name, int type, void* data,
                    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, int chunk_id);

    // run forward pass - returns the next token id
    // n_past = number of tokens already stored in the kv cache
    int32_t forward_pass(const int32_t* tokens, int num_tokens, int n_past = 0);

    //  static utilities (dont need an engine instance) 

    // pre-flight saftey check - will exit() if things are bad
    static double check_threshold(double current_ram_mb, double crash_threshold_mb, double model_size_mb,
                                  double total_storage_gb, double free_storage_gb, double used_ram_mb,
                                  double total_ram_mb, double override_storage_gb, const char* session_id,
                                  int temp_chat, double ram_limit_gb, double ram_buffer);

    // auto detect best GPU/CPU backend for the current OS
    static string detect_hardware();

    // get system ram stats {total_mb, free_mb}
    static std::pair<double, double> get_ram_stats_mb();

    // get disk space {total_gb, free_gb}
    static std::pair<double, double> get_storage_stats_gb();

    // simple file size helper
    static size_t file_size_bytes(const string& path);

private:
    // resolves user-freindly backend names to ggml ones
    static string resolve_backend(const string& input_name);

    struct ggml_tensor* get_weight(const string& name) const;
    struct ggml_tensor* build_transformer_graph(struct ggml_context* graph_ctx, struct ggml_cgraph* gf,
                                                struct ggml_tensor* inp_tokens, struct ggml_tensor* inp_pos,
                                                struct ggml_tensor* kq_mask, int num_tokens, int n_past);

    struct ggml_context* ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocr = nullptr;
    std::unordered_map<string, struct ggml_tensor*> tensor_registry;
    std::unordered_map<string, void*> raw_data_registry; // mmap pointers for deferred GPU upload

    ModelHParams hparams;
    bool backend_is_cpu = true;
    bool weights_finalized = false;
    ggml_backend_buffer_t weight_buffer = nullptr;

    // persistent KV cache (one K and one V tensor per layer)
    struct ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buffer = nullptr;
    std::vector<struct ggml_tensor*> k_cache;
    std::vector<struct ggml_tensor*> v_cache;
};


// Tokenizer - reads vocab from gguf, does encode/decode
// handles BPE style tokenization
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
    int max_tokens = 60;
    int context_length = 4096;
    double slack_buffer_mb = 64.0;
    double crash_threshold_mb = 200.0;
    double override_storage_gb = -1.0;
    double ram_limit_gb = -1.0;
    double ram_buffer = 0.20;
    bool temp_chat = false;
};


// Loader - memory maps the gguf file, parses metadata,
// chunks it into ram-friendly blocks and streams into engine
// takes a ComputeEngine reference - no global state
class Loader {
public:
    Loader(const string& path, double allowed_ram_mb, ComputeEngine& engine);
    ~Loader();

    std::vector<TensorInfo> parse_gguf_metadata() const;
    ModelHParams parse_hparams() const;
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

} // namespace floatllm
