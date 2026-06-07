#include "FloatLLM.h"

namespace floatllm {

Tokenizer::Tokenizer(const string& path) : model_path(path) {
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

Tokenizer::~Tokenizer() {
    if (ctx) {
        gguf_free(ctx);
        ctx = nullptr;
    }
}

std::vector<int32_t> Tokenizer::encode(const string& text) const {
    std::vector<int32_t> token_ids;

    if (bos_token_id >= 0) {
        token_ids.push_back(bos_token_id);
    }

    string transformed = text;
    size_t pos = 0;
    while ((pos = transformed.find(' ', pos)) != string::npos) {
        transformed.replace(pos, 1, " Ġ");
        pos += 2;
    }

    std::istringstream stream(transformed);
    string word;
    while (std::getline(stream, word, ' ')) {
        if (word.empty()) {
            continue;
        }

        // greedy longest match tokenization
        while (!word.empty()) {
            bool matched = false;
            for (size_t len = word.size(); len > 0; --len) {
                const string chunk = word.substr(0, len);
                auto it = token_to_id.find(chunk);
                if (it != token_to_id.end()) {
                    token_ids.push_back(it->second);
                    word.erase(0, len);
                    matched = true;
                    break;
                }
            }

            if (!matched) {
                word.erase(0, 1); // skip unknown chars
            }
        }
    }

    return token_ids;
}

string Tokenizer::decode(const std::vector<int32_t>& token_ids) const {
    string output;
    for (int32_t id : token_ids) {
        if (id == bos_token_id || id == eos_token_id) {
            continue;
        }
        if (id >= 0 && static_cast<size_t>(id) < vocab.size()) {
            string token = vocab[static_cast<size_t>(id)];
            size_t start = 0;
            while ((start = token.find("Ġ", start)) != string::npos) {
                token.replace(start, 2, " ");
                start += 1;
            }
            output += token;
        }
    }
    // trim leading space
    while (!output.empty() && output.front() == ' ') {
        output.erase(output.begin());
    }
    return output;
}

// handles different integer types gguf might store values as
int64_t Tokenizer::read_scalar_integer(const struct gguf_context* ctx, const char* key_name) {
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
            const void* arr_data = gguf_get_arr_data(ctx, key_id);
            switch (arr_type) {
                case GGUF_TYPE_UINT8:  return static_cast<int64_t>(static_cast<const uint8_t*>(arr_data)[0]);
                case GGUF_TYPE_INT8:   return static_cast<int64_t>(static_cast<const int8_t*>(arr_data)[0]);
                case GGUF_TYPE_UINT16: return static_cast<int64_t>(static_cast<const uint16_t*>(arr_data)[0]);
                case GGUF_TYPE_INT16:  return static_cast<int64_t>(static_cast<const int16_t*>(arr_data)[0]);
                case GGUF_TYPE_UINT32: return static_cast<int64_t>(static_cast<const uint32_t*>(arr_data)[0]);
                case GGUF_TYPE_INT32:  return static_cast<int64_t>(static_cast<const int32_t*>(arr_data)[0]);
                case GGUF_TYPE_UINT64: return static_cast<int64_t>(static_cast<const uint64_t*>(arr_data)[0]);
                case GGUF_TYPE_INT64:  return static_cast<int64_t>(static_cast<const int64_t*>(arr_data)[0]);
                default: break;
            }
        } break;
        default:
            break;
    }

    return -1;
}

void Tokenizer::extract_metadata() {
    const int64_t model_key = gguf_find_key(ctx, "tokenizer.ggml.model");
    if (model_key >= 0) {
        model_type = gguf_get_val_str(ctx, model_key);
        cout << "[FloatLLM] Tokenizer Architecture: " << model_type << "\n";
    }

    bos_token_id = static_cast<int32_t>(read_scalar_integer(ctx, "tokenizer.ggml.bos_token_id"));
    eos_token_id = static_cast<int32_t>(read_scalar_integer(ctx, "tokenizer.ggml.eos_token_id"));

    cout << "[FloatLLM] BOS ID: " << bos_token_id << " | EOS ID: " << eos_token_id << "\n";

    const int64_t tokens_key = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    if (tokens_key < 0) {
        throw std::runtime_error("tokenizer.ggml.tokens not found in GGUF");
    }

    const size_t n_tokens = gguf_get_arr_n(ctx, tokens_key);
    vocab.reserve(n_tokens);
    for (size_t i = 0; i < n_tokens; ++i) {
        const char* raw = gguf_get_arr_str(ctx, tokens_key, i);
        string token = raw ? raw : "";
        vocab.push_back(token);
        token_to_id[token] = static_cast<int32_t>(vocab.size() - 1);
    }

    cout << "[FloatLLM] Successfully extracted " << vocab.size()
              << " offline tokens into memory\n";
}

} // namespace floatllm
