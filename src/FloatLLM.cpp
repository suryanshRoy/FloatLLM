#include "FloatLLM.h"

int main(int argc, char** argv) {
    try {
        floatllm::CliOptions opts;
        if (!floatllm::parse_args(argc, argv, opts)) {
            return 0;
        }

        // figure out which hardware to use
        string selected_backend = opts.hardware;
        string lower_backend = selected_backend;
        std::transform(lower_backend.begin(), lower_backend.end(), lower_backend.begin(), ::tolower);
        if (lower_backend == "auto") {
            selected_backend = floatllm::ComputeEngine::detect_hardware();
        }

        cout << PURPLE("[FloatLLM] Hardware Router engaged: Backend -> [" << selected_backend << "]\n");

        // grab system stats for the safety checks
        const auto [total_ram_mb, free_ram_mb] = floatllm::ComputeEngine::get_ram_stats_mb();
        const auto [total_storage_gb, free_storage_gb] = floatllm::ComputeEngine::get_storage_stats_gb();
        const double model_size_mb = static_cast<double>(floatllm::ComputeEngine::file_size_bytes(opts.model_path)) / (1024.0 * 1024.0);

        // pre-flight safety checks
        const double calculated_limit = floatllm::ComputeEngine::check_threshold(
            free_ram_mb, opts.crash_threshold_mb, model_size_mb, total_storage_gb,
            free_storage_gb, -1.0, total_ram_mb, opts.override_storage_gb,
            opts.session_id.c_str(), opts.temp_chat ? 1 : 0, opts.ram_limit_gb, opts.ram_buffer);

        // create the engine and tokenizer
        floatllm::ComputeEngine engine;
        floatllm::Tokenizer tokenizer(opts.model_path);

        // loader takes engine reference - no globals needed
        floatllm::Loader loader(opts.model_path, calculated_limit, engine);

        // parse model, init engine, load tensors
        const auto tensor_map = loader.parse_gguf_metadata();
        engine.init(selected_backend, static_cast<int>(tensor_map.size()), opts.slack_buffer_mb);

        floatllm::ModelHParams hparams = loader.parse_hparams();
        hparams.n_ctx = std::min(opts.context_length, hparams.n_ctx_train > 0 ? hparams.n_ctx_train : opts.context_length);
        engine.set_hparams(hparams);

        loader.set_allowed_ram_mb(calculated_limit);
        loader.build_dynamic_chunks(tensor_map);
        loader.stream_all_chunks();

        // one-time weight upload (GPU) / zero-copy bind (CPU) + KV cache allocation
        if (!engine.finalize_weights()) {
            throw std::runtime_error("failed to finalize weights / allocate KV cache");
        }

        cout << "[FloatLLM] Engine successfully mapped. Handing to AI...\n";
        cout << "[FloatLLM] --------------------------------------------------------------------------------\n";
        cout << "[FloatLLM] \nUser: " << opts.prompt << "\n";
        cout << "[FloatLLM] ";

        // tokenize and start generating (user-tunable via --max-tokens)
        std::vector<int32_t> token_ids = tokenizer.encode(opts.prompt);
        const int max_tokens_to_generate = opts.max_tokens;

        // prefill: run the whole prompt through once, filling the KV cache
        int n_past = 0;
        int32_t next_token_id = engine.forward_pass(token_ids.data(), static_cast<int>(token_ids.size()), n_past);
        n_past += static_cast<int>(token_ids.size());

        int generated = 0;
        for (int step = 0; step < max_tokens_to_generate; ++step) {
            if (next_token_id < 0 || next_token_id == tokenizer.eos_id()) {
                break;
            }

            string word = tokenizer.decode({next_token_id});
            cout << "\033[92m" << word << "\033[0m ";
            cout.flush();
            ++generated;

            // decode: feed only the newest token, the KV cache remembers the rest
            next_token_id = engine.forward_pass(&next_token_id, 1, n_past);
            ++n_past;
        }

        cout << "\n\n";
        cout << GREEN("[FloatLLM] Generated ") << generated << GREEN(" tokens in output!") << "\n";
        cout << "[FloatLLM] --------------------------------------------------------------------------------\n";
        engine.shutdown();
        cout << "[FloatLLM] Closing C++ memory maps...\n";
        return 0;
    } catch (const std::exception& e) {
        cerr << RED("[FloatLLM] ERROR: ") << e.what() << "\n";
        return 1;
    }
}