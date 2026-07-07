#include "FloatLLM.h"
#include <chrono>

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

        cout << PURPLE("Hardware Router engaged: Backend -> [" << selected_backend << "]\n");

        // grab system stats for the safety checks
        const auto [total_ram_mb, free_ram_mb] = floatllm::TerminalUI::get_ram_stats_mb(opts.override_ram_mb);
        const auto [total_storage_gb, free_storage_gb] = floatllm::TerminalUI::get_storage_stats_gb();
        const double model_size_mb = static_cast<double>(floatllm::TerminalUI::file_size_bytes(opts.model_path)) / (1024.0 * 1024.0);

        // pre-flight safety checks
        const double calculated_limit = floatllm::TerminalUI::check_threshold(
            free_ram_mb, opts.crash_threshold_mb, model_size_mb, total_storage_gb,
            free_storage_gb, -1.0, total_ram_mb, opts.override_storage_gb,
            opts.session_id.c_str(), opts.temp_chat ? 1 : 0, opts.ram_limit_gb, opts.ram_buffer, opts.quantize_memory);

        // create the engine and tokenizer
        floatllm::ComputeEngine engine;
        floatllm::Tokenizer tokenizer(opts.model_path);

        // loader takes engine reference - no globals needed
        floatllm::Loader loader(opts.model_path, calculated_limit, engine);

        // parse model, init engine, load tensors
        const auto tensor_map = loader.parse_gguf_metadata();

        // check if a output head exists in GGUF metadata if neither is present, the model uses tied embeddings
        bool has_output_head = false;
        for (const auto& t : tensor_map) {
            if (t.name == "output.weight" || t.name == "lm_head.weight") {
                has_output_head = true;
                break;
            }
        }
        engine.set_tied_embd(!has_output_head);
        if (!has_output_head) {
            cout << YELLOW("No dedicated output head found. Enabling tied-embeddings mode.\n");
        }

        engine.init(selected_backend, static_cast<int>(tensor_map.size()));
        loader.set_allowed_ram_mb(calculated_limit);
        loader.build_dynamic_chunks(tensor_map);
        loader.stream_all_chunks();

        cout << "Engine successfully mapped. Handing to AI...\n";
        cout << "--------------------------------------------------------------------------------\n";
        cout << "\nUser: " << opts.prompt << "\n";

        // tokenize and start generating
        std::vector<int32_t> token_ids = tokenizer.encode(opts.prompt);
        const int max_tokens_to_generate = opts.max_tokens;

        int tokens_generated = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < max_tokens_to_generate; ++step) {
            // Safety Guardian: Check for system-wide RAM overload before every single compute step
            floatllm::TerminalUI::if_overload(opts.override_ram_mb);

            std::vector<int32_t> working_tokens = token_ids;
            int32_t next_token_id = engine.forward_pass(working_tokens.data(), static_cast<int>(working_tokens.size()));
            if (next_token_id == tokenizer.eos_id()) {
                break;
            }

            string word = tokenizer.decode({next_token_id});
            cout << "\033[92m" << word << "\033[0m ";
            cout.flush();
            token_ids.push_back(next_token_id);
            tokens_generated++;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> totalT = end_time - start_time;
        double total_time_sec = totalT.count();
        double tokens_per_sec = total_time_sec > 0 ? (tokens_generated / total_time_sec) : 0.0;

        cout << "\n\n";
        cout << GREEN("Generated " << tokens_generated << " tokens in output!") << "\n";
        cout << PURPLE("Performance: " << std::fixed << std::setprecision(2) << tokens_per_sec << " token/s (Time: " << total_time_sec << "s)") << "\n";
        cout << " --------------------------------------------------------------------------------\n";
        engine.shutdown();
        cout << "Closing C++ memory maps...\n";
        return 0;
    } catch (const std::exception& e) {
        cerr << RED("ERROR: ") << e.what() << "\n";
        return 1;
    }
}