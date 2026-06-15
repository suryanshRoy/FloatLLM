#pragma once

#include <string>
#include <utility>

namespace floatllm {

class TerminalUI {
public:
    // pre-run safety check will exit() if things are bad
    static double check_threshold(double current_ram_mb, double crash_threshold_mb, double model_size_mb,
                                  double total_storage_gb, double free_storage_gb, double used_ram_mb,
                                  double total_ram_mb, double override_storage_gb, const char* session_id,
                                  int temp_chat, double ram_limit_gb, double ram_buffer);

    // monitors system RAM and kills process if usage > 95%
    static void abort_if_overloaded();

    // get system ram stats
    static std::pair<double, double> get_ram_stats_mb();

    // get disk space
    static std::pair<double, double> get_storage_stats_gb();

    // simple file size helper
    static size_t file_size_bytes(const std::string& path);
};

} // namespace floatllm
