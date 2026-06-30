#include "FloatUI.h"
#include "FloatLLM.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif
#include <sys/stat.h>
#include <fcntl.h>

namespace floatllm {

double TerminalUI::check_threshold(double current_ram_mb, double crash_threshold_mb, double model_size_mb,
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
    cout << PURPLE("--- User Execution Blueprint ---\n");
    cout << PURPLE("Session ID           : [" << session << "]\n");
    cout << PURPLE("Context Saving       : " << (temp_chat ? "Temporary (Delete on Exit)" : "PERSISTENT (Saved to SSD)") << "\n\n");

    return allowed_ram_mb;
}

void TerminalUI::abort_if_overloaded(double override_ram_mb) {
    auto stats = get_ram_stats_mb(override_ram_mb);
    double total_ram = stats.first;
    double free_ram = stats.second;

    if (total_ram <= 0) return;

    double used_ram = total_ram - free_ram;
    double usage_percent = (used_ram / total_ram) * 100.0;

    if (usage_percent > 95.0) {
        cerr << RED("\n\n!!! CRITICAL SYSTEM OVERLOAD !!!") << endl;
        cerr << RED("System RAM usage is at " << std::fixed << std::setprecision(2) << usage_percent << "%") << endl;
        cerr << RED("Aborting FloatLLM execution to prevent system crash.") << endl;
        cerr << RED("TERMINATED DUE TO OVERLOAD") << endl;
        
        // Use quick_exit or exit to ensure immediate termination and OS cleanup of buffers
        std::exit(1);
    }
}

std::pair<double, double> TerminalUI::get_ram_stats_mb(double override_ram_mb) {
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
    return {0.0, 0.0};

#elif defined(__linux__) || ((defined(__unix__)) && !defined(__APPLE__))
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open()) {
        std::string line;
        long long memTotalKB = 0, memAvailableKB = 0;
        while (std::getline(meminfo, line)) {
            std::istringstream iss(line);
            std::string key;
            long long value;
            std::string unit;
            if (iss >> key >> value >> unit) {
                if (key == "MemTotal:") {
                    memTotalKB = value;
                }
                else if (key == "MemAvailable:") {
                    memAvailableKB = value;
                }
            }
            if (memTotalKB > 0 && memAvailableKB > 0) break;
        }
        if (memTotalKB > 0 && memAvailableKB > 0) {
            double total_mb = static_cast<double>(memTotalKB) / 1024.0;
            double free_mb = static_cast<double>(memAvailableKB) / 1024.0;
            return {total_mb, free_mb};
        }
    }

    const long page_size = sysconf(_SC_PAGESIZE);
    const long phys_pages = sysconf(_SC_PHYS_PAGES);
    const double total = (page_size > 0 && phys_pages > 0) ? (static_cast<double>(page_size) * static_cast<double>(phys_pages)) / (1024.0 * 1024.0) : 0.0;

    if (override_ram_mb >= 0.0) {
        return {total, override_ram_mb};
    }

    const double free = total * 0.5; // rough estimate when we cant get
    return {total, free};

#else
    return {0.0, 0.0};

#endif
}

std::pair<double, double> TerminalUI::get_storage_stats_gb() {
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

size_t TerminalUI::file_size_bytes(const string& path) {
    struct stat st {};
    if (stat(path.c_str(), &st) != 0) {
        throw std::runtime_error("failed to stat model file: " + path);
    }
    return static_cast<size_t>(st.st_size);
}

} // namespace floatllm
