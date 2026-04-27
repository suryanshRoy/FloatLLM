#include <iostream>
#include <cstdint>

// Using extern "C" to disable C++ name mangling so python ctypes can find them

extern "C" {
    // 1. Catches the hardware string (e.g, "mps", "vulkan") from router

    void init_compute_engine(const char* backend_name) {
        std::cout << "[FloatLLM C++] Engine waking up. Hardware targeted:[" << backend_name << "]" <<std::endl;
    }
    // 2. The execution socket
    void execute_tensor_chunk(void* raw_memory_pointer, size_t byte_size, int chunk_id) {
        double size_mb = static_cast<double>(byte_size) / (1024.0*1024.0);

        std::cout << "[FloatLLM C++] Caught Chunk" << chunk_id
                  << "| Memory Pointer: " << raw_memory_pointer
                  << "| Size: " << size_mb << " MB" << std::endl;
    }
}