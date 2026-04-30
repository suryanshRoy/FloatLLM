import argparse
import platform
import sys
import logging
import psutil
import shutil
import os

# CLI logging format
logging.basicConfig(level=logging.INFO, format="[FloatLLM] %(message)s")

def get_hardware_backend():
    """Dynamically route the workload based on host hardware."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda" # NVIDIA
        elif torch.backends.mps.is_available():
            return "mps" # for Apple silicon
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            return "xpu" # Intel GPUs
        elif hasattr(torch, 'npu') and torch.npu.is_available():
            return "npu_ascend" # for Huawei 
        elif hasattr(torch, 'is_vulkan_available') and torch.is_vulkan_available():
            return "vulkan"
    except ImportError:
        pass # Pytorch not installed/needed for this run

    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux":
        if shutil.which('vulkaninfo') or os.path.exists("//system/lib64/libvulkan.so"):
            return "vulkan_kompute" # GPU backend for Linux/Android
        elif ("aarch64" in machine or "arm" in machine):
            return "native_arm"
    
    return "cpu" 

def get_ram_stats():
    """Cross-platform hardware interrogation."""
    mem_info = psutil.virtual_memory()
    total = mem_info.total / (1024*1024)
    free = mem_info.available / (1024*1024)
    return total, free

def get_storage_stats():
    """Interrogates physical SSD/Hard Drives storage in Gigabytes."""
    real_path = os.path.expanduser("~")
    total_bytes, used_bytes, free_bytes = shutil.disk_usage(real_path)
    total_gb = total_bytes / (1024**3)
    free_gb = free_bytes / (1024**3) # Default physical free space
    return total_gb, free_gb

def check_failsafe_threshold(current_ram_mb, crash_threshold_mb, model_size_mb,
                            total_storage_gb=None, free_storage_gb=None, used_ram_mb=None,
                            total_ram_mb=None, quantize_on_fly=False, save_quantized=False,
                            no_ram_protocol=False, override_storage=None, session_id='default', temp_chat=False,
                            ram_limit=None, ram_buffer=0.20):
    """Monitors RAM limits, SSD limits, and provide the emergency escape menu."""

    model_size_gb = model_size_mb / 1024
    trusted_free_gb = free_storage_gb

    # 1. Storage Override
    if override_storage is not None:
        trusted_free_gb = override_storage
        logging.warning(f"\n Overriding UNIX limits. Trusting your input of {trusted_free_gb:.2f} GB.")
        if total_storage_gb and trusted_free_gb > total_storage_gb:
            logging.error(f"CRITICAL: Override ({trusted_free_gb} GB) exceeds total disk size ({total_storage_gb:.2f} GB). Halting.")
            sys.exit(1)

    # 2. DARWIN WARNING: For macOS device due to Purgeable space
    elif platform.system().lower() == 'darwin' and free_storage_gb and model_size_gb > free_storage_gb:
        logging.warning(f"\n⚠️ UNIX sees {free_storage_gb:.2f} GB. Model needs {model_size_gb:.2f} GB.")
        logging.warning("macOS hides Purgeable space. If you have enough space in System Settings, run with: [--override-storage YOUR_GB]")


    # 3. STORAGE INTERCEPT: Check if SSD can hold the model
    if trusted_free_gb and model_size_gb > trusted_free_gb:
        logging.error("\n" + "-"*80) 
        logging.error("🚨 FloatLLM STORAGE FAILSAFE TRIGGERED")
        logging.error(f"CRITICAL: Model requires {model_size_gb:.2f} GB, but only {trusted_free_gb:.2f} GB is free.")
        logging.error("Action: Halting to prevent storage corruption.")
        logging.error("-"*80 + "\n")
        sys.exit(1)

    # 4. RUNTIME INTERCEPT: If system hit threshold, stop gracefully
    if current_ram_mb <= crash_threshold_mb:
        logging.error("\n" + "-"*80)
        logging.error("🚨 FloatLLM OOM Failsafe triggered to stop crashing/freezing of device.")
        logging.error("-"*80)
        logging.error(f"CRITICAL: Free RAM ({current_ram_mb:.2f} MB) hit the crash threshold ({crash_threshold_mb:.2f} MB).")
        logging.error(f"Target Model Size: {model_size_mb:.2f} MB")
        if used_ram_mb:
            logging.error(f"FloatLLM Consumed: {used_ram_mb:.2f} MB (Max Peak)")
        logging.error("Action: Halting execution gracefully. Model data safely flushed.")
        logging.error("Adjust [--crash-threshold] or increase [--ram-limit] for more safety.") 
        logging.error("For extreme offload: Enable [--no-ram-protocol] to dump KV Cache & Hidden States to SSD.")
        logging.error("Or Compression: Enable [--quantize-on-fly] to compress weights in memory.")
        logging.error("Or Quantize the model permanently using --save-quantized to run the saved quantize model.")
        logging.error("-"*80 + "\n")
        sys.exit(1)

    # 5. PRE-FLIGHT Report
    elif used_ram_mb is None:
        safe_ram_mb = (current_ram_mb * (1.0 - ram_buffer)) - crash_threshold_mb
        safe_ram_mb = max(1.0, safe_ram_mb) 
        if ram_limit:
            ram_limit_mb = ram_limit * 1024
            allowed_ram_mb = min(safe_ram_mb, ram_limit_mb)
        else:
            allowed_ram_mb = safe_ram_mb

        logging.info("\n--- Pre-Flight Memory Dashboard ---")
        if total_ram_mb:
            logging.info(f"Host Total Ram       : {total_ram_mb:.2f} MB")
            logging.info(f"Host Used RAM        : {(total_ram_mb - current_ram_mb):.2f} MB")
        logging.info(f"Host Free Ram        : {current_ram_mb:.2f} MB")
        logging.info(f"Allowed RAM (Chunk)  : {allowed_ram_mb:.2f} MB (Buffer: {ram_buffer*100:.0f}%)")
        if trusted_free_gb:
            logging.info(f"Host Free Storage    : {trusted_free_gb:.2f} GB " + ("(OVERRIDEN)" if override_storage else ""))
        logging.info(f"Target Model Size    : {model_size_mb:.2f} MB")
        logging.info(f"Kill threshold       : {crash_threshold_mb:.2f} MB")
        logging.info("--- User Execution Blueprint ---")
        logging.info(f"Live Quantization    : {'ENABLED' if quantize_on_fly else 'DISABLED'}")
        logging.info(f"AOT Quantization (Save): {'ACTIVE' if save_quantized else 'DISABLED'}")
        logging.info(f"No-RAM Protocol (SSD): {'ACTIVE' if no_ram_protocol else 'DISABLED'}")
        logging.info(f"Session ID           : [{session_id}]")
        logging.info(f"Context Saving       : {'Temporary (Delete on Exit)' if temp_chat else 'PERSISTENT (Saved to SSD)'}")
        logging.info("-"*80+"\n")

        return allowed_ram_mb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloatLLM Engine")

    # Freedom & Failsafe Flags
    parser.add_argument("--hardware", type=str, default="auto", help="Force backend override (e.g., cuda, opencl, vulkan, metal, rocm, oneapi, cpu)")
    parser.add_argument("--quantize-on-fly", action="store_true", help="Explict consent to quantize weights")
    parser.add_argument("--no-ram-protocol", action="store_true", help="Offload all Hidden States and KV Cache to SSD")
    parser.add_argument("--session-id", type=str, default="default_chat", help="Name of the chat to save/resume KV Cache")
    parser.add_argument("--temp-chat", action="store_true", help="Delete the KV Cache on exit")
    parser.add_argument("--override-storage", type=float, default=None, help="Manually override strict UNIX storage limit in GB")
    parser.add_argument("--crash-threshold", type=float, default=200.0, help="Failsafe buffer in MB to stop execution before OOM is triggered.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the .gguf model file")
    parser.add_argument("--save-quantized", action="store_true", help="Save the compressed model to SSD so original can be deleted.")
    parser.add_argument("--ram-limit", type=float, default=None, help="Hard adjustment on RAM usage in GB")
    parser.add_argument("--ram-buffer", type=float, default=0.20, help="Percentage of RAM to reserve for KV cache/OS (default 0.20)")

    args = parser.parse_args()

    # Initialize Hardware
    backend = args.hardware.lower() 
    logging.info(f"Hardware Router engaged: Backend -> [{backend.upper()}]")

    total_ram, free_ram = get_ram_stats()
    total_storage, free_storage = get_storage_stats()

    if not os.path.exists(args.model_path):
        logging.error(f"Model file not found at {args.model_path}")
        sys.exit(1)

    actual_model_size_mb = os.path.getsize(args.model_path) / (1024**2)

    calculated_limit = check_failsafe_threshold(
                            current_ram_mb=free_ram,
                            crash_threshold_mb=args.crash_threshold,
                            model_size_mb=actual_model_size_mb,
                            total_storage_gb=total_storage,
                            free_storage_gb=free_storage,
                            total_ram_mb=total_ram,
                            quantize_on_fly=args.quantize_on_fly,
                            save_quantized=args.save_quantized,
                            no_ram_protocol=args.no_ram_protocol,
                            override_storage=args.override_storage,
                            session_id=args.session_id,
                            temp_chat=args.temp_chat,
                            ram_limit=args.ram_limit,
                            ram_buffer=args.ram_buffer)
    
    logging.info("Blueprint validated. Proceeding to Model Loader...\n")

    from floatllm_loader import FloatLLM_Loader

    loader = FloatLLM_Loader(model_path= args.model_path, allowed_ram_mb=calculated_limit, backend_name = backend)
    tensor_map = loader.parse_gguf_metadata()
    loader.wake_engine(len(tensor_map))
    loader.build_dynamic_chunks(tensor_map)

    logging.info("-"*80)
    for chunk in loader.chunks:
        loader.stream_chunk(chunk["id"])
    logging.info("-"*80)
    logging.info("Engine successfully mapped.")
    
    # Execute Math test once, then shut down gracefully
    loader.cpp_engine.execute_graph_test()
    loader.shutdown_engine()