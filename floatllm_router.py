import argparse
import platform
import sys
import logging
import psutil
import shutil
import os
import ctypes

# CLI logging format
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

class ColorFormatter(logging.Formatter):
    """Dynamically applies colors based on log level or injected extra parameters."""
    def format(self, record):
        # Default level-based colors
        color = Colors.WHITE
        if record.levelno >= logging.ERROR:
            color = Colors.RED
        elif record.levelno == logging.WARNING:
            color = Colors.YELLOW

        # Override if a custom color is passed via 'extra' dictionary
        if hasattr(record, 'color'):
            color = record.color

        # Format the base message and wrap it in the resolved color
        base_msg = record.getMessage()
        return f"{color}[FloatLLM] {base_msg}{Colors.RESET}"

# Configure the root logger with the dynamic formatter
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(ColorFormatter())
logger.addHandler(ch)

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
    parser.add_argument("--prompt", type=str, required=True, help="The prompt you want to send to the LLM")

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
    
    from floatllm_loader import FloatLLM_Loader
    loader = FloatLLM_Loader(model_path=args.model_path, allowed_ram_mb=1.0, backend_name=backend)

    calculated_limit = loader.cpp_engine.check_failsafe_threshold(
        free_ram,
        args.crash_threshold,
        actual_model_size_mb,
        total_storage,
        free_storage,
        -1.0,
        total_ram,
        int(args.quantize_on_fly),
        int(args.save_quantized),
        int(args.no_ram_protocol),
        -1.0 if args.override_storage is None else float(args.override_storage),
        args.session_id.encode("utf-8"),
        int(args.temp_chat),
        -1.0 if args.ram_limit is None else float(args.ram_limit),
        float(args.ram_buffer)
    )

    loader.allowed_ram_bytes = int(calculated_limit * (1024 ** 2))

    from floatllm_tokenizer import FloatLLM_Tokenizer
    tokenizer = FloatLLM_Tokenizer(args.model_path)
    token_ids = tokenizer.encode(args.prompt)

    tensor_map = loader.parse_gguf_metadata()
    loader.wake_engine(len(tensor_map))
    loader.build_dynamic_chunks(tensor_map)

    # logging.info("-"*80)
    for chunk in loader.chunks:
        loader.stream_chunk(chunk["id"])

    logging.info("Engine successfully mapped. Handing to AI...\n")
    logging.info("-"*80)

    # --- THE GENERATION LOOP ---
    logging.info(f"\nUser: {args.prompt}")
    sys.stdout.write("[FloatLLM] ")
    sys.stdout.flush()

    max_tokens_to_generate = 60 # Let's generate 60 words for this test

    for step in range(max_tokens_to_generate):
        # Convert our growing list of token IDs into a raw C-array
        c_token_array = (ctypes.c_int32 * len(token_ids))(*token_ids)
        c_num_tokens = ctypes.c_int(len(token_ids))

        next_token_id = loader.cpp_engine.execute_forward_pass(c_token_array, c_num_tokens)

        # Stop if the AI decides the sentence is finished!
        if next_token_id == tokenizer.eos_token_id:
            break

        word = tokenizer.decode([next_token_id])

        # Stream to the terminal without a newline
        sys.stdout.write(f"{Colors.GREEN}{word}{Colors.RESET} ")
        sys.stdout.flush()

        token_ids.append(next_token_id)

    sys.stdout.write("\n\n")
    sys.stdout.flush()
    logging.info("Generated first 60 tokens in output!")
    logging.info("-" * 80)
    
    loader.shutdown_engine()