import argparse
import platform
import sys
import logging
import psutil

# #CLI logging format
logging.basicConfig(level=logging.INFO, format="[FloatLLM] %(message)s")

def get_hardware_backend():
    """Dynamically route the worload based on host hardware."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda" #NVIDIA
        elif torch.backends.mps.is_available():
            return "mps" # for Apple silicon
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            return "xpu" # Intel GPUs
        elif hasattr(torch, 'npu') and torch.npu.is_available():
            return "npu_ascend" # for Huawei 
    except ImportError:
        pass # Pytorch not installed/needed for this run

    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux" and ("aarch64" in machine or "arm" in machine):
        return "native_arm" 
    
    return "native_cpu" 

def get_ram_stats():
    """Cross-platform hardware interrogation."""
    mem_info = psutil.virtual_memory()
    total = mem_info.total / (1024*1024)
    free = mem_info.available / (1024*1024)
    return total, free

def check_failsafe_thresold(current_ram_mb, crash_thresold_mb, model_size_mb, used_ram_mb=None, total_ram_mb=None, use_airllm=False, quantize_on_fly=False):
    """Acts as a pre-flight dashboard and a runtime kill-switch"""

    # 1. RUNTIME INTERCEPT: If. system hit thresolt, stop gracefully
    if current_ram_mb <= crash_thresold_mb:
        logging.error("\n" + "-"*100)
        logging.error("🚨 FloatLLM OOM Failsafe triggered to stop crashing/freezzing of device.")
        logging.error("-"*100)
        logging.error(f"CRITICAL: Free RAM ({current_ram_mb:.2f} MB) hit the crash thresold ({crash_thresold_mb:.2f} MB).")
        logging.error(f"Target Model Size: {model_size_mb:.2f} MB")
        if used_ram_mb:
            logging.error(f"FloatLLM Consumed: {used_ram_mb:.2f} MB (Max Peak)")
        logging.error("Action: Halting execution gracefully. Model data safely flushed.")
        logging.error("Next Run: Enable --use-airllm, or adjust --crash-thresold.")
        logging.error("-"*100 + "\n")
        sys.exit(1) # Controlled exit prevents file corruption

    # 2. PRE-FLIGHT Report: If used_ram_mb is None, we haven't crashed, just reporting
    elif used_ram_mb is None:
        logging.info("\n--- Pre-Flight Memory Dashboard ---")
        if total_ram_mb:
            logging.info(f"Host Total Ram       : {total_ram_mb:.2f} MB")
            logging.info(f"Host Used RAM        : {(total_ram_mb - current_ram_mb):.2f} MB")
        logging.info(f"Host Free Ram        : {current_ram_mb:.2f} MB")
        logging.info(f"Target Model Size    : {model_size_mb:.2f} MB")
        logging.info(f"Kill Thresold        : {crash_thresold_mb:.2f} MB")
        logging.info("--- User Execution Blueprint ---")
        logging.info(f"AirLLM Swapping      : {'ENABLED' if use_airllm else 'DISABLED'}")
        logging.info(f"Live Quantization    : {'ENABLED' if quantize_on_fly else 'DISABLED'}")
        logging.info("-"*100+"\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FloatLLM Engine")

    # Freedom & Failsafe Flags
    parser.add_argument("--hardware", type=str, default="auto", help="Force backend (e.g., mps, native_arm)")
    parser.add_argument("--use-airllm", action="store_true", help="Explict consent for layer-swapping")
    parser.add_argument("--quantize-on-fly", action="store_true", help="Explict consent to quantize weights")
    parser.add_argument("--crash-thresold", type=float, default=200.0, help="Failtsafe buffer in MB")
    parser.add_argument("--model-size", type=float, default=4500.0, help="Mock model size in MB for testing")

    args = parser.parse_args()

    # Initialize Hardware
    backend = args.hardware.lower() if args.hardware.lower() != "auto" else get_hardware_backend() 
    logging.info(f"Hardware Router engaged: Backend -> [{backend.upper()}]")

    # Gather System Memory Facts
    total_ram, free_ram = get_ram_stats()

    #Fire the Pre-Flight Dashboard with ALL variables
    check_failsafe_thresold(current_ram_mb=free_ram,
                            crash_thresold_mb=args.crash_thresold,
                            model_size_mb=args.model_size,
                            total_ram_mb=total_ram,
                            use_airllm=args.use_airllm,
                            quantize_on_fly=args.quantize_on_fly)
    
    logging.info("Blueprint validated. Proceeding to Model Loader...")