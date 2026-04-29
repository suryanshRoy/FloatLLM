import os
import mmap
import logging
import ctypes
import platform

logging.basicConfig(level=logging.INFO, format="[FloatLLM] %(message)s")

class FloatLLM_Loader:
    def __init__(self, model_path, allowed_ram_mb, backend_name):
        """Initializes the loader with stric RAM boundaries from the hardware router."""
        self.model_path = model_path
        self.allowed_ram_bytes = int(allowed_ram_mb * (1024 ** 2))
        self.backend_name = backend_name

        if not os.path.exists(self.model_path):
            logging.error(f"CRITICAL: Model file not found at {self.model_path}")
            raise FileNotFoundError
        
        self.file_size = os.path.getsize(self.model_path)
        self.chunks = []

        # --- C++ COMPUTE BRIDGE WAKE-UP ---
        # 1. Locate and load the compiled file based on user's OS 
        system_os = platform.system().lower()
        if system_os == "windows":
            ext = ".dll"
        elif system_os == "darwin":
            ext = ".dylib"
        else:
            ext = ".so"
        
        lib_path = os.path.abspath(f"floatllm_compute{ext}")

        if not os.path.exists(lib_path):
            logging.error(f"Compiled C++ backend not found!\nPlease run the compilation command for your OS.")
            raise FileNotFoundError

        self.cpp_engine = ctypes.CDLL(lib_path)

        # 2. Exact C++ argument so Python doesn't crash the memory
        self.cpp_engine.init_compute_engine.argtypes = [ctypes.c_char_p, ctypes.c_int]
        # Send name, true type, pointer, and all 4 dimensions
        self.cpp_engine.execute_tensor_chunk.argtypes = [
            ctypes.c_char_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64,  ctypes.c_int]

        # Graph execution signal 
        self.cpp_engine.execute_graph_test.argtypes = []

    def wake_engine(self, total_tensors):
        """Fires the wake-up signal safely after metadata is parsed"""
        self.cpp_engine.init_compute_engine(self.backend_name.encode('utf-8'), total_tensors)

    def parse_gguf_metadata(self):
        """Scans the GGUF file header to find exact tensor byte offsets."""
        import gguf

        logging.info(f"Scanning GGUF metadata for building {self.model_path}...")

        reader = gguf.GGUFReader(self.model_path)

        tensors = []
        for tensor in reader.tensors:
            shape = list(tensor.shape)

            while len(shape) < 4:
                shape.append(1)

            tensors.append({
                "name": tensor.name,
                "type":  int(tensor.tensor_type), #Quantization type (e.g. F16, F32, Q4)
                "offset": tensor.data_offset, 
                "size": tensor.n_bytes,
                "shape":  shape
            })
        logging.info(f"Discovered {len(tensors)} individual tensors in the model architecture.")
        return tensors
    
    def build_dynamic_chunks(self, tensors):
        """Slices the tensor map into execution blocks based on floatllm_router.py safety limits."""

        logging.info(f"Chucking Engine Active. Max RAM per block: {self.allowed_ram_bytes/(1024**2):.2f} MB")

        current_chunk = []
        current_chunk_size = 0

        for tensor in tensors:
            # If the next tensor breaks the RAM limit, seal the current chunk
            if current_chunk_size + tensor["size"] > self.allowed_ram_bytes:
                if not current_chunk:
                    logging.error(f"CRITICAL: Tensor {tensor['name']} ({tensor['size']/(1024**2):.2f} MB exceeds allowed RAM.)")
                    raise MemoryError
                self.chunks.append({
                    "id": len(self.chunks) +1,
                    "tensors": current_chunk,
                    "total_size_mb": current_chunk_size / (1024**2)
                })
                current_chunk = []
                current_chunk_size = 0
            current_chunk.append(tensor)
            current_chunk_size += tensor["size"]
        
        # Seal the final chunk
        if current_chunk:
            self.chunks.append({
                "id": len(self.chunks)+1,
                "tensors": current_chunk,
                "total_size_mb": current_chunk_size / (1024**2)
            })
        logging.info(f"Model succesfuly sliced into {len(self.chunks)} dynamic blocks")

    def stream_chunk(self, chunk_id):
        """Creates a zero-copy mmap bridge to the SSD for a specific block of weights."""
        target_chunk = next((c for c in self.chunks if c["id"] == chunk_id), None)
        if not target_chunk:
            return None
        
        logging.info(f"Streaming Chunk {chunk_id}/{len(self.chunks)} -> RAM [{target_chunk['total_size_mb']:.2f} MB...]")
        with open(self.model_path, "rb") as f:
            mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            c_obj = ctypes.py_object(mmapped_file)
            address = ctypes.c_void_p()
            length = ctypes.c_ssize_t()
            ctypes.pythonapi.PyObject_AsReadBuffer(c_obj, ctypes.byref(address), ctypes.byref(length))
            base_ptr = address.value

            for tensor in target_chunk["tensors"]:
                # Calculate the exact RAM address for this specific tensor
                tensor_ptr = base_ptr + tensor["offset"]
                tensor_name_bytes = tensor["name"].encode("utf-8")
                ne0, ne1, ne2, ne3 = tensor["shape"]

                self.cpp_engine.execute_tensor_chunk(
                    tensor_name_bytes, ctypes.c_int(tensor["type"]),
                    ctypes.c_void_p(tensor_ptr), ctypes.c_int64(ne0),
                    ctypes.c_int64(ne1), ctypes.c_int64(ne2), ctypes.c_int64(ne3),
                    ctypes.c_int(chunk_id))
            
            mmapped_file.close()
        
        logging.info(f"Chunk {chunk_id} Executed. Hardware link closed.")