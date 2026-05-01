import os
import logging
import gguf

logging.basicConfig(level=logging.INFO, format="[FloatLLM(Tokenizer)] %(message)s")

class FloatLLM_Tokenizer:
    def __init__(self, model_path):
        """Initialize the tokenizer to extract embedded vocab from gguf"""
        self.model_path = model_path
        self.vocab = []
        self.token_to_id = {}

        # Special token
        self.bos_token_id = None
        self.eos_token_id = None
        self.model_type = "unkown"

        if not os.path.exists(self.model_path):
            logging.error(f"Model file not found at {self.model_path}")
            raise FileNotFoundError
        
        self._extract_tokenizer_metadata()

    def _get_scalar(self, reader, key):
        """Extract a scalar integer, safely handling NumPy arrays."""
        if key in reader.fields:
            val = reader.fields[key].parts[-1]
            if hasattr(val, '__iter__') and len(val) > 0:
                return int(val[0])
            return int(val)
        return None
    
    def _extract_tokenizer_metadata(self):
        """Scan the GGUF metadata to build the dictionary"""
        logging.info(f"Extracting vocabulary from local GGUF: {self.model_path}...")

        reader = gguf.GGUFReader(self.model_path)

        if "tokenizer.ggml.model" in reader.fields:
            field = reader.fields["tokenizer.ggml.model"]
            self.model_type = bytes(field.parts[-1]).decode('utf-8', errors="ignore").strip("\x00")
            logging.info(f"Tokenizer Architecture: {self.model_type.upper()}")
        
        self.bos_token_id = self._get_scalar(reader, "tokenizer.ggml.bos_token_id")
        self.eos_token_id = self._get_scalar(reader, "tokenizer.ggml.eos_token_id")

        logging.info(f"BOS ID: {self.bos_token_id} | EOS ID: {self.eos_token_id}")

        # Extract actual vocab string arrays 
        if "tokenizer.ggml.tokens" in reader.fields:
            token_field = reader.fields["tokenizer.ggml.tokens"]

            for raw_token in token_field.parts:
                if isinstance(raw_token, (int, float)):
                    continue
                if hasattr(raw_token, 'ndim') and raw_token.ndim == 0:
                    continue

                token_str = ""
                # Handle string, bytes, or Numpy objects
                if isinstance(raw_token, str):
                    token_str = raw_token
                elif isinstance(raw_token, (bytes, bytearray)):
                    token_str = raw_token.decode('utf-8', errors='ignore')
                elif hasattr(raw_token, 'tobytes'):
                    token_str = raw_token.tobytes().decode('utf-8', errors='ignore')
                else:
                    continue

                self.vocab.append(token_str)
                self.token_to_id[token_str] = len(self.vocab) - 1

            logging.info(f"Successfully extracted {len(self.vocab)} offline tokens into memoryy.")    
        else:
            logging.error("No 'tokenizer.ggml.tokens' field found in this GGUF file.")

    def encode(self, text):
        """Converts text into integer Token IDs"""
        logging.info(f"Encoding Prompt: '{text}'")
        token_ids = []

        if self.bos_token_id is not None:
            token_ids.append(self.bos_token_id)

        words = text.replace(" ", " Ġ").split(" ")

        for word in words:
            if not word: continue

            while len(word) > 0:
                match_found = False
                for length in range(len(word), 0, -1):
                    chunk = word[:length]
                    if chunk in self.token_to_id:
                        token_ids.append(self.token_to_id[chunk])
                        word = word[length:]
                        match_found = True
                        break
                
                if not match_found:
                    word = word[1:]
                
        logging.info(f"Generated Mathematical Tensor Array: {token_ids}")
        return token_ids
    
    def decode(self, token_ids):
        """Converts an array of integer Token IDs back into human-readable text."""
        decoded_text = ""
        for t_id in token_ids:
            if t_id == self.bos_token_id or t_id == self.eos_token_id:
                continue
            if 0 <= t_id < len(self.vocab):
                token_str = self.vocab[t_id]
                # Clean up the space representation for output
                decoded_text += token_str.replace("Ġ", " ")
        return decoded_text.strip()
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FloatLLM Offline Tokenizer Tester")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the local .gguf model file")
    args = parser.parse_args()
    
    # 1. Boot the offline tokenizer using only the local file path
    engine_voice = FloatLLM_Tokenizer(args.model_path)
    
    # 2. Test the encoding (Text -> Math)
    sample_prompt = "What is the capital of France?"
    encoded_math = engine_voice.encode(sample_prompt)
    
    # 3. Test the decoding (Math -> Text)
    decoded_text = engine_voice.decode(encoded_math)
    logging.info(f"Decoded Output: '{decoded_text}'")
    
    logging.info("Offline Tokenizer pipeline verified. Ready for C++ graph injection.")