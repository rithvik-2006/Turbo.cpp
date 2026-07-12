import gguf
import numpy as np

print("Generating dummy GGUF file...")

# Initialize writer
writer = gguf.GGUFWriter("dummy_model.gguf", "llama")

# 1. Add Metadata KV pairs
writer.add_name("Turbo-Dummy-Test-Model")
writer.add_uint32("general.alignment", 32)
writer.add_uint32("llama.context_length", 2048)

# Add tokenizer vocabulary
vocab = ["<unk>", "<s>", "</s>", "h", "e", "l", "o", "w", "r", "d", "\u2581", "!", "he", "ll", "lo", "hel", "wo", "wor", "worl", "world", "hello"]
scores = [0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.2, 0.6, 0.8, 0.5, 0.6, 0.7, 0.9, 1.0]
writer.add_token_list(vocab)
writer.add_token_scores(scores)

# 2. Create some fake tensor data
tensor_q = np.ones((128, 128), dtype=np.float32)
tensor_k = np.zeros((64, 128), dtype=np.float16)

# 3. Register tensors
writer.add_tensor("blk.0.attn_q.weight", tensor_q)
writer.add_tensor("blk.0.attn_k.weight", tensor_k)

# 4. Write binary data to disk
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()

print("Created dummy_model.gguf successfully!")
