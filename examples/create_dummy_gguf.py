import gguf
import numpy as np

print("Generating dummy GGUF file...")

# Initialize writer
writer = gguf.GGUFWriter("dummy_model.gguf", "llama")

# 1. Add Metadata KV pairs
writer.add_name("Turbo-Dummy-Test-Model")
writer.add_uint32("general.alignment", 32)
writer.add_uint32("llama.context_length", 2048)

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
