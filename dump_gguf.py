import gguf
reader = gguf.GGUFReader('Llama-3.2-1B-Instruct-Q8_0.gguf')
for tensor in reader.tensors:
    if "output" in tensor.name or "embd" in tensor.name:
        print(tensor.name, tensor.shape, tensor.tensor_type)
