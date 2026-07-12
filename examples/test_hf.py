from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print("system:", tokenizer.encode("<|system|>"))
print("user:", tokenizer.encode("<|user|>"))
print("assistant:", tokenizer.encode("<|assistant|>"))
