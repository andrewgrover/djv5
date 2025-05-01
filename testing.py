import torch

print("CUDA available? ", torch.cuda.is_available())        # → False (expected)
print("MPS available?  ", torch.backends.mps.is_available())  # → True on Apple Silicon
