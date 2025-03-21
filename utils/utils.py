import os
import torch

def get_gpu_device(device):
    import os
    if device is not None and torch.cuda.is_available():
        return f"cuda:{device}"
    else:
        return "cpu"