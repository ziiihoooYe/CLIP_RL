import os
import torch

def get_gpu_device(device):
    import os
    if device is not None and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        return "cuda"
    else:
        return "cpu"