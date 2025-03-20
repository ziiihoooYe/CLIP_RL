import torch
import clip
from framework.model import ImageTextEncoder

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_gpu_device(device):
    import os
    if device is not None and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        return "cuda"
    else:
        return "cpu"
    
class CLIP_ViTB32(ImageTextEncoder):
    def __init__(self, config):
        self.device = get_gpu_device(config.get("gpu", None))
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model = self.model.eval()
        self.model = self.model.to(self.device)
    
    def encode_image(self, image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        return self.model.encode_image(image)
    
    def encode_text(self, text):
        text_tokens = clip.tokenize([text]).to(self.device)
        return self.model.encode_text(text_tokens)


