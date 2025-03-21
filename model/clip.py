import torch
import clip
from framework.model import ImageTextEncoder
from utils.utils import get_gpu_device
 

class CLIP_ViTB32(ImageTextEncoder):
    def __init__(self, config):
        self.device = get_gpu_device(config.get("gpu", None))
        self.model, _ = clip.load("ViT-B/32", device=self.device)
        self.model = self.model.to(self.device)
    
    def encode_image(self, images):
        images = torch.stack(images).to(self.device)
        return self.model.encode_image(images)
    
    def encode_text(self, texts):
        text_tokens = torch.cat(texts, dim=0).to(self.device)
        return self.model.encode_text(text_tokens)
