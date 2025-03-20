import torch
import clip
from framework.model import ImageTextEncoder
from utils.utils import get_gpu_device
    
class CLIP_ViTB32(ImageTextEncoder):
    def __init__(self, config):
        self.device = get_gpu_device(config.get("gpu", None))
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model = self.model.to(self.device)
    
    def encode_image(self, images):
        if isinstance(images, list):
            processed_images = []
            for image in images:
                if isinstance(image, torch.Tensor):
                    processed_images.append(image)
                else:
                    processed_images.append(self.preprocess(image))
            images = torch.stack(processed_images)
        else:
            images = self.preprocess(images).unsqueeze(0)
        images = images.to(self.device)
        return self.model.encode_image(images)
    
    def encode_text(self, texts):
        if isinstance(texts, list):
            text_tokens = clip.tokenize(texts)
        else:
            text_tokens = clip.tokenize([texts])
        text_tokens = text_tokens.to(self.device)
        return self.model.encode_text(text_tokens)
