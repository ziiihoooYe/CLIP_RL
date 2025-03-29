import torch
import clip
from framework.model import ImageTextEncoder
from utils.utils import get_main_device
 

class CLIP_ViTB32(ImageTextEncoder):
    def __init__(self, config):
        super(CLIP_ViTB32, self).__init__(config)
        self.model, _ = clip.load("ViT-B/32", device=get_main_device(self.config.get("gpu", None)))
        self.model = self.model.float()
    
    def encode_image(self, images):
        embeds = self.model.encode_image(images)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        return embeds

    def encode_text(self, texts): 
        embeds = self.model.encode_text(texts)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        return embeds

    def forward(self, images, texts):
        image_embeds = self.encode_image(images)
        text_embeds  = self.encode_text(texts)
        return image_embeds, text_embeds
        