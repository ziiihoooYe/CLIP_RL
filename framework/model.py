import torch.nn as nn

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        # config is a dict containing the configuration for the model
        # Model Object should have the following attributes:
        # - config: configuration for the model
        # - model: the callable model object
        self.config = config
        self.model = None
        self.iter_now = None # this is used to track the current iteration during training
        self.stored_img_feat = {}
        self.stored_txt_feat = {}
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward method must be implemented")
    
    def device(self):
        return next(self.model.parameters()).device
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if hasattr(self.model, 'module'):
                return getattr(self.model.module, name)
            return getattr(self.model, name)
    
    def parameters(self):
        return self.model.parameters()

# There are three types of encoders: 
#   - ImageEncoder (encode image)
#   - TextEncoder (encode text) 
#   - ImageTextEncoder (encode both image and text)

class ImageEncoder(Model):
    def __init__(self, config):
        super().__init__(config)
    
    # encode_image method could call forward method of the model
    def encode_image(self, images):
        raise NotImplementedError("encode_image method must be implemented")

class TextEncoder(Model):
    def __init__(self, config):
        super().__init__(config)        
    
    # encode_text method could call forward method of the model
    def encode_text(self, texts):
        raise NotImplementedError("encode_text method must be implemented")

# ImageTextEncoder is a model that can encode both image and text
# emsemble image encoder and text encoder in one model
class ImageTextEncoder(Model):
    def __init__(self, config):
        super().__init__(config)
    
    def encode_image(self, images):
        raise NotImplementedError("encode_image method must be implemented")
    
    def encode_text(self, texts):
        raise NotImplementedError("encode_text method must be implemented")