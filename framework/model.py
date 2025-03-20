
class Model:
    def __init__(self, config):
        # config is a dict containing the configuration for the model
        # Model Object should have the following attributes:
        # - config: configuration for the model
        # - model: the callable model object
        self.config = config
        self.model = None
    
    def load(self, model_path):
        raise NotImplementedError("load method must be implemented")

    def save(self, model_path):
        raise NotImplementedError("save method must be implemented")

    def to(self, device):
        self.model.to(device)

# There are three types of encoders: 
#   - ImageEncoder (encode image)
#   - TextEncoder (encode text) 
#   - ImageTextEncoder (encode both image and text)

class ImageEncoder(Model):
    def __init__(self, config):
        super().__init__(config)
    
    def encode_image(self, image):
        raise NotImplementedError("encode_image method must be implemented")

class TextEncoder(Model):
    def __init__(self, config):
        super().__init__(config)        
    
    def encode_text(self, text):
        raise NotImplementedError("encode_text method must be implemented")

# ImageTextEncoder is a model that can encode both image and text
# emsemble image encoder and text encoder in one model
class ImageTextEncoder(Model):
    def __init__(self, config):
        super().__init__(config)
    
    def encode_image(self, image):
        raise NotImplementedError("encode_image method must be implemented")
    
    def encode_text(self, text):
        raise NotImplementedError("encode_text method must be implemented")