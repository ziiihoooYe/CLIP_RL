
class Model:
    def __init__(self, config):
        # config is a dict containing the configuration for the model
        # Model Object should have the following attributes:
        # - config: configuration for the model
        # - model: the callable model object
        self.config = config
        self.model = None
    
    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def encode(self, data):
        raise NotImplementedError("encode method must be implemented")

    def to(self, device):
        self.model.to(device)

     

class ImageEncoder(Model):
    def __init__(self, config):
        super().__init__(config)
        device = config.get("device", "cpu")
        model, _ = clip.load(config.get("clip_model", "ViT-B/32"), device=device)
        self.model = model.visual  # Extract image encoder

    def encode(self, image_path):
        device = next(self.model.parameters()).device
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            return self.model(image)
    
class TextEncoder(Model):
    def __init__(self, config):
        super().__init__(config)
        device = config.get("device", "cpu")
        model, _ = clip.load(config.get("clip_model", "ViT-B/32"), device=device)
        self.model = model.encode_text  # Extract text encoder

    def encode(self, text):
        device = next(self.model.parameters()).device
        text_tokens = clip.tokenize([text]).to(device)
        with torch.no_grad():
            return self.model(text_tokens)
