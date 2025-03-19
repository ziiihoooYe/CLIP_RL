
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

     

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Image Encoder
class ImageEncoder:
    def __init__(self, model):
        self.model = model.visual.to(device)

    def encode(self, image):
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            return self.model(image)

# Text Encoder
class TextEncoder:
    def __init__(self, model):
        self.model = model

    def encode(self, text):
        text_tokens = clip.tokenize([text]).to(device)
        with torch.no_grad():
            return self.model.encode_text(text_tokens)

