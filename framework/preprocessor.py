

class Preprocessor:
    def __init__(self, config):
        # config is a dict containing the configuration for the preprocessor
        # should be defined in the config yaml file
        self.config = config

    def preprocess(self, img_data, txt_data, context=None):
        # Should return preprocessed image, text, and optionally context
        raise NotImplementedError
