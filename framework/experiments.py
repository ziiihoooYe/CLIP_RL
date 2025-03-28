class Experiment:
    def __init__(self, config):
        self.config = config
    
    def run(self, model, dataset, preprocessor_list, logger):
        logger.info(self.config)
        return model