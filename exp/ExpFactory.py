from utils.utils import instantiate_module, merge_config

class ExperimentFactory:
    @staticmethod
    def get_dataset(config, defaults):
        if config is None:
            raise ValueError("Dataset configuration is required")
        return instantiate_module('dataset', config, defaults)

    @staticmethod
    def get_experiment(config, defaults):
        return instantiate_module('exp', config, defaults)

    @staticmethod
    def run(model, exp_config, preprocessor_list, defaults, logger, device):
        # obtain merged configuration
        exp_config = merge_config(exp_config, defaults)
        dataset_config = merge_config(exp_config.get('dataset', None), defaults)

        # instantiate experiment and dataset
        exp = ExperimentFactory.get_experiment(exp_config, defaults)
        if 'class' in dataset_config:
            dataset = ExperimentFactory.get_dataset(dataset_config, defaults)
        else:
            dataset = None
        
        # run experiment
        logger.info(f"Running experiment: {exp.__class__.__name__}")
        logger.info(f"Config: {exp_config}")
        model = exp.run(model, dataset, preprocessor_list, logger, device)
        
        return model