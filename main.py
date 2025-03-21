import argparse
import logging
from framework.utils import *
from exp.ContrastiveLearning import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='clip_train.yaml', help='config file')
    args = parser.parse_args()
    
    logger.info("Starting main function with config file: %s", args.config)

    # --------------------
    # Module Instantiation
    # --------------------
    config = read_config(args.config)
    defaults = config["default"] 
    fh = logging.FileHandler(defaults.get("log_file", "training.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(defaults.get("log_format", '%(asctime)s - %(name)s - %(levelname)s - %(message)s')))
    logger.addHandler(fh)
    
    # model
    model = instantiate_module('model', config["model"], defaults)

    # dataset
    dataset = instantiate_module('dataset', config["dataset"], defaults)

    # preprocessor
    preprocessor_list = [instantiate_module('preprocessor', preprocessor_config, defaults) for preprocessor_config in config["preprocessor"]]
    
    if defaults["exp"] == "ContrastiveLearning":
        ContrastiveLearning(model, dataset, preprocessor_list, config, logger)

if __name__ == "__main__":
    main()
