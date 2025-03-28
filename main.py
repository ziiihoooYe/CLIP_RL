import argparse
from utils.utils import *
from exp.ContrastiveLearning import *
from exp.ExpFactory import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='clip_train.yaml', help='config file')
    args = parser.parse_args()
    

    # --------------------
    # Module Instantiation
    # --------------------
    config = read_config(args.config)
    defaults = config["default"] 
    logger = get_logger(defaults.get("log_file", "training.log"))
    logger.info("Starting main function with config file: %s", args.config)
    
    # model
    model = instantiate_module('model', config["model"], defaults)

    # preprocessor
    preprocessor_list = [
        instantiate_module('preprocessor', preprocessor_config, defaults) 
        for preprocessor_config
        in config["preprocessor"]
    ]
    
    
    # --------------------
    # Experiment
    # ----------------
    for exp_config in config["exp"]:
        model = ExperimentFactory.run(model, exp_config, preprocessor_list, defaults, logger)
        

if __name__ == "__main__":
    main()
