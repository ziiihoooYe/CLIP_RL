import os
import torch
import yaml
import importlib

module_config_map = {
    "model": "model/config.yaml",
    "preprocessor": "preprocessor/config.yaml",
    "dataset": "dataset/config.yaml",
    "exp": "exp/config.yaml"
}

def get_gpu_device(device):
    import os
    if device is not None and torch.cuda.is_available():
        return f"cuda:{device}"
    else:
        return "cpu"


def read_config(path: str) -> dict:
    with open(path, 'r') as f:
        contents = f.readlines()
    data = ''
    for c in contents:
        left = c.find('[')
        right = c.rfind(']')
        if left == -1 or right == -1 or right < left:
            data += c
            continue
        c0 = c[:left] + '"' + c[left:right+1] + '"' + c[right+1:]
        data += c0
    return yaml.safe_load(data)


def list_to_dict(args_list):
    result = {}
    for d in args_list:
        result.update(d)
    return result


def merge_config(default_config: dict, custom_config: dict) -> dict:
    if custom_config is None:
        return default_config
    elif default_config is None:
        return custom_config
    return {**default_config, **custom_config}


def instantiate_module(module_name, config: dict, defaults: dict):
    # update module config with class config
    module_config_path = module_config_map[module_name]
    module_config = read_config(module_config_path)[config["class"]]
    if "args" not in module_config:
        module_config["args"] = {}
    module_config["args"] = merge_config(module_config["args"], config.get("args", None))
    module_config["args"] = merge_config(module_config["args"], defaults)

    # instantiate class
    module_path, class_name = module_config["class"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    instance = cls(module_config["args"])
    return instance


def get_logger(file_name: str = 'training.log'):
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(file_name)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

    return logger