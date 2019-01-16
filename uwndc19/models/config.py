import os
import yaml
from uwndc19.utils import root_dir


def read_config(model_type, model_name):
    with open(root_dir(os.path.join('configs', model_type, model_name + '.yaml'))) as f:
        res = yaml.load(f)

    return res
