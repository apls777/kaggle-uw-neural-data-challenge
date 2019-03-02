import logging
import os
import yaml
from time import time
from tensorflow.python.lib.io import file_io
from .utils import root_dir, fix_s3_separators, is_s3_path

CONFIG_FILENAME = 'config.yaml'


def get_model_config_path(model_dir: str):
    model_dir = root_dir(model_dir)
    model_config_path = fix_s3_separators(os.path.join(model_dir, CONFIG_FILENAME))

    return model_config_path


def write_model_config(model_dir: str, config_str: str):
    """Writes the config to the model's directory."""
    model_dir = root_dir(model_dir)
    model_config_path = get_model_config_path(model_dir)

    # if the model config file already exists, rename it
    if file_io.file_exists(model_config_path):
        prev_config_filename = '%s_%d.yaml' % (model_config_path.split('.')[0], int(time()))
        file_io.rename(model_config_path, prev_config_filename)
        logging.info('Previous model config file was renamed: %s' % prev_config_filename)

    # save the config file to the model directory
    if not is_s3_path(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    file_io.write_string_to_file(model_config_path, config_str)


def load_model_config(model_dir: str):
    """Loads a config from the model's directory."""
    config_path = get_model_config_path(model_dir)
    config_content = file_io.read_file_to_string(config_path)
    config = yaml.safe_load(config_content)

    return config


def create_model_config(model_dir: str, config_path: str = None):
    """Creates a new configuration file in the model directory and returns the config."""

    # read the config file
    config_content = file_io.read_file_to_string(root_dir(config_path))

    # save the config file to the model directory
    write_model_config(model_dir, config_content)

    # load config
    config = yaml.safe_load(config_content)

    return config
