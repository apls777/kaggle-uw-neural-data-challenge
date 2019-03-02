import argparse
import tensorflow as tf
from tensorflow.python.platform import tf_logging
from uwndc19.core.estimator import train
from uwndc19.core.config import load_model_config, create_model_config
from uwndc19.core.builder_factory import create_builder


def train_model(model_type: str, model_dir: str, config_path: str = None):
    """Trains the model.

    If only model's directory is specified, the script will be looking for a configuration file in that directory
    and will resume the training.

    If a configuration file is provided, it will be used for training and will be copied to the model's directory.
    If the directory already exist, then the previous configuration file will be renamed and training will be
    resumed using a new configuration.
    """

    # load an existing model config or create a new config in the model directory
    config = create_model_config(model_dir, config_path) if config_path else load_model_config(model_dir)

    # create the model builder
    model_builder = create_builder(model_type, config)

    # train and evaluate the model
    train(model_builder, model_dir)


def main():
    # enable TensorFlow logging
    tf.logging.set_verbosity(tf.logging.INFO)
    tf_logging._get_logger().propagate = False  # fix double messages

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--model-type', type=str, default='multiclass', help='Model type')
    parser.add_argument('-d', '--model-dir', type=str, help='Model\'s directory')
    parser.add_argument('-c', '--config', type=str, help='Path to the config')

    args = parser.parse_args()

    model_type = args.model_type
    model_dir = args.model_dir
    config_path = args.config

    # default values for local testing
    if not model_dir and not config_path:
        model_dir = 'training/%s/local/test1' % model_type
        config_path = 'configs/%s/default.yaml' % model_type

    # train and evaluate the model
    train_model(model_type, model_dir, config_path)


if __name__ == '__main__':
    main()
