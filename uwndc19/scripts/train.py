import tensorflow as tf
from tensorflow.python.platform import tf_logging
from uwndc19.core.estimator import train
from uwndc19.core.config import get_model_config, create_model_config
from uwndc19.core.builder_factory import create_builder


def train_model(model_type: str, model_dir: str, config_path: str = None):
    # load an existing model config or create a new config in the model directory
    config = create_model_config(model_dir, config_path) if config_path else get_model_config(model_dir)

    # create the model builder
    model_builder = create_builder(model_type, config)

    # train and evaluate the model
    train(model_builder, model_dir)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    tf_logging._get_logger().propagate = False  # fix double messages

    model_type = 'multiclass'
    model_dir = 'training/multiclass/experiments1/do01-d512l2-do0'
    config_path = 'configs/multiclass/local.yaml'

    train_model(model_type, model_dir, config_path)


if __name__ == '__main__':
    main()
