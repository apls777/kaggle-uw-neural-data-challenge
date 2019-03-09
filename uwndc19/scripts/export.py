import argparse
import tensorflow as tf
from tensorflow.python.platform import tf_logging
from uwndc19.core.builder_factory import create_builder
from uwndc19.core.config import load_model_config
from uwndc19.core.estimator import export


def main():
    # enable TensorFlow logging
    tf.logging.set_verbosity(tf.logging.INFO)
    tf_logging._get_logger().propagate = False  # fix double messages

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--model-type', type=str, default='multiclass', help='Model type')
    parser.add_argument('-d', '--model-dir', type=str, required=True, help='Model\'s directory')
    parser.add_argument('-s', '--checkpoint-step', type=int, help='Export the model from the checkpoint taken on '
                                                                  'the specified global step.')

    args = parser.parse_args()

    # load the model config
    config = load_model_config(args.model_dir)

    # create the model builder
    model_builder = create_builder(args.model_type, config)

    # train and evaluate the model
    export(model_builder, args.model_dir, args.checkpoint_step)


if __name__ == '__main__':
    main()
