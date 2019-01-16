from importlib import import_module
import tensorflow as tf
from uwndc19.models.config import read_config
from uwndc19.models.estimator import train


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    model_type = 'multiclass'
    model_name = '80px-valid-do04-d512-do04-test'

    # load model config
    config = read_config(model_type, model_name)

    # import build function
    build_fn = getattr(import_module('uwndc19.models.%s.build' % model_type), 'build')

    # get model and input functions
    model_fn, train_input_fn, eval_input_fn, serving_input_receiver_fn = build_fn(config)

    return train(model_fn, train_input_fn, eval_input_fn, serving_input_receiver_fn, config)


if __name__ == '__main__':
    main()
