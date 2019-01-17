from importlib import import_module
import os
import tensorflow as tf
from uwndc19.models.config import read_config
from uwndc19.models.estimator import train
from uwndc19.utils import root_dir


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    model_type = 'multiclass'
    model_name = 'do04-d512-do04-sub-d128-do04'

    # load model config
    config = read_config(model_type, model_name)

    # import build function
    build_fn = getattr(import_module('uwndc19.models.%s.build' % model_type), 'build_fn')

    # get model and input functions
    model_fn, train_input_fn, eval_input_fn, serving_input_receiver_fn = build_fn(config)

    # get model directory
    model_dir = root_dir(os.path.join('training', model_type, model_name))

    # train the model
    train(model_dir, model_fn, train_input_fn, eval_input_fn, serving_input_receiver_fn, config)


if __name__ == '__main__':
    main()
