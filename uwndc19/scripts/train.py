import tensorflow as tf
from uwndc19.manager.model_manager import ModelManager


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    model_type = 'multiclass'
    model_name = 'do07-d512-do04-sub-d128-do04-tanh'

    model_manager = ModelManager(model_type, model_name)
    model_manager.train()


if __name__ == '__main__':
    main()
