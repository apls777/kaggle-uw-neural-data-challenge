import tensorflow as tf


def serving_input_receiver_fn():
    receiver_tensors = {
        'image': tf.placeholder(tf.float32, [None, 48, 48, 3], name='ph_image'),
    }

    return tf.estimator.export.ServingInputReceiver(receiver_tensors, receiver_tensors)
