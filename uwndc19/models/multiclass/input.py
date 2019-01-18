import tensorflow as tf


def build_serving_input_receiver_fn(image_size: int):
    def serving_input_receiver_fn():
        receiver_tensors = {
            'image': tf.placeholder(tf.float32, [None, image_size, image_size, 3], name='ph_image'),
        }
        return tf.estimator.export.ServingInputReceiver(receiver_tensors, receiver_tensors)

    return serving_input_receiver_fn


def build_input_fn(imgs, labels, nan_mask, num_epochs=None):
    features = {
        'image': imgs,
        'nan_mask': nan_mask,
    }

    # no shuffling, otherwise several epochs will be mixed together
    return tf.estimator.inputs.numpy_input_fn(x=features, y=labels, batch_size=len(labels),
                                              num_epochs=num_epochs, shuffle=False)

