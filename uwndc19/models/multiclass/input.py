import tensorflow as tf


def build_serving_input_receiver_fn(image_size: int):
    def serving_input_receiver_fn():
        receiver_tensors = {
            'image': tf.placeholder(tf.float32, [None, image_size, image_size, 3], name='ph_image'),
        }
        return tf.estimator.export.ServingInputReceiver(receiver_tensors, receiver_tensors)

    return serving_input_receiver_fn


def build_input_fn(imgs, labels, nan_mask, num_epochs=None, shuffle=True):
    features = {
        'image': imgs,
        'nan_mask': nan_mask,
    }

    return tf.estimator.inputs.numpy_input_fn(x=features, y=labels, batch_size=len(labels),
                                              num_epochs=num_epochs, shuffle=shuffle)


def build_train_input_fn(train_imgs, train_labels, train_nan_mask):
    return build_input_fn(train_imgs, train_labels, train_nan_mask, num_epochs=None, shuffle=True)


def build_eval_input_fn(eval_imgs, eval_labels, eval_nan_mask):
    return build_input_fn(eval_imgs, eval_labels, eval_nan_mask, num_epochs=1, shuffle=False)
