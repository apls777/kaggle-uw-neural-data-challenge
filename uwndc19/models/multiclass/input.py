import tensorflow as tf
import uwndc19.helpers.image_distortion_funcs


def build_serving_input_receiver_fn(image_size: int):
    def serving_input_receiver_fn():
        receiver_tensors = {
            'image': tf.placeholder(tf.float32, [None, image_size, image_size, 3], name='ph_image'),
        }
        return tf.estimator.export.ServingInputReceiver(receiver_tensors, receiver_tensors)

    return serving_input_receiver_fn


def build_input_fn(imgs, labels, nan_mask, distortions=None, num_epochs=None):
    def input_fn():
        features = {
            'image': imgs,
            'nan_mask': nan_mask,
        }

        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.repeat(num_epochs)

        if distortions:
            dataset = dataset.map(lambda f, l: _distort_image(f, l, distortions))

        dataset = dataset.batch(len(labels))

        iterator = dataset.make_one_shot_iterator()

        return iterator.get_next()

    return input_fn


def _distort_image(features, labels, distortions: dict):
    for func_name, params in distortions.items():
        features['image'] = getattr(uwndc19.helpers.image_distortion_funcs, func_name)(features['image'], **params)

    return features, labels
