import tensorflow as tf
from tensorflow.python.ops import random_ops


def flip_left_right(image):
    return tf.image.random_flip_left_right(image)


def flip_up_down(image):
    return tf.image.random_flip_up_down(image)


def saturation(image, min_value: float, max_value: float):
    return tf.image.random_saturation(image, min_value, max_value)


def brightness(image, max_factor: float):
    assert max_factor > 0
    factor = random_ops.random_uniform([], 1 - max_factor, 1 + max_factor)
    image = tf.clip_by_value(image * factor, 0.0, 1.0)
    return image


def rotate(image, angle: float, interpolation: str):
    assert angle > 0
    assert interpolation in ['NEAREST', 'BILINEAR']

    rotate_angle = random_ops.random_uniform([], -angle, angle)
    image = tf.contrib.image.rotate(image, rotate_angle, interpolation=interpolation)

    return image
