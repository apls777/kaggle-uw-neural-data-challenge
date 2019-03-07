from math import pi
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


def rotate(image, max_angle: int, interpolation: str = 'BILINEAR'):
    assert 0 < max_angle <= 180
    assert interpolation in ['NEAREST', 'BILINEAR']

    max_angle_radians = tf.cast(max_angle, dtype=tf.float32) * pi / 180
    rotate_angle_radians = random_ops.random_uniform([], -max_angle_radians, max_angle_radians)

    return tf.contrib.image.rotate(image, rotate_angle_radians, interpolation=interpolation)


def rotate_choice(image, angles: list, interpolation: str = 'BILINEAR'):
    assert interpolation in ['NEAREST', 'BILINEAR']

    rotate_angle = tf.random.shuffle(angles)[0]
    rotate_angle_radians = tf.cast(rotate_angle, dtype=tf.float32) * pi / 180

    return tf.contrib.image.rotate(image, rotate_angle_radians, interpolation=interpolation)
