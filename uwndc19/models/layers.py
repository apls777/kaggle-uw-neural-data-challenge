from functools import reduce
import tensorflow as tf


def build_conv_layers(input_tensor, conv_layers_params: list):
    for layer_params in conv_layers_params:
        input_tensor = tf.layers.conv2d(
            inputs=input_tensor,
            filters=layer_params['num_filters'],
            kernel_size=[layer_params['kernel_size'], layer_params['kernel_size']],
            padding=layer_params['padding'],
            activation=tf.nn.relu)
        input_tensor = tf.layers.max_pooling2d(inputs=input_tensor, pool_size=[2, 2], strides=2)

    flat_dim = reduce(lambda x, y: x * y, input_tensor.get_shape()[1:])
    flat = tf.reshape(input_tensor, [-1, flat_dim])

    return flat


def build_dense_layers(input_tensor, dense_layers_params: list, is_training: bool):
    for layer_params in dense_layers_params:
        dropout_rate = layer_params.get('dropout_rate', 0)
        if dropout_rate:
            input_tensor = tf.layers.dropout(inputs=input_tensor, rate=dropout_rate, training=is_training)

        l2_regularization = layer_params.get('l2_regularization', 0)
        kernel_regularizer = None
        if l2_regularization:
            kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_regularization)

        input_tensor = tf.layers.dense(inputs=input_tensor, units=layer_params['num_units'], activation=tf.nn.relu,
                                       kernel_regularizer=kernel_regularizer)

    return input_tensor
