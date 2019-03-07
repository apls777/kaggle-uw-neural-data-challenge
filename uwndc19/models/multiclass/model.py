import tensorflow as tf
from uwndc19.models.layers import build_conv_layers, build_dense_layers
from uwndc19.core.utils import root_dir


def model_fn(features, labels, mode, params):
    image = features['image']
    num_classes = params['model']['num_classes']
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # build convolutional layers
    conv = build_conv_layers(image, params['model']['conv_layers'], is_training)

    # load convolutional and dense layers from a checkpoint
    freeze_variables = {}
    checkpoint_path = params['training'].get('checkpoint_path')
    freeze_restored_variables = params['training'].get('freeze_restored_variables', False)
    if checkpoint_path:
        tvars = tf.trainable_variables()
        assignment_map = {}
        for var in tvars:
            assignment_map[var.name[:-2]] = var
            if freeze_restored_variables:
                freeze_variables[var.name] = True

        tf.train.init_from_checkpoint(root_dir(checkpoint_path), assignment_map)

    # build dense layers
    dense = build_dense_layers(conv, params['model']['dense_layers'], is_training)

    # get logits
    if 'subnet' in params:
        # build NN for each neuron
        subnet_dropout_rate = params['model']['subnet'].get('subnet_dropout_rate', 0)
        if subnet_dropout_rate:
            dense = tf.layers.dropout(inputs=dense, rate=subnet_dropout_rate, training=is_training)

        logits_layer_params = dict(params['model']['logits_layer'])
        logits_layer_params['num_units'] = 1

        logits_concat = []
        for i in range(num_classes):
            subnet_dense = build_dense_layers(dense, params['model']['subnet']['dense_layers'], is_training)
            subnet_logits = build_dense_layers(subnet_dense, [logits_layer_params], is_training)
            logits_concat.append(subnet_logits)

        logits = tf.concat(logits_concat, axis=-1)
    else:
        # a single layer to get a spike
        logits_layer_params = dict(params['model']['logits_layer'])
        logits_layer_params['num_units'] = num_classes
        logits = build_dense_layers(dense, [logits_layer_params], is_training)

    # return prediction specification
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'spikes': logits})

    # make sure that images were distorted correctly and display them in TensorBoard
    max_images = 12
    images = image[:max_images]
    assert_min = tf.assert_greater_equal(tf.reduce_min(images), 0.0, message='Image contains values less than 0')
    assert_max = tf.assert_less_equal(tf.reduce_max(images), 1.0, message='Image contains values greater than 1')
    with tf.control_dependencies([assert_min, assert_max]):
        tf.summary.image('images', tf.cast(images * 255, dtype=tf.uint8), max_outputs=max_images)

    # compute the loss
    nan_mask = tf.cast(features['nan_mask'], tf.float32)
    mse_loss = tf.losses.mean_squared_error(labels=labels, predictions=logits, weights=nan_mask)
    loss = mse_loss + tf.losses.get_regularization_loss()

    # get train variables
    train_vars = [var for var in tf.trainable_variables() if var.name not in freeze_variables]

    # return training specification
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=params['training']['learning_rate'],
            optimizer='Adam',
            summaries=['learning_rate', 'loss', 'gradients', 'gradient_norm'],
            variables=train_vars,
        )

        # perform update ops for batch normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group([train_op, update_ops])

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # evaluation metrics
    eval_metric_ops = {
        'rmse': tf.metrics.root_mean_squared_error(labels=labels, predictions=logits, weights=nan_mask),
    }

    # RMSE per column
    for i in range(num_classes):
        eval_metric_ops['rmse/column%d' % i] = tf.metrics.root_mean_squared_error(labels=labels[:, i],
                                                                                  predictions=logits[:, i],
                                                                                  weights=nan_mask[:, i])

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
