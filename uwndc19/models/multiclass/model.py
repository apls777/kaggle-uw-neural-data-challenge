import tensorflow as tf
from uwndc19.models.layers import build_conv_layers, build_dense_layers


def model_fn(features, labels, mode, params):
    image = features['image']
    num_classes = params['model']['num_classes']
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # build convolutional layers
    conv = build_conv_layers(image, params['model']['conv_layers'])

    # build dense layers
    dense = build_dense_layers(conv, params['model']['dense_layers'], is_training)

    # get logits
    if 'subnet' in params:
        # build NN for each neuron
        subnet_dropout_rate = params['model']['subnet'].get('subnet_dropout_rate', 0)
        if subnet_dropout_rate:
            dense = tf.layers.dropout(inputs=dense, rate=subnet_dropout_rate, training=is_training)

        logits_concat = []
        for i in range(num_classes):
            subnet_dense = build_dense_layers(dense, params['model']['subnet']['dense_layers'], is_training)
            logits_dropout_rate = params['model'].get('logits_dropout_rate', 0)
            if logits_dropout_rate:
                subnet_dense = tf.layers.dropout(inputs=subnet_dense, rate=logits_dropout_rate, training=is_training)
            subnet_logits = tf.layers.dense(inputs=subnet_dense, units=1, activation=tf.nn.relu)
            logits_concat.append(subnet_logits)

        logits = tf.concat(logits_concat, axis=-1)
    else:
        # a single layer to get a spike
        logits_dropout_rate = params['model'].get('logits_dropout_rate', 0)
        if logits_dropout_rate:
            dense = tf.layers.dropout(inputs=dense, rate=logits_dropout_rate, training=is_training)
        logits = tf.layers.dense(inputs=dense, units=num_classes, activation=tf.nn.relu)

    # return prediction specification
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'spikes': logits})

    # compute the loss
    nan_mask = tf.cast(features['nan_mask'], tf.float32)
    mse_loss = tf.losses.mean_squared_error(labels=labels, predictions=logits, weights=nan_mask)

    # return training specification
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=mse_loss,
            global_step=tf.train.get_global_step(),
            learning_rate=params['training']['learning_rate'],
            optimizer='Adam',
            summaries=['learning_rate', 'loss', 'gradients', 'gradient_norm'],
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=mse_loss, train_op=train_op)

    # evaluation metrics
    eval_metric_ops = {
        'rmse': tf.metrics.root_mean_squared_error(labels=labels, predictions=logits, weights=nan_mask),
    }

    # RMSE per column
    for i in range(num_classes):
        eval_metric_ops['rmse/column%d' % i] = tf.metrics.root_mean_squared_error(labels=labels[:, i],
                                                                                  predictions=logits[:, i],
                                                                                  weights=nan_mask[:, i])

    return tf.estimator.EstimatorSpec(mode=mode, loss=mse_loss, eval_metric_ops=eval_metric_ops)
