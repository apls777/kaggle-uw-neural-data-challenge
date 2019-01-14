import tensorflow as tf
from tensorflow.python.ops.losses.losses_impl import Reduction


def model_fn(features, labels, mode, params):
    image = features['image']

    conv1 = tf.layers.conv2d(
        inputs=image,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=96,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    flat = tf.reshape(pool3, [-1, 6 * 6 * 96])

    dropout = tf.layers.dropout(inputs=flat, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    dense = tf.layers.dense(inputs=dropout, units=512, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=18, activation=tf.nn.relu)

    predictions = {
        'spikes': logits,
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # compute the loss
    nan_mask = tf.cast(features['nan_mask'], tf.float32)
    mse_loss = tf.losses.mean_squared_error(labels=labels, predictions=logits, weights=nan_mask)

    # compute MSE per column
    mse_losses = tf.losses.mean_squared_error(labels=labels, predictions=logits, weights=nan_mask,
                                              reduction=Reduction.NONE)
    mse_per_column = [tf.reduce_sum(mse_losses[:, i]) / tf.reduce_sum(nan_mask[:, i]) for i in range(18)]

    # log the training RMSE for the batch
    tf.summary.scalar('rmse', tf.sqrt(mse_loss))

    # log the training RMSE per column for the batch
    for i in range(18):
        tf.summary.scalar('rmse/column%d' % i, tf.sqrt(mse_per_column[i]))

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=mse_loss,
            global_step=tf.train.get_global_step(),
            learning_rate=0.001,
            optimizer='Adam',
            summaries=['learning_rate', 'loss', 'gradients', 'gradient_norm'],
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=mse_loss, train_op=train_op)

    eval_metric_ops = {
        'rmse': tf.metrics.root_mean_squared_error(labels=labels, predictions=logits, weights=nan_mask),
    }

    # RMSE per column
    for i in range(18):
        eval_metric_ops['rmse/column%d' % i] = tf.metrics.root_mean_squared_error(labels=labels[:, i],
                                                                                  predictions=logits[:, i],
                                                                                  weights=nan_mask[:, i])

    return tf.estimator.EstimatorSpec(mode=mode, loss=mse_loss, eval_metric_ops=eval_metric_ops)
