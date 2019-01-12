import tensorflow as tf


def model_fn(features, labels, mode, params):
    input_layer = tf.reshape(features['image'], [-1, 48, 48, 3])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
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

    # conv3 = tf.layers.conv2d(
    #     inputs=pool2,
    #     filters=128,
    #     kernel_size=[3, 3],
    #     padding='same',
    #     activation=tf.nn.relu)
    # pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    # flat = tf.reshape(pool3, [-1, 6 * 6 * 128])

    flat = tf.reshape(pool2, [-1, 12 * 12 * 64])

    # dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
    # dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    # logits = tf.layers.dense(inputs=dropout, units=1)

    logits = tf.layers.dense(inputs=flat, units=1)
    logits = tf.reshape(logits, shape=(-1,))

    predictions = {
        'spike': tf.maximum(tf.constant(0, dtype=tf.float32), logits),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=0.001,
            optimizer='Adam',
            summaries=['learning_rate', 'loss', 'gradients', 'gradient_norm'],
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'rmse': tf.metrics.root_mean_squared_error(labels=labels, predictions=logits),
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
