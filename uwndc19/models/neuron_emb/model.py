import tensorflow as tf
from uwndc19.models.layers import build_conv_layers, build_dense_layers


def model_fn(features, labels, mode, params):
    image = features['image']
    neuron_id = features['neuron_id']
    num_classes = params['model']['num_classes']
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # build convolutional layers
    conv = build_conv_layers(image, params['model']['conv_layers'])

    # build dense layers
    dense = build_dense_layers(conv, params['model']['dense_layers'], is_training)  # [batch_size, units]

    # get neurons embeddings
    neuron_embeddings = tf.get_variable('neuron_embeddings', shape=(num_classes, dense.get_shape()[-1]),
                                        dtype=tf.float32)
    neurons_batch = tf.nn.embedding_lookup(neuron_embeddings, neuron_id)  # [batch_size, units]

    # get predictions
    logits = tf.nn.relu(tf.reduce_sum(dense * neurons_batch, axis=-1))  # [batch_size]

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'spikes': logits})

    # compute the loss
    mse_loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)

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
        'rmse': tf.metrics.root_mean_squared_error(labels=labels, predictions=logits),
    }

    # RMSE per column
    for i in range(num_classes):
        class_indices = tf.reshape(tf.where(tf.equal(neuron_id, tf.constant(i, dtype=tf.int32))), shape=(-1,))
        eval_metric_ops['rmse/column%d' % i] = tf.metrics.root_mean_squared_error(labels=tf.gather(labels, class_indices),
                                                                                  predictions=tf.gather(logits, class_indices))

    return tf.estimator.EstimatorSpec(mode=mode, loss=mse_loss, eval_metric_ops=eval_metric_ops)
