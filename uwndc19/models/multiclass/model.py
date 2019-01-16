import tensorflow as tf
from tensorflow.python.ops.losses.losses_impl import Reduction
from uwndc19.models.layers import build_conv_layers, build_dense_layers


def model_fn(features, labels, mode, params):
    image = features['image']

    # build convolutional layers
    conv = build_conv_layers(image, params['model']['conv_layers'])

    # build dense layers
    dense = build_dense_layers(conv, params['model']['dense_layers'], params['model']['logits_dropout_rate'],
                               mode == tf.estimator.ModeKeys.TRAIN)

    # get logits
    logits = tf.layers.dense(inputs=dense, units=params['model']['num_classes'], activation=tf.nn.relu)

    # return prediction specification
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'spikes': logits}
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
    for i in range(params['model']['num_classes']):
        tf.summary.scalar('rmse/column%d' % i, tf.sqrt(mse_per_column[i]))

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
    for i in range(params['model']['num_classes']):
        eval_metric_ops['rmse/column%d' % i] = tf.metrics.root_mean_squared_error(labels=labels[:, i],
                                                                                  predictions=logits[:, i],
                                                                                  weights=nan_mask[:, i])

    return tf.estimator.EstimatorSpec(mode=mode, loss=mse_loss, eval_metric_ops=eval_metric_ops)
