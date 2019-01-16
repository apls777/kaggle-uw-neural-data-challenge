import tensorflow as tf
from tensorflow.contrib.estimator.python.estimator import early_stopping
from uwndc19.utils import root_dir


def train(model_fn, train_input_fn, eval_input_fn, serving_input_receiver_fn, config):
    model_dir = root_dir(config['data']['model_dir'])
    eval_steps = config['training']['eval_steps']
    export_best_models = config['training']['export_best_models']

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf.estimator.RunConfig(
            model_dir=model_dir,
            save_checkpoints_steps=eval_steps,
            save_summary_steps=eval_steps,
            keep_checkpoint_max=3,
        ),
        params=config,
    )

    # hooks
    early_stopping_hook = early_stopping.stop_if_no_decrease_hook(estimator, 'rmse', eval_steps * 10,
                                                                  run_every_secs=None, run_every_steps=eval_steps)
    exporter = tf.estimator.BestExporter(name='best',
                                         serving_input_receiver_fn=serving_input_receiver_fn,
                                         exports_to_keep=3,
                                         compare_fn=lambda best_eval_result, current_eval_result:
                                             # should be "<=" to export the best model on the 1st evaluation
                                             current_eval_result['rmse'] <= best_eval_result['rmse'],
                                         )

    # train and evaluate
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, hooks=[early_stopping_hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      exporters=(exporter if export_best_models else None),
                                      throttle_secs=0)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
