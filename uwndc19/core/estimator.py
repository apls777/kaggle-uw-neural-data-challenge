import os
import tensorflow as tf
from tensorflow.contrib import predictor
from tensorflow.contrib.estimator import InMemoryEvaluatorHook
from tensorflow.contrib.estimator.python.estimator import early_stopping
from .utils import root_dir
from .abstract_builder import AbstractBuilder


def train(builder: AbstractBuilder, model_dir: str):
    config = builder.config
    model_dir = root_dir(model_dir)

    eval_steps = config['training']['eval_steps']
    export_best_models = config['training']['export_best_models']

    estimator = tf.estimator.Estimator(
        model_fn=builder.build_model_fn(),
        config=tf.estimator.RunConfig(
            model_dir=model_dir,
            save_checkpoints_steps=eval_steps,
            save_summary_steps=eval_steps,
            keep_checkpoint_max=config['training']['keep_checkpoint_max'],
        ),
        params=config,
    )

    # hooks
    early_stopping_hook = early_stopping.stop_if_no_decrease_hook(estimator, 'rmse', eval_steps * 10,
                                                                  run_every_secs=None, run_every_steps=eval_steps)
    exporter = tf.estimator.BestExporter(name='best',
                                         serving_input_receiver_fn=builder.build_serving_input_receiver_fn(),
                                         exports_to_keep=config['training']['exports_to_keep'],
                                         compare_fn=lambda best_eval_result, current_eval_result:
                                             # should be "<=" to export the best model on the 1st evaluation
                                             current_eval_result['rmse'] <= best_eval_result['rmse'],
                                         )
    train_evaluator = InMemoryEvaluatorHook(estimator, builder.build_train_input_fn(), name='train', steps=1,
                                            every_n_iter=eval_steps)

    # train and evaluate
    train_spec = tf.estimator.TrainSpec(input_fn=builder.build_train_input_fn(),
                                        hooks=[early_stopping_hook, train_evaluator])
    eval_spec = tf.estimator.EvalSpec(input_fn=builder.build_eval_input_fn(),
                                      exporters=(exporter if export_best_models else None),
                                      throttle_secs=0)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def get_predictor(export_dir):
    # find the latest version of the model
    latest_model_subdir = sorted(os.listdir(export_dir), reverse=True)[0]
    latest_model_dir = os.path.join(export_dir, latest_model_subdir)

    # get the predictor
    predict_fn = predictor.from_saved_model(latest_model_dir)

    return predict_fn
