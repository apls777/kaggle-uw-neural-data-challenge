import os
import tensorflow as tf
from tensorflow.contrib import predictor
from tensorflow.contrib.estimator import InMemoryEvaluatorHook
from tensorflow.contrib.estimator.python.estimator import early_stopping
from uwndc19.hp_tuning.report_exporter import ReportExporter
from .utils import root_dir
from .abstract_builder import AbstractBuilder


def train(builder: AbstractBuilder, model_dir: str, reporter=None, session_config=None):
    config = builder.config
    model_dir = root_dir(model_dir)

    eval_steps = config['training']['eval_steps']
    export_best_models = config['training']['export_best_models']
    performance_metric = 'rmse'

    estimator = tf.estimator.Estimator(
        model_fn=builder.build_model_fn(),
        config=tf.estimator.RunConfig(
            model_dir=model_dir,
            session_config=session_config,
            save_checkpoints_steps=eval_steps,
            save_summary_steps=eval_steps,
            keep_checkpoint_max=config['training']['keep_checkpoint_max'],
        ),
        params=config,
    )

    # hooks
    early_stopping_hook = early_stopping.stop_if_no_decrease_hook(estimator, performance_metric, eval_steps * 10,
                                                                  run_every_secs=None, run_every_steps=eval_steps)
    train_evaluator = InMemoryEvaluatorHook(estimator, builder.build_train_input_fn(), name='train', steps=1,
                                            every_n_iter=eval_steps)

    # exporters
    exporters = []
    if export_best_models:
        best_exporter = tf.estimator.BestExporter(
            name='best',
            serving_input_receiver_fn=builder.build_serving_input_receiver_fn(),
            exports_to_keep=config['training']['exports_to_keep'],
            compare_fn=lambda best_eval_result, current_eval_result:
                # should be "<=" to export the best model on the 1st evaluation
                current_eval_result[performance_metric] <= best_eval_result[performance_metric],
        )
        exporters.append(best_exporter)

    if reporter:
        report_exporter = ReportExporter(reporter, [performance_metric])
        exporters.append(report_exporter)

    # train and evaluate
    train_spec = tf.estimator.TrainSpec(input_fn=builder.build_train_input_fn(),
                                        hooks=[early_stopping_hook, train_evaluator])
    eval_spec = tf.estimator.EvalSpec(input_fn=builder.build_eval_input_fn(),
                                      exporters=exporters,
                                      throttle_secs=0)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def get_predictor(export_dir):
    # find the latest version of the model
    latest_model_subdir = sorted(os.listdir(export_dir), reverse=True)[0]
    latest_model_dir = os.path.join(export_dir, latest_model_subdir)

    # get the predictor
    predict_fn = predictor.from_saved_model(latest_model_dir)

    return predict_fn