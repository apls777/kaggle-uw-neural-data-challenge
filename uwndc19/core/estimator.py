import tensorflow as tf
from tensorflow.contrib.estimator import InMemoryEvaluatorHook
from tensorflow.contrib.estimator.python.estimator import early_stopping
from uwndc19.hp_tuning.report_exporter import ReportExporter
from .utils import root_dir
from .abstract_builder import AbstractBuilder
import os


def train(builder: AbstractBuilder, model_dir: str, reporter=None, session_config=None):
    config = builder.config
    model_dir = root_dir(model_dir)

    eval_steps = config['training']['eval_steps']
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

    # training hooks
    hooks = []

    # early stopping hook
    early_stopping_evals = config['training'].get('early_stopping_evals')
    if early_stopping_evals:
        early_stopping_steps = eval_steps * early_stopping_evals
        early_stopping_hook = early_stopping.stop_if_no_decrease_hook(estimator, performance_metric,
                                                                      early_stopping_steps, run_every_secs=None,
                                                                      run_every_steps=eval_steps)
        hooks.append(early_stopping_hook)

    # in-memory evaluation on the training data (metrics from this evaluation will be more accurate than
    # actual training metrics, because dropouts are disabled)
    train_evaluator = InMemoryEvaluatorHook(estimator, builder.build_eval_train_input_fn(), name='train', steps=1,
                                            every_n_iter=eval_steps)
    hooks.append(train_evaluator)

    # evaluation exporters
    exporters = []

    # export best models
    export_best_models = config['training']['export_best_models']
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

    # report evaluation metrics to Ray
    if reporter:
        report_exporter = ReportExporter(reporter, [performance_metric])
        exporters.append(report_exporter)

    # train and evaluate
    train_spec = tf.estimator.TrainSpec(input_fn=builder.build_train_input_fn(),
                                        hooks=hooks)
    eval_spec = tf.estimator.EvalSpec(input_fn=builder.build_eval_input_fn(),
                                      exporters=exporters,
                                      throttle_secs=0)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def export(builder: AbstractBuilder, model_dir: str, checkpoint_step: int = None):
    model_dir = root_dir(model_dir)
    export_dir = os.path.join(model_dir, 'export', 'checkpoints')
    checkpoint_path = os.path.join(model_dir, 'model.ckpt-%s' % checkpoint_step) if checkpoint_step else None

    # create an estimator
    estimator = tf.estimator.Estimator(
        model_fn=builder.build_model_fn(),
        model_dir=model_dir,
        params=builder.config,
    )

    estimator.export_savedmodel(export_dir, serving_input_receiver_fn=builder.build_serving_input_receiver_fn(),
                                checkpoint_path=checkpoint_path)
