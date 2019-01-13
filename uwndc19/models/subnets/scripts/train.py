import tensorflow as tf
from tensorflow.contrib.estimator.python.estimator import early_stopping
from uwndc19.dataset import load_data, get_datasets
from uwndc19.models.subnets.input import serving_input_receiver_fn
from uwndc19.models.subnets.model import model_fn
from uwndc19.utils import root_dir


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    eval_steps = 10
    eval_size = 30
    export_best_models = True
    model_name = 'do04-d128-do04'

    # load the data
    df, imgs = load_data()
    columns = list(df.columns)[1:]
    single_neuron = [int(column_name[-1:]) for column_name in columns]
    train_imgs, train_labels, train_nan_mask, eval_imgs, eval_labels, eval_nan_mask = get_datasets(df, imgs, eval_size)

    print('Train size: %d, eval size: %d' % (len(train_labels), len(eval_labels)))

    # create the estimator
    model_dir = root_dir('training/subnets/%s' % model_name)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=tf.estimator.RunConfig(
            model_dir=model_dir,
            save_checkpoints_steps=eval_steps,
            save_summary_steps=eval_steps,
            keep_checkpoint_max=3,
        ),
        params={
            'single_neuron': single_neuron,
        }
    )

    # training input function
    train_data = {
        'image': train_imgs,
        'nan_mask': train_nan_mask,
    }
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x=train_data, y=train_labels,
                                                        batch_size=len(train_labels), num_epochs=None, shuffle=True)
    # evaluation training function
    eval_data = {
        'image': eval_imgs,
        'nan_mask': eval_nan_mask,
    }
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x=eval_data, y=eval_labels,
                                                       batch_size=len(eval_labels), num_epochs=1, shuffle=False)

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


if __name__ == '__main__':
    main()
