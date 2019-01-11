import tensorflow as tf
from tensorflow.contrib.estimator.python.estimator import early_stopping
from uwndc19.input import load_data, get_column_data, serving_input_receiver_fn
from uwndc19.model import model_fn
from uwndc19.utils import root_dir


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    eval_steps = 10
    export_best_models = False
    train_column_id = None
    models_dir = 'training/models2'
    models_postfix = ''

    # load the data
    df, stim = load_data()
    columns = list(df.columns)[1:]

    # train a model for each column
    for i, column_name in enumerate(columns):
        if (train_column_id is not None) and (i != train_column_id):
            continue

        print('Training the model for the column "%s"...' % column_name)

        train_imgs, train_labels, eval_imgs, eval_labels = get_column_data(column_name, df, stim)

        # create the estimator
        model_dir = root_dir('%s/%d_%s' % (models_dir, i, column_name))
        if models_postfix:
            model_dir += '_' + models_postfix

        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=tf.estimator.RunConfig(
                model_dir=model_dir,
                save_checkpoints_steps=eval_steps,
                save_summary_steps=eval_steps,
            )
        )

        # training input function
        train_data = {'image': train_imgs}
        train_input_fn = tf.estimator.inputs.numpy_input_fn(x=train_data, y=train_labels,
                                                            batch_size=len(train_labels), num_epochs=None, shuffle=True)
        # evaluation training function
        eval_data = {'image': eval_imgs}
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(x=eval_data, y=eval_labels, num_epochs=1, shuffle=False)

        # hooks
        early_stopping_hook = early_stopping.stop_if_no_decrease_hook(estimator, 'rmse', eval_steps * 10,
                                                                      run_every_secs=None, run_every_steps=eval_steps)
        exporter = tf.estimator.BestExporter(name='best',
                                             serving_input_receiver_fn=serving_input_receiver_fn,
                                             exports_to_keep=1,
                                             compare_fn=lambda best_eval_result, current_eval_result:
                                                 current_eval_result['rmse'] < best_eval_result['rmse'],
                                             )

        # train and evaluate
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, hooks=[early_stopping_hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                          exporters=(exporter if export_best_models else None),
                                          throttle_secs=0)

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    main()
