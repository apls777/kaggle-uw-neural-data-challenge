import tensorflow as tf
from uwndc19.input import load_data, get_column_data
from uwndc19.model import model_fn
from uwndc19.utils import root_dir


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    df, stim = load_data()

    columns = list(df.columns)[1:]
    print(columns)

    for i, column_name in enumerate(columns):
        print('Column %s' % column_name)

        train_imgs, train_labels, eval_imgs, eval_labels, test_imgs = get_column_data(column_name, df, stim)

        # create the estimator
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=tf.estimator.RunConfig(
                model_dir=root_dir('training/final_models/%d_%s' % (i, column_name)),
                save_checkpoints_steps=10,
                save_summary_steps=10,
            )
        )

        tensors_to_log = {}  # {'probabilities': 'logits_test'}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=10)

        # train the model
        train_data = {'image': train_imgs}
        train_input_fn = tf.estimator.inputs.numpy_input_fn(x=train_data, y=train_labels,
                                                            batch_size=len(train_labels), num_epochs=None, shuffle=True)

        # evaluate the model and print results
        eval_data = {'image': eval_imgs}
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(x=eval_data, y=eval_labels, num_epochs=1, shuffle=False)

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=100)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                          hooks=[logging_hook],
                                          throttle_secs=0)

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    main()
