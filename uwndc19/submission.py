import csv
import tensorflow as tf
from uwndc19.input import load_data
from uwndc19.model import model_fn
from uwndc19.utils import root_dir


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    df, stim = load_data()
    columns = list(df.columns)[1:]

    test_imgs = stim[:50, 16:-16, 16:-16]
    test_data = {'image': test_imgs}
    test_input_fn = tf.estimator.inputs.numpy_input_fn(x=test_data, num_epochs=1, shuffle=False)

    predictions = {}
    for i, column_name in enumerate(columns):
        print('Column %s' % column_name)

        # create the estimator
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=root_dir('training/final_models/%d_%s' % (i, column_name)),
        )

        predictor = estimator.predict(input_fn=test_input_fn)
        predictions[column_name] = [abs(prediction['spike']) for prediction in predictor]

    with open(root_dir('data/submission/2.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Id'] + columns)
        for i in range(len(test_imgs)):
            writer.writerow([i] + [predictions[column_name][i] for column_name in columns])


if __name__ == "__main__":
    main()
