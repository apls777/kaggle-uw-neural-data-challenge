import csv
import os
import tensorflow as tf
from uwndc19.helpers.dataset import load_data
from uwndc19.core.utils import root_dir
from tensorflow.contrib import predictor


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    models_dir = 'training/single_column/models3'

    # load the data
    df, stim = load_data()
    columns = list(df.columns)[1:]
    test_data = {'image': stim[:50, 16:-16, 16:-16]}

    # get predictions from all the models
    column_predictions = {}
    for i, column_name in enumerate(columns):
        print('Predicting values for the column "%s"...' % column_name)

        # find the model directory
        best_models_path = root_dir('%s/%d_%s/export/best' % (models_dir, i, column_name))
        latest_model_subdir = sorted(os.listdir(best_models_path), reverse=True)[0]
        latest_model_dir = os.path.join(best_models_path, latest_model_subdir)

        # create predictor
        predict_fn = predictor.from_saved_model(latest_model_dir)

        # get predictions
        column_predictions[column_name] = predict_fn(test_data)['spike']

    # generate a submission file
    with open(root_dir('data/submission/single_column/3.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Id'] + columns)
        for i in range(len(test_data['image'])):
            writer.writerow([i] + [column_predictions[column_name][i] for column_name in columns])


if __name__ == '__main__':
    main()
