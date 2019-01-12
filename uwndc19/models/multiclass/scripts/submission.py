import csv

import os
import tensorflow as tf
from uwndc19.dataset import load_data
from uwndc19.utils import root_dir
from tensorflow.contrib import predictor


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    submission_num = 5
    model_name = 'do02-d512-d02-eval30-2'
    model_dir = root_dir('training/multiclass/experiments1/%s/export/best' % model_name)
    latest_model_subdir = sorted(os.listdir(model_dir), reverse=True)[0]
    latest_model_dir = os.path.join(model_dir, latest_model_subdir)

    # load the data
    df, stim = load_data()
    columns = list(df.columns)[1:]
    test_data = {'image': stim[:50, 16:-16, 16:-16]}

    # create predictor
    predict_fn = predictor.from_saved_model(latest_model_dir)

    # get predictions
    predictions = predict_fn(test_data)['spikes']

    # generate a submission file
    with open(root_dir('data/submission/multiclass/%d.csv' % submission_num), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Id'] + columns)
        for i in range(len(test_data['image'])):
            writer.writerow([i] + list(predictions[i]))


if __name__ == '__main__':
    main()
