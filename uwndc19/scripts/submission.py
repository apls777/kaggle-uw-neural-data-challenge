import csv
import os
import tensorflow as tf
from uwndc19.dataset import load_data, get_test_dataset
from uwndc19.models.config import read_config
from uwndc19.utils import root_dir
from tensorflow.contrib import predictor


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    submission_num = 12
    model_name = '80px-valid-do04-d512-do04-3'
    model_type = 'multiclass'

    # load model config
    config = read_config(model_type, model_name)

    # find the latest version of the model
    export_model_dir = root_dir('training/%s/%s/export/submission%d' % (model_type, model_name, submission_num))
    latest_model_subdir = sorted(os.listdir(export_model_dir), reverse=True)[0]
    latest_model_dir = os.path.join(export_model_dir, latest_model_subdir)

    # load the data
    df, stim = load_data()
    columns = list(df.columns)[1:]
    test_data = {'image': get_test_dataset(stim, config['model']['image_size'])}

    # create predictor
    predict_fn = predictor.from_saved_model(latest_model_dir)

    # get predictions
    predictions = predict_fn(test_data)['spikes']

    # generate a submission file
    with open(root_dir('data/submission/%s/%d.csv' % (model_type, submission_num)), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Id'] + columns)
        for i in range(len(test_data['image'])):
            writer.writerow([i] + list(predictions[i]))


if __name__ == '__main__':
    main()
