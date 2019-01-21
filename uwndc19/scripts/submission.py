import csv
import os
import tensorflow as tf
from uwndc19.dataset import load_data, get_test_dataset
from uwndc19.manager.config import read_config
from uwndc19.manager.model_manager import ModelManager
from uwndc19.utils import root_dir


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    submission_num = 13
    model_name = 'do04-d512-do04-sub-d128-do04'
    model_type = 'multiclass'

    # load the data
    df, stim = load_data()
    columns = list(df.columns)[1:]
    config = read_config(model_type, model_name)
    test_data = {'image': get_test_dataset(stim, config['model']['image_size'])}

    # create the predictor
    model_manager = ModelManager(model_type, model_name)
    export_dir = os.path.join('export', 'submission%d' % submission_num)
    predict_fn = model_manager.get_predictor(export_dir)

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
