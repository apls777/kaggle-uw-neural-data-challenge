import csv
import os
import tensorflow as tf
from uwndc19.core.config import load_model_config
from uwndc19.core.estimator import get_predictor
from uwndc19.dataset import load_data, get_test_dataset
from uwndc19.core.utils import root_dir


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    submission_num = 18
    model_dir = 'training/multiclass/4conv-do04-d512-do04-l2'
    model_type = 'multiclass'

    # load the data
    df, stim = load_data()
    columns = list(df.columns)[1:]
    config = load_model_config(model_dir)
    test_data = {'image': get_test_dataset(stim, config['model']['image_size'])}

    # create the predictor
    export_dir = root_dir(os.path.join(model_dir, 'export', 'submission%d' % submission_num))
    predict_fn = get_predictor(export_dir)

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
