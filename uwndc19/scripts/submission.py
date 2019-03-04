import argparse
import csv
import os
from shutil import copyfile, copytree
import tensorflow as tf
from tensorflow.contrib import predictor
from tensorflow.python.platform import tf_logging
from uwndc19.core.config import load_model_config, get_model_config_path, CONFIG_FILENAME
from uwndc19.helpers.dataset import load_data, get_test_dataset
from uwndc19.core.utils import root_dir


def generate_submission(model_type, model_dir, submission_num):
    submission_dir = root_dir('data/submission/%s/%s' % (model_type, submission_num))
    if os.path.isdir(submission_dir):
        raise ValueError('Submission #%d already exists' % submission_num)

    os.makedirs(submission_dir)

    # load the data
    df, stim = load_data()
    columns = list(df.columns)[1:]
    config = load_model_config(model_dir)
    test_data = {'image': get_test_dataset(stim, config['model']['image_size'])}

    # create the predictor
    export_dir = root_dir(os.path.join(model_dir, 'export', 'best'))
    latest_model_subdir = sorted(os.listdir(export_dir), reverse=True)[0]
    latest_model_dir = os.path.join(export_dir, latest_model_subdir)

    # get predictor
    predict_fn = predictor.from_saved_model(latest_model_dir)

    # get predictions
    predictions = predict_fn(test_data)['spikes']

    # generate a submission file
    with open(os.path.join(submission_dir, 'submission_%d.csv' % submission_num), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Id'] + columns)
        for i in range(len(test_data['image'])):
            writer.writerow([i] + list(predictions[i]))

    # copy config file
    config_path = get_model_config_path(model_dir)
    copyfile(config_path, os.path.join(submission_dir, CONFIG_FILENAME))

    # copy the model
    copytree(latest_model_dir, os.path.join(submission_dir, 'model'))


def main():
    # enable TensorFlow logging
    tf.logging.set_verbosity(tf.logging.INFO)
    tf_logging._get_logger().propagate = False  # fix double messages

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--model-type', type=str, default='multiclass', help='Model type')
    parser.add_argument('-d', '--model-dir', type=str, required=True, help='Model\'s directory')
    parser.add_argument('-n', '--submission-num', type=int, required=True, help='Submission number')

    args = parser.parse_args()

    # generate submission and copy the model
    generate_submission(args.model_type, args.model_dir, args.submission_num)


if __name__ == '__main__':
    main()
