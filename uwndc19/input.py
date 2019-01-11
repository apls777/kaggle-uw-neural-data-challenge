import numpy as np
import pandas as pd
from uwndc19.utils import root_dir
import tensorflow as tf


def load_data():
    df = pd.read_csv(root_dir('data/train.csv'))
    stim = np.load(root_dir('data/stim.npy'))

    return df, stim


def get_column_data(column_name, df, stim):
    stim = stim[:, 16:-16, 16:-16]
    eval_imgs = stim[50:60]
    train_imgs = stim[60:]

    # get labels
    labels = np.array(df[column_name].tolist(), dtype=np.float32)
    eval_labels = labels[:10]
    train_labels = labels[10:]

    # remove NaN values from the training dataset
    filter_nans = np.logical_not(np.isnan(train_labels))
    train_imgs = train_imgs[filter_nans]
    train_labels = train_labels[filter_nans]

    # remove NaN values from the evaluation dataset
    filter_nans = np.logical_not(np.isnan(eval_labels))
    eval_imgs = eval_imgs[filter_nans]
    eval_labels = eval_labels[filter_nans]

    return train_imgs, train_labels, eval_imgs, eval_labels


def serving_input_receiver_fn():
    receiver_tensors = {
        'image': tf.placeholder(tf.float32, [None, 48, 48, 3], name='ph_image'),
    }

    return tf.estimator.export.ServingInputReceiver(receiver_tensors, receiver_tensors)
