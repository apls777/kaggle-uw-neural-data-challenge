import numpy as np
import pandas as pd
from uwndc19.utils import root_dir


def load_data():
    df = pd.read_csv(root_dir('data/train.csv'))
    imgs = np.load(root_dir('data/stim.npy'))

    return df, imgs


def get_datasets(df, imgs, eval_size):
    imgs = imgs[50:, 16:-16, 16:-16]

    # get labels and the mask for NaN values
    labels = df[df.columns[1:]].values.astype(np.float32)
    nan_mask = np.logical_not(np.isnan(labels))

    # replace NaN values with 0
    labels = np.nan_to_num(labels)

    # shuffle the dataset
    perm = np.random.RandomState(seed=112233).permutation(len(labels))
    imgs = imgs[perm]
    labels = labels[perm]
    nan_mask = nan_mask[perm]

    # evaluation dataset
    eval_imgs = imgs[:eval_size]
    eval_labels = labels[:eval_size]
    eval_nan_mask = nan_mask[:eval_size]

    # training dataset
    train_imgs = imgs[eval_size:]
    train_labels = labels[eval_size:]
    train_nan_mask = nan_mask[eval_size:]

    return train_imgs, train_labels, train_nan_mask, eval_imgs, eval_labels, eval_nan_mask


def get_column_datasets(column_name, df, imgs, eval_size):
    imgs = imgs[50:, 16:-16, 16:-16]

    # get labels
    labels = np.array(df[column_name].tolist(), dtype=np.float32)

    # remove NaN values from the dataset
    filter_nans = np.logical_not(np.isnan(labels))
    labels = labels[filter_nans]
    imgs = imgs[filter_nans]

    # shuffle the dataset
    perm = np.random.RandomState(seed=112233).permutation(len(labels))
    labels = labels[perm]
    imgs = imgs[perm]

    # evaluation dataset
    eval_labels = labels[:eval_size]
    eval_imgs = imgs[:eval_size]

    # training dataset
    train_labels = labels[eval_size:]
    train_imgs = imgs[eval_size:]

    return train_imgs, train_labels, eval_imgs, eval_labels
