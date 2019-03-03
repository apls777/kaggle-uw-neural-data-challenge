import tensorflow as tf
import numpy as np
from uwndc19.helpers.dataset import RANDOM_SEED


def build_serving_input_receiver_fn(image_size: int):
    def serving_input_receiver_fn():
        receiver_tensors = {
            'image': tf.placeholder(tf.float32, [None, image_size, image_size, 3], name='ph_image'),
            'neuron_id': tf.placeholder(tf.int32, [None], name='ph_neuron_id'),
        }
        return tf.estimator.export.ServingInputReceiver(receiver_tensors, receiver_tensors)

    return serving_input_receiver_fn


def build_input_fn(imgs, labels, batch_size=None, num_epochs=None, shuffle=False):
    neuron_ids = []
    new_imgs = []
    new_labels = []
    for i, img in enumerate(imgs):
        spikes = labels[i]
        for j, spike in enumerate(spikes):
            if not np.isnan(spike):
                neuron_ids.append(j)
                new_imgs.append(img)
                new_labels.append(spike)

    perm = np.random.RandomState(seed=RANDOM_SEED).permutation(len(new_labels))

    new_imgs = np.array(new_imgs, dtype=np.float32)[perm]
    neuron_ids = np.array(neuron_ids, dtype=np.int32)[perm]
    new_labels = np.array(new_labels, dtype=np.float32)[perm]

    features = {
        'image': new_imgs,
        'neuron_id': neuron_ids,
    }

    # don't shuffle if a batch is a whole dataset, otherwise several epochs will be mixed together
    shuffle = shuffle and bool(batch_size)
    batch_size = batch_size if batch_size else len(new_labels)

    return tf.estimator.inputs.numpy_input_fn(x=features, y=new_labels, batch_size=batch_size,
                                              num_epochs=num_epochs, shuffle=shuffle)
