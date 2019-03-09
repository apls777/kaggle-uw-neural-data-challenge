from pprint import pprint
import tensorflow as tf
from tensorflow.python.platform import tf_logging
from uwndc19.core.utils import root_dir
from uwndc19.helpers.dataset import load_data, get_test_dataset


def main():
    # enable TensorFlow logging
    tf.logging.set_verbosity(tf.logging.INFO)
    tf_logging._get_logger().propagate = False  # fix double messages

    # directory with the exported model
    saved_model_dir = root_dir('export/final_model')

    # image size that the model accepts
    image_size = 48

    # load the images from the dataset
    _, imgs = load_data()

    # get test images and crop them to the right size
    imgs = get_test_dataset(imgs, image_size)

    # load the model
    predict_fn = tf.contrib.predictor.from_saved_model(saved_model_dir)

    # get predictions
    res = predict_fn({'image': imgs})

    # print predicted spikes
    pprint(res['spikes'])


if __name__ == '__main__':
    main()
