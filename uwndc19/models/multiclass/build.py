from uwndc19.dataset import load_data, get_train_datasets
from uwndc19.models.multiclass.input import build_serving_input_receiver_fn, build_train_input_fn, build_eval_input_fn
from uwndc19.models.multiclass.model import model_fn


def build_fn(config: dict):
    # load the data
    df, imgs = load_data()
    train_imgs, train_labels, train_nan_mask, eval_imgs, eval_labels, eval_nan_mask = \
        get_train_datasets(df, imgs, config['data']['eval_size'], config['model']['image_size'])

    print('Train size: %d, eval size: %d' % (len(train_labels), len(eval_labels)))

    # build input functions
    train_input_fn = build_train_input_fn(train_imgs, train_labels, train_nan_mask)
    eval_input_fn = build_eval_input_fn(eval_imgs, train_labels, train_nan_mask)
    serving_input_receiver_fn = build_serving_input_receiver_fn(config['model']['image_size'])

    return model_fn, train_input_fn, eval_input_fn, serving_input_receiver_fn
