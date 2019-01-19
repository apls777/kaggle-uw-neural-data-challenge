from uwndc19.dataset import load_data, get_neuron_train_datasets
from uwndc19.manager.abstract_builder import AbstractBuilder
from uwndc19.models.neuron_emb.input import build_input_fn, build_serving_input_receiver_fn
from uwndc19.models.neuron_emb.model import model_fn


class Builder(AbstractBuilder):

    def __init__(self, config: dict):
        super().__init__(config)

        # load the data
        df, imgs = load_data()
        train_imgs, train_labels, eval_imgs, eval_labels = \
            get_neuron_train_datasets(df, imgs, config['data']['eval_size'], config['model']['image_size'])

        # build input functions
        batch_size = config['training'].get('batch_size')
        self._train_input_fn = build_input_fn(train_imgs, train_labels, batch_size=batch_size, num_epochs=None,
                                              shuffle=True)
        self._eval_input_fn = build_input_fn(eval_imgs, eval_labels, num_epochs=1)
        self._serving_input_receiver_fn = build_serving_input_receiver_fn(config['model']['image_size'])

    def build_model_fn(self):
        return model_fn

    def build_train_input_fn(self):
        return self._train_input_fn

    def build_eval_input_fn(self):
        return self._eval_input_fn

    def build_serving_input_receiver_fn(self):
        return self._serving_input_receiver_fn
