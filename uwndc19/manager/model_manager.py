from importlib import import_module
import os
from tensorflow.contrib import predictor
from uwndc19.manager.config import read_config
from uwndc19.manager.estimator import train
from uwndc19.utils import root_dir


class ModelManager(object):

    def __init__(self, model_type: str, model_name: str):
        # load the model config
        self._config = read_config(model_type, model_name)

        # create the model builder
        builder_class = getattr(import_module('uwndc19.models.%s.builder' % model_type), 'Builder')
        self._builder = builder_class(self._config)

        # get the model directory
        self._model_dir = root_dir(os.path.join('training', model_type, model_name))

    def train(self):
        train(model_fn=self._builder.build_model_fn(),
              train_input_fn=self._builder.build_train_input_fn(),
              eval_input_fn=self._builder.build_eval_input_fn(),
              serving_input_receiver_fn=self._builder.build_serving_input_receiver_fn(),
              config=self._config,
              model_dir=self._model_dir)

    def get_predictor(self, export_dir):
        # find the latest version of the model
        export_model_dir = os.path.join(self._model_dir, export_dir)
        latest_model_subdir = sorted(os.listdir(export_model_dir), reverse=True)[0]
        latest_model_dir = os.path.join(export_model_dir, latest_model_subdir)

        # get the predictor
        predict_fn = predictor.from_saved_model(latest_model_dir)

        return predict_fn
