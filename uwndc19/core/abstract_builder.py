from abc import ABC, abstractmethod


class AbstractBuilder(ABC):

    def __init__(self, config: dict):
        self._config = config

    @property
    def config(self):
        return self._config

    @abstractmethod
    def build_model_fn(self):
        raise NotImplementedError

    @abstractmethod
    def build_train_input_fn(self):
        raise NotImplementedError

    @abstractmethod
    def build_eval_train_input_fn(self):
        raise NotImplementedError

    @abstractmethod
    def build_eval_input_fn(self):
        raise NotImplementedError

    @abstractmethod
    def build_serving_input_receiver_fn(self):
        raise NotImplementedError
