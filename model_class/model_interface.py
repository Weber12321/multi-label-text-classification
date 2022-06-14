from abc import ABC, abstractmethod

from settings import MODEL_CLASS


class ModelWorker(ABC):
    def __init__(
            self, dataset_name: str, model_name: str,
        ):
        self.dataset_name = dataset_name
        self.model_name = model_name

    def create_model_ckpt(self):
        """creating model class refer to settings"""
        if mod_ckpt := MODEL_CLASS.get(self.model_name):
            return mod_ckpt
        # if self.model_name in MODEL_CLASS:
        #     module_path, class_name = MODEL_CLASS[self.model_name]['model'].rsplit(sep='.', maxsplit=1)
        #     model_ckpt = MODEL_CLASS[self.model_name]['ckpt']
        #     return getattr(importlib.import_module(module_path), class_name), class_name, model_ckpt
        else:
            error_message = f"model {self.model_name} not found"
            raise ValueError(error_message)

    @abstractmethod
    def initialize_model(self):
        """initialize model"""
        pass

    @abstractmethod
    def data_preprocess(self):
        """add new data preprocess flow in build_dataset"""
        pass
