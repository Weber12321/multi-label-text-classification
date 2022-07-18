from abc import ABC, abstractmethod
from typing import Iterable, Any

from interface.model_interface.model_interface import DeepModelInterface


class BertModelInterface(DeepModelInterface, ABC):
    def __init__(self, ckpt: str, epochs: int, learning_rate: float, batch_size: int,
                 dataset: Iterable[Any], optimizer: Any, max_len: int):
        super().__init__(epochs, learning_rate, batch_size, dataset, optimizer)
        self.max_len = max_len
        self.ckpt = ckpt

    @abstractmethod
    def preprocess(self):
        """preprocess"""
        pass

    @abstractmethod
    def init_model(self):
        """initialize model"""
        pass

    @abstractmethod
    def train(self):
        """training"""
        pass

    @abstractmethod
    def evaluate(self):
        """evaluating"""
        pass

    @abstractmethod
    def run(self):
        """executing the experiment"""
        pass
