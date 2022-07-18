import torch
from abc import ABC, abstractmethod
from typing import Iterable, Any

from config.settings import LogDir
from utils.log_helper import create_logger


class DeepModelInterface(ABC):
    def __init__(
            self, epochs: int, learning_rate: float,
            batch_size: int, dataset: Iterable[Any],
            optimizer: Any = None
    ):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dataset = dataset
        self.optimizer = optimizer
        self.device = torch.device("cuda") \
            if torch.cuda.is_available() \
            else torch.device("cpu")
        self.logger = create_logger(LogDir.training)

    @abstractmethod
    def preprocess(self):
        """data preprocessing"""
        pass

    @abstractmethod
    def init_model(self):
        """initialize model"""
        pass

    @abstractmethod
    def train(self):
        """training flow"""
        pass

    @abstractmethod
    def evaluate(self):
        """evaluating flow"""
        pass

    @abstractmethod
    def run(self):
        """run the experiment task"""
        pass

