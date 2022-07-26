from abc import ABC, abstractmethod
from typing import Iterable

from settings import LogDir
from utils.log_helper import create_logger


class InferenceInterface(ABC):
    def __init__(
            self, dataset: Iterable[str],
            model_name: str, model_version: int,
            url: str):
        self.model_name = model_name
        self.model_version = str(model_version)
        self.dataset = dataset
        self.url = url
        self.logger = create_logger(LogDir.inference)

    @abstractmethod
    def preprocess(self):
        """tokenize the input dataset, or preprocess the dataset"""
        pass

    @abstractmethod
    def init_service(self):
        """set up the triton client"""
        pass

    @abstractmethod
    def run(self):
        """execute the inference, return probabilities with labels"""
        pass
