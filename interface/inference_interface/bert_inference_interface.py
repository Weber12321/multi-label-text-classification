import os
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import tritonclient.grpc as grpcclient
from transformers import AutoTokenizer

from config.definition import TOKEN_DIR
from config.settings import INFERENCE_TYPE
from interface.inference_interface.inference_interface import InferenceInterface
from utils.exception_helper import BackendNotFoundError, ModelNotFoundError


class BertInferenceInterface(InferenceInterface, ABC):
    def __init__(
            self,
            dataset: List[str],
            model_name: str,
            model_version: int,
            url: str,
            backend: str,
            max_len: int,
            model_type: str = 'bert',
            verbose: bool = False
    ):
        super().__init__(dataset, model_name, model_version, url)

        if backend not in INFERENCE_TYPE.keys():
            raise BackendNotFoundError(f"Backend {backend} is not "
                                       f"set up in inference configuration")
        self.max_len = max_len
        self.backend = backend
        self.input_name = INFERENCE_TYPE[backend][model_type]['input_name']
        self.output_name = INFERENCE_TYPE[backend][model_type]['output_name']
        self.verbose = verbose

    def preprocess(self) -> List[List[np.array, np.array]]:
        token_path = os.path.join(TOKEN_DIR / self.model_name / self.model_version)

        try:
            tokenizer = AutoTokenizer.from_pretrained(token_path)
        except Exception:
            self.logger.error(f"Tokenizer files of model: {self.model_name} and "
                              f"version: {self.model_version} are not found.")

            raise ModelNotFoundError(f"Tokenizer files of model: {self.model_name} and "
                                     f"version: {self.model_version} are not found.")

        raw_dataset = []
        for data in self.dataset:
            input_ids, attention_mask = tokenizer(
                data, tokenizer, max_len=self.max_len
            )
            raw_dataset.append([input_ids, attention_mask])

        return raw_dataset

    def init_service(self):
        triton_client = grpcclient.InferenceServerClient(
            url=self.url, verbose=self.verbose
        )
        inputs = []
        outputs = []

        inputs.append(grpcclient.InferInput(
            self.input_name[0],
            [1, self.max_len],
            'INT64'
        ))
        inputs.append(grpcclient.InferInput(
            self.input_name[1],
            [1, self.max_len],
            'INT64'
        ))

        return triton_client, inputs, outputs

    @abstractmethod
    def run(self):
        """implement the inference"""
        pass


def tokenize(data, tokenizer, max_len) -> Tuple[np.array, np.array]:
    encoding = tokenizer.encode_plus(
        data,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        truncation=True
    )
    input_ids = np.array(encoding['input_ids'], dtype=np.int64)
    attention_mask = np.array(encoding['attention_mask'], dtype=np.int64)
    return input_ids, attention_mask
