from typing import List

import numpy as np
from loguru import logger
import tritonclient.grpc as grpcclient

from interface.inference_interface.bert_inference_interface import BertInferenceInterface


class BertInferenceWorker(BertInferenceInterface):
    def __init__(
            self, dataset: List[str], model_name: str,
            model_version: int, url: str, backend: str,
            max_len: int, label_cols: List[str]):
        super().__init__(
            dataset, model_name, model_version, url, backend, max_len
        )
        self.label_cols = label_cols

    @logger.catch
    def run(self):
        self.logger.info('Preparing dataset ...')
        input_dataset = self.preprocess()

        self.logger.info('Initializing connection ...')
        triton_client, inputs, outputs = self.init_service()

        for data in input_dataset:
            inputs[0].set_data_from_numpy(data[0])
            inputs[1].set_data_from_numpy(data[1])

            outputs.append(grpcclient.InferRequestedOutput(self.output_name))

            response = triton_client.async_infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=inputs,
                outputs=outputs
            )

            logits = response.as_numpy(self.output_name)
            logits = np.asarray(logits, dtype=np.float32)

    @staticmethod
    def sigmoid(x):
        z = np.exp(-x)
        sig = 1 / (1 + z)

        return sig

    @staticmethod
    def logits_to_labels(arr, tresh=0.5): ...
        # TODO: create a logits to label function



