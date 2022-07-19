from typing import List

import numpy as np
import tritonclient.http as httpclient

from interface.inference_interface.bert_inference_interface import BertInferenceInterface
from utils.train_helper import sigmoid_logits_to_one_hot


class BertInferenceWorker(BertInferenceInterface):
    def __init__(
            self, dataset: List[str], model_name: str,
            model_version: int, url: str, backend: str,
            max_len: int, chunk_size: int,
            label_cols: List[str] = None
    ):

        super().__init__(
            dataset, model_name, model_version,
            url, backend, max_len, chunk_size
        )
        self.label_cols = label_cols

    def run(self):
        self.logger.info('Preparing dataset ...')
        input_dataset = self.preprocess()

        self.logger.info('Initializing connection ...')
        triton_client, inputs, outputs = self.init_service()

        self.logger.info(triton_client)

        # for data in input_dataset:
        # inputs[0].set_data_from_numpy(data[0].reshape(32, 100), binary_data=False)
        # inputs[1].set_data_from_numpy(data[1].reshape(32, 100), binary_data=False)

        inputs[0].set_data_from_numpy(input_dataset['input_ids'], binary_data=False)
        inputs[1].set_data_from_numpy(input_dataset['attention_mask'], binary_data=False)

        outputs.append(httpclient.InferRequestedOutput(
            self.output_name, binary_data=False
        ))

        response = triton_client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=inputs,
            outputs=outputs
        )

        logits = response.as_numpy(self.output_name)
        logits = np.asarray(logits, dtype=np.float32)

        sigmoid_vectorization = np.vectorize(self.sigmoid)

        sig_logits = sigmoid_vectorization(logits)

        return sig_logits

        # return sigmoid_logits_to_one_hot(sig_logits)
        # return results

    @staticmethod
    def sigmoid(x):
        z = np.exp(-x)
        sig = 1 / (1 + z)
        return sig



