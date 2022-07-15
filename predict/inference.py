import os
from typing import List

import tritonclient.grpc as grpcclient
from transformers import AutoTokenizer

from config.definition import MODEL_DIR
from data_set.data_loader import create_data_loader_pred


input_name = ['input__0', 'input__2']
output_name = 'output__0'


def run_inference(
        model_name: str,
        version: int,
        dataset: List[str],
        max_len: int,
        batch_size: int = 32):
    # preprocess
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR / f"tokenizer/{version}/"))
    pred_loader = create_data_loader_pred(dataset, tokenizer, max_len)

    input_name = ['input__0', 'input__1']
    output_name = 'output__0'
    model_name = 'audience_bert'

    triton_client = grpcclient.InferenceServerClient(url='triton:8001', verbose=False)

    model_metadata = triton_client.get_model_metadata(
        model_name=model_name,
        model_version=f"{version}"
    )
    model_config = triton_client.get_model_config(
        model_name=model_name,
        model_version=f"{version}"
    )


    ...
