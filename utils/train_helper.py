import json
import os.path
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config.definition import DATA_DIR, MODEL_PT_DIR


def sigmoid_logits_to_one_hot(arr: np.array, thresh=0.5):
    arr[arr > thresh] = 1
    arr[arr <= thresh] = 0
    return arr.astype(int)


def one_hot_to_label(arr, label_col):
    temp = []
    for i in range(len(arr)):
        if arr[i] == 1:
            temp.append(label_col[i])
    return temp


# https://huggingface.co/docs/transformers/serialization#using-torchscript-in-python
def get_dummy_input(data, tokenizer, max_len, device):
    encoding = tokenizer.encode_plus(
        data,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )

    return encoding['input_ids'].to(device), encoding['attention_mask'].to(device)


def load_dataset(file_name, label_file_name, test_size=0.2, random_state=42):
    df_path = os.path.join(DATA_DIR / file_name)
    label_path = os.path.join(DATA_DIR / label_file_name)

    df = pd.read_json(df_path)

    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    with open(label_path) as f:
        labels = json.load(f)

    return df_train, df_test, labels


def create_model_dir(model_name: str):
    AUDIENCE_BERT_DIR = Path(MODEL_PT_DIR / model_name)
    Path(AUDIENCE_BERT_DIR).mkdir(exist_ok=True)
    return AUDIENCE_BERT_DIR
