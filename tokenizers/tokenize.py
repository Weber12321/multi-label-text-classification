import pandas as pd
from loguru import logger
from transformers import AutoTokenizer


def build_tokenizer(
        token_name: str, df_train: pd.DataFrame, df_test: pd.DataFrame,
        context_col: str, label_col: str, log: logger):
    log.DEBUG(f"loading {token_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(token_name)
    log.DEBUG('tokenizing dataset')
    train_encodings = tokenizer(df_train[context_col].values.tolist(), truncation=True)
    test_encodings = tokenizer(df_test[context_col].values.tolist(), truncation=True)
    train_labels = df_train[label_col].values.tolist()
    test_labels = df_test[label_col].values.tolist()

    return train_encodings, test_encodings, train_labels, test_labels, tokenizer


