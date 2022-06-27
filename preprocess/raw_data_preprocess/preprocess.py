import os
from typing import Union

import pandas as pd
from datasets import load_dataset
from loguru import logger
from sklearn.model_selection import train_test_split

from definition import DATA_DIR
from metrics.nultilabel_metrics import label_transform
from tokenizers_class.tokenize import build_tokenizer
from utils.enum_helper import DatasetName
from utils.preprocess_helper import custom_train_test_split, get_data_sample
from workers.datasets_builder.loader import create_data_loader


def build_dataset(
        dataset_name: str, token_name: str,
        batch_size: int, max_len: int,
        n_sample: Union[int, str] = None,
        train_val_split_rate: float = 0.8,
        version: str = 'small'
):
    """

    :param version:
    :param dataset_name: dataset name in settings and enum_helper.
    :param token_name: token name is equal to MODEL_CLASS keys name in settings.
    :param batch_size: training and validating batch_size.
    :param max_len: max length of training and validating.
    :param n_sample: sampling size
    :param train_val_split_rate: split rate of training and validating.
    :return: training, validating DataLoader and n_labels
    """

    if dataset_name == DatasetName.go_emotion.value:
        logger.debug('loading go_emotion dataset ...')
        df = load_dataset("go_emotions", "raw")
        df = df['train'].to_pandas()
        logger.info(f"dataset shape: {df.shape}")

        if version == 'origin':
            label_cols = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
                          'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
                          'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
                          'relief', 'remorse', 'sadness', 'surprise', 'neutral']
        elif version == 'small':
            label_cols = ['neutral', 'approval', 'admiration', 'annoyance',
                          'gratitude', 'disapproval', 'curiosity', 'amusement']
        else:
            raise ValueError(f"dataset version {version} is unclear")

        logger.debug(f"dataset version {version}\nlength of labels: {len(label_cols)}")

        df["labels"] = df[label_cols].values.tolist()

        sample_label_idx = [i for i, row in enumerate(df['labels'].values) if sum(row) >= 1]
        df = df.iloc[sample_label_idx]
        df = df.drop_duplicates('text').reset_index(drop=True)
        logger.debug(f"dataset shape: {df.shape}")
        logger.debug(f"Unique comments: {df.text.nunique() == df.shape[0]}")
        logger.debug(f"Null values: {df.isnull().values.any()}")

        logger.info(f"max sentence length: {df.text.str.split().str.len().max()}")
        logger.info(f"average sentence length: {df.text.str.split().str.len().mean()}")
        logger.info(f"standard deviation sentence length: {df.text.str.split().str.len().std()}")

        # create id-label list
        # id2label = {str(i): label for i, label in enumerate(label_cols)}
        # label2id = {label: str(i) for i, label in enumerate(label_cols)}
        if isinstance(n_sample, int):
            df_sample = get_data_sample(df, logger, n_sample)
        else:
            df_sample = df

        df_train, df_test = custom_train_test_split(
            df_sample, logger, rate=train_val_split_rate
        )
        tokenizer = build_tokenizer(
            token_name=token_name
        )
        logger.info('building loader...')
        train_loader = create_data_loader(
            df_train, tokenizer, max_len, batch_size
        )
        test_loader = create_data_loader(
            df_test, tokenizer, max_len, batch_size
        )

        return train_loader, test_loader, len(label_cols)

    elif dataset_name == DatasetName.audience_tiny.value:
        data_path = os.path.join(DATA_DIR / "au_450.json")
        df = pd.read_json(data_path)
        logger.debug(df.head())

        label_col = ['男性', '女性', '已婚', '未婚', '上班族', '學生', '青年', '有子女']
        num_idx = list(range(0, 8))
        label_col_dict = dict(zip(label_col, num_idx))

        df = label_transform(df, label_col_dict, target='label')
        df[label_col] = pd.DataFrame(df.labels.tolist(), index=df.index)

        logger.debug(f"max sentence length: {df.text.str.split().str.len().max()}")
        logger.debug(f"average sentence length: {df.text.str.split().str.len().mean()}")
        logger.debug(f"stdev sentence length: {df.text.str.split().str.len().std()}")

        logger.debug(f"label counting")
        logger.debug(df[label_col].eq(1).sum())
        df_train, df_test = train_test_split(df, test_size=1-train_val_split_rate)

        tokenizer = build_tokenizer(token_name=token_name)

        train_loader = create_data_loader(df_train, tokenizer, max_len, batch_size)
        test_loader = create_data_loader(df_test, tokenizer, max_len, batch_size)

        return train_loader, test_loader, len(label_col)

    else:
        error_message = f"dataset {dataset_name} is not found"
        logger.error(error_message)
        raise ValueError(error_message)
