from datetime import datetime
from typing import Union

from datasets import load_dataset
from loguru import logger

from settings import LogDir, LogVar
from tokenizers_class.tokenize import build_tokenizer
from utils.enum_helper import DatasetName
from utils.log_helper import get_log_name
from utils.preprocess_helper import custom_train_test_split, get_data_sample
from workers.datasets_builder.loader import create_data_loader

logger.add(
    get_log_name(LogDir.preprocess, datetime.now()),
    level=LogVar.level,
    format=LogVar.format,
    enqueue=LogVar.enqueue,
    diagnose=LogVar.diagnose,
    catch=LogVar.catch,
    serialize=LogVar.serialize,
    backtrace=LogVar.backtrace,
    colorize=LogVar.color
)


def build_dataset(
        dataset_name: str, token_name: str,
        batch_size: int, max_len: int,
        n_sample: Union[int, str],
        train_val_split_rate: float = 0.8,
        **kwargs
):
    """

    :param dataset_name: dataset name in settings and enum_helper.
    :param token_name: token name is equal to MODEL_CLASS keys name in settings.
    :param batch_size: training and validating batch_size.
    :param max_len: max length of training and validating.
    :param n_sample: sampling size
    :param train_val_split_rate: split rate of training and validating.
    :param kwargs:
    :return: training, validating DataLoader and n_labels
    """

    if dataset_name == DatasetName.go_emotion.value:
        logger.debug('loading go_emotion dataset ...')
        df = load_dataset("go_emotions", "raw")
        df = df['train'].to_pandas()
        logger.info(f"dataset shape: {df.shape}")

        version = kwargs.get('version')

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
            token_name=token_name, log=logger
        )
        logger.info('building loader...')
        train_loader = create_data_loader(
            df_train, dataset_name, tokenizer, max_len, batch_size
        )
        test_loader = create_data_loader(
            df_test, dataset_name, tokenizer, max_len, batch_size
        )

        return train_loader, test_loader, len(label_cols)
    else:
        error_message = f"dataset {dataset_name} is not found"
        logger.error(error_message)
        raise ValueError(error_message)
