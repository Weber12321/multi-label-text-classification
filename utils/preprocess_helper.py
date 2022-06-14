import json
import os

import numpy as np
import pandas as pd

from loguru import logger

from definition import DATA_DIR


def get_data_sample(df: pd.DataFrame, log: logger, n_sample: int = 500):
    df_sample = df.sample(n=n_sample)
    log.info(f"sample data size: {df_sample.sample}")
    return df_sample


def custom_train_test_split(df: pd.DataFrame, log: logger, rate: float = 0.8):
    # create train / test splits
    log.debug(f"splitting rate: {rate}")
    mask = np.random.rand(len(df)) < rate
    df_train = df[mask]
    df_test = df[~mask]
    log.debug(f"size of training set {df_train.shape}; size of testing set {df_test.shape}")
    return df_train, df_test


def read_rule_json(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r') as f:
        rules = json.load(f)
    return rules


# todo: create a db data extracting function
def read_dataset_from_db():
    pass

