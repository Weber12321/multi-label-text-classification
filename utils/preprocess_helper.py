import contextlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import pymysql

from loguru import logger

from definition import DATA_DIR
from settings import DatabaseScrapConfig, RULE_FILE_EXT


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
    filename = filename + RULE_FILE_EXT
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r') as f:
        rules = json.load(f)
    return rules


def read_dataset_from_db(
        db: str,
        sql_statement = None,
        start: datetime = None,
        end: datetime = None,
        interval: timedelta = timedelta(hours=4),
        char_length: int = 500,
        filter_word_list: List[str] = None
):
    """
    :param filter_word_list: words pattern you wanna exclude from retrieval data
    :param char_length: int, max len of retrieval data
    :param db: dataset name
    :param sql_statement: custom sql statement
    :param start: start datetime of scraping
    :param end: end datetime of scraping
    :param interval: timedelta, default is four hours
    :return: dataset
    """

    if sql_statement:
        with get_connection(db=db) as conn:
            cursor = conn.cursor()
            cursor.execute(sql_statement)
            results = cursor.fetchall()
            output = [result.get('content') for result in results]
            cursor.close()
            return output

    elif start and end:
        while start + interval < end:
            temp_end = start + interval
            with get_connection(db=db) as conn:
                cursor = conn.cursor()
                query = get_timedelta_query(
                    start=start,
                    end=temp_end,
                    char_length=char_length,
                    filter_word_list=filter_word_list
                )
                cursor.execute(query)
                results = cursor.fetchall()
                output = [result.get('content') for result in results]
                yield output, start
                start += interval
                cursor.close()

    else:
        return None


@contextlib.contextmanager
def get_connection(db: str):
    """yields connection"""
    _config = asdict(DatabaseScrapConfig())
    _config.update({'db': db})
    _config.update({'cursorclass': pymysql.cursors.DictCursor})

    conn = pymysql.connect(**_config)

    try:
        yield conn
    finally:
        conn.close()


def get_timedelta_query(
        start: datetime, end: datetime,
        table_name: str = "ts_page_content",
        char_length: int = 500,
        filter_word_list: List[str] = None
):
    if filter_word_list:
        filter_char = "%' AND content NOT LIKE '%".join(filter_word_list)
        filter_char = " content NOT LIKE " + "'%" + filter_char + "%'"

        query = f"""
        SELECT content 
        FROM {table_name} 
        WHERE post_time >= '{start}' 
        AND post_time <= '{end}' 
        AND content IS NOT NULL 
        AND char_length(content) < {char_length} 
        AND {filter_char};"""
    else:
        query = f"""
        SELECT content 
        FROM {table_name} 
        WHERE post_time >= '{start}' 
        AND post_time <= '{end}' 
        AND content IS NOT NULL 
        AND char_length(content) < {char_length};"""

    return query
