import os
from dotenv import load_dotenv
from loguru import logger

from preprocess.sql_preprocess.preprocess import sqlite_fetch, save_list_dict_csv

if __name__ == '__main__':

    logger.add('logs/2022-05-23.log')
    logger.info('Hello')
    logger.info('Bye')

