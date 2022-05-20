import os
from dotenv import load_dotenv
from loguru import logger

from preprocess.data_preprocess import sqlite_fetch, str_decode, save_list_dict_csv

if __name__ == '__main__':

    load_dotenv('preprocess.env')

    logger.info("fetching the data")
    dataset = sqlite_fetch(
        db_path=os.getenv('DB'),
        file_path=os.getenv('SQL_FILE'),
        _decode=True
    )
    logger.info(f"The data length is {len(dataset)}, saving in {type(dataset)}")
    logger.info(f"printing the samples ...")

    for i in dataset[0:9]:
        print(i)

    save_list_dict_csv(data=dataset, file_name="question_tag.csv")

