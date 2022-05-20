import os
import sqlite3 as sq
import pandas as pd

from loguru import logger
from pathlib import Path
from typing import Union, List

from definition import DATA_DIR, SQL_DIR


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


def str_decode(data: List[dict], logg=logger):
    new_list = []
    for idx, i in enumerate(data):
        try:
            new_dc = {
                'id': i.get('Id'),
                'text': i.get('Body').decode('utf-8'),
                'tag': i.get('Tag').decode('utf-8')
            }
            new_list.append(new_dc)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logg.error(e)
            logg.error(f"line {i.get('Id')} error text {i.get('Body')}")
            raise
    return new_list

def sqlite_fetch(db_path: Union[str, Path], file_path: Union[str, Path] = None,
                 logg=logger, _decode=False):
    """read and extract dataset from
    sqlite database with sql file"""

    db = os.path.join(DATA_DIR, db_path)
    if not db:
        err_msg = 'db path is missing'
        logg.error(err_msg)
        raise ValueError(err_msg)
    if not file_path:
        err_msg = 'SQL file is missing'
        logg.error(err_msg)
        raise ValueError(err_msg)
    else:
        sql_path = os.path.join(SQL_DIR, file_path)
        with open(sql_path, 'r') as f:
            sql_string = f.read()
            sql_string = sql_string.replace("\n", " ")

    logg.info(db)
    logg.info(sql_string)

    con = sq.connect(db)
    con.row_factory = dict_factory
    con.text_factory = bytes
    cur = con.cursor()
    dataset = cur.execute(sql_string).fetchall()
    con.close()

    return str_decode(dataset) if _decode else dataset


def save_list_dict_csv(data: List[dict], file_name: str):
    path = os.path.join(DATA_DIR, file_name)
    df = pd.DataFrame(data)
    df.to_csv(path, encoding='utf-8')

