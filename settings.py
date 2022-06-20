import os
from datetime import date

import torch
from dotenv import load_dotenv
from dataclasses import dataclass
from pydantic import BaseSettings, BaseModel

load_dotenv('.env')

DEBUG: bool = True if os.getenv('DEBUG') else False

DEVICE = torch.device(os.getenv('DEVICE'))


# logs
class LogDir:
    preprocess = 'preprocess'
    model = 'model'
    training = 'training'
    api = 'api'


class LogVar:
    level = 'DEBUG' if DEBUG else 'INFO'
    format = '{time:HH:mm:ss.SS} | {level} | {message}'
    color = True
    serialize = False  # True if you want to save it as json format to NoSQL db
    enqueue = True
    catch = True if DEBUG else False
    backtrace = True if DEBUG else False
    diagnose = True if DEBUG else False
    rotation = '00:00'


# celery
class CeleryConfig(BaseSettings):
    CELERY_NAME: str = 'celery_app'
    CELERY_BROKER: str = 'redis://localhost'
    CELERY_BACKEND: str = 'db+sqlite:///save.db'
    CELERY_TIMEZONE: str = 'Asia/Taipei'
    CELERY_ENABLE_UTC: bool = False
    CELERY_TASK_TRACK_STARTED: bool = True
    CELERY_ACKS_LATE: bool = True


# api
class APIConfig(BaseSettings):
    API_HOST = '127.0.0.1'
    API_VERSION = 1.0
    API_TITLE = 'MultiLabel Text Training Experiment API'


class PostTaskData(BaseModel):
    # DATASET_NAME: DatasetName = DatasetName.go_emotion
    # MODEL_NAME: ModelName = ModelName.distil_bert
    N_SAMPLE: int = 1000
    EPOCH: int = 10
    MAX_LEN: int = 50
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 5e-5
    SPLIT_RATE: float = 0.8
    VERSION: str = 'small'


class AutoAnnotation(BaseModel):
    START_TIME: date = "2020-01-01"
    END_TIME: date = "2021-01-01"


# dataset class
# small is less classes version of dataset
DATA_CLASS = {
    "go_emotion": "datasets_class.go_emotion.GoEmotionDataset",
}

# models
MODEL_CLASS = {
    # "distil_bert": {
    #     "ckpt": "distilbert-base-uncased",
    #     "model": "model_class.distilbert.DistilBertForMultilabelSequenceClassification"
    # },
    "bert_base": "bert-base-uncased",
    "XLNet": "xlnet-base-cased",
    "roberta": "roberta-base",
    "albert": "albert-base-v2",
    "XLM_roberta": "xlm-roberta-base",
    "rule_model": "rule_model"
}

# database config for saving training results
if DEBUG:
    DATABASE_URL = "sqlite:///training.db"
else:
    DATABASE_URL = f'mysql+pymysql://{os.getenv("USER")}:' \
                   f'{os.getenv("PASSWORD")}@{os.getenv("HOST")}:' \
                   f'{os.getenv("PORT")}/{os.getenv("SCHEMA")}?charset=utf8mb4'


# database config for scraping data
@dataclass
class DatabaseScrapConfig:
    host: str = os.getenv('A_HOST')
    port: int = int(os.getenv('A_PORT'))
    user: str = os.getenv('A_USER')
    password: str = os.getenv('A_PASSWORD')
    charset: str = 'utf8mb4'


CUSTOMIZE_SQL = None

RULE_FILE_EXT = '.json'

EXCLUDE_WORDS = [
    "魔羯", "山羊",
    "水瓶", "寶瓶",
    "雙魚",
    "牡羊", "白羊",
    "金牛",
    "雙子",
    "巨蟹",
    "獅子",
    "處女",
    "天秤",
    "天平",
    "天蠍",
    "射手"
]
