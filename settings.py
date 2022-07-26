import os

from pydantic import BaseSettings
from dotenv import load_dotenv

load_dotenv('.env')
DEBUG: bool = True if os.getenv('LEVEL') == 'DEBUG' else False

MODEL_CKPT = {
    'bert-chinese': 'bert-base-chinese',
    'chinese-bert-wwm': 'hfl/chinese-bert-wwm-ext',
    'chinese-roberta-wwm': 'hfl/chinese-roberta-wwm-ext'
}

# corresponds to the config.pbtxt inside the model-repo
INFERENCE_TYPE = {
    'pytorch': {
        'bert': {
            'input_name': [
                'input__0', 'input__1'
            ],
            'output_name': 'output__0',
        },
    },
}


class LogDir:
    training = 'training'
    inference = 'inference'


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


class TrainingFileName:
    dataset = 'au_2234_p.json'
    labels = 'labels.json'


class ServerConfig(BaseSettings):
    celery_name: str = 'bert_celery_app'
    celery_broker: str = 'redis://redis:6379'
    celery_backend: str = 'db+sqlite:///save.db'
    celery_queue: str = 'deep_model'
    celery_result_expires: int = 60 * 60 * 24
    celery_task_track_started: bool = True
    celery_model_ckpt: str = MODEL_CKPT.get('chinese-bert-wwm')
    inference_url: str = 'localhost:8000'
    inference_backend: str = 'pytorch'

    class Config:
        case_sensitive = False
        env_file = '.env'








