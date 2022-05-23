import os

from dotenv import load_dotenv

from definition import LOGS_DIR

load_dotenv('.env')


# logs
class LogDir:
    preprocess = 'preprocess'
    model = 'model'


class LogVar:
    log_level = 'DEBUG' if os.getenv('DEBUG') else 'INFO'
    log_format = '{time} {level} {message}'



