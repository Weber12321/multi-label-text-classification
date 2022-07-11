import os
from datetime import datetime

from loguru import logger

from config.definition import LOGS_DIR
from config.settings import LogVar


def get_log_name(directory: str, file_name):
    string_file_name = f"{file_name}.log"
    return os.path.join(LOGS_DIR, directory, string_file_name)


def create_logger(log_dir):
    logger.add(
        get_log_name(log_dir, datetime.now().date()),
        level=LogVar.level,
        format=LogVar.format,
        enqueue=LogVar.enqueue,
        diagnose=LogVar.diagnose,
        catch=LogVar.catch,
        serialize=LogVar.serialize,
        backtrace=LogVar.backtrace,
        colorize=LogVar.color,
        encoding='utf-8'
    )

    return logger
