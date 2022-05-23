import os
from datetime import datetime

from definition import LOGS_DIR


def get_log_name(directory: str, file_name):
    string_file_name = f"{file_name.strftime('%Y-%m-%d')}.log" \
        if isinstance(file_name, datetime) \
        else f"{file_name}.log"
    return os.path.join(LOGS_DIR, directory, string_file_name)

