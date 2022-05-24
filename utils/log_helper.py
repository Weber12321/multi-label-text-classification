import os
from datetime import datetime

from loguru import logger
from transformers import TrainerCallback

from definition import LOGS_DIR
from settings import LogDir, LogVar


def get_log_name(directory: str, file_name):
    string_file_name = f"{file_name.strftime('%Y-%m-%d')}.log" \
        if isinstance(file_name, datetime) \
        else f"{file_name}.log"
    return os.path.join(LOGS_DIR, directory, string_file_name)


class LoggerLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        logger.add(
            get_log_name(LogDir.model, datetime.now()),
            level=LogVar.level,
            format=LogVar.format,
            enqueue=LogVar.enqueue,
            diagnose=LogVar.diagnose,
            catch=LogVar.catch,
            serialize=LogVar.serialize,
            backtrace=LogVar.backtrace,
            colorize=LogVar.color
        )

        control.should_log = False
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            logger.info(logs)


