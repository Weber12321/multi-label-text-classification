from datetime import datetime

from loguru import logger

from settings import LogDir, LogVar
from utils.log_helpers import get_log_name


logger.add(get_log_name(LogDir.preprocess, datetime.now()),
           level=LogVar.log_level, format=LogVar.log_format)


