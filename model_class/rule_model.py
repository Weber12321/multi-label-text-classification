import re
from datetime import datetime
from typing import Dict, List, Iterable

from loguru import logger

from model_class.model_interface import ModelWorker
from settings import LogDir, LogVar
from utils.log_helper import get_log_name
from utils.preprocess_helper import read_rule_json


class RuleModelWorker(ModelWorker):
    def __init__(
            self, filename: str = None, dataset_name: str = None,
            model_name: str = "rule_model", case_sensitive: bool = False):
        super().__init__(dataset_name, model_name)
        self.rules = None
        self.labels = []
        self.logger = logger
        self.case_sensitive = case_sensitive
        self.config_logger()
        if filename:
            self.load(filename)

    def initialize_model(self):
        pass

    def data_preprocess(self):
        pass

    def load(self, filename):
        self.rules: Dict[str, List[str]] = read_rule_json(filename)
        self.labels = [label for label in self.rules.keys()]
        self.logger.debug(f"label size: {len(self.labels)}")

    def predict(self, input_examples: Iterable[str]):
        x = parse_predict_target(input_examples=input_examples,
                                 case_sensitive=self.case_sensitive)
        matched_labels = []
        match_count_list = []
        for _predict_str in x:
            _match_count_list = []
            _matched_labels = []
            for label, patterns in self.rules.items():
                _matched_count = 0
                _match_pattern = []
                for pattern in patterns:
                    # logger.info(pattern)
                    if result := re.search(pattern=pattern, string=_predict_str.lower()):
                        _matched_count += 1
                        _match_pattern.append(result.group(0))

                if _matched_count > 0:
                    _match_count_list.append((label, _match_pattern))
                    _matched_labels.append(label)

            if len(_matched_labels) > 0:
                matched_labels.append(tuple(_matched_labels))
                match_count_list.append(_match_count_list)

        self.logger.debug(f"Matched labels size: {len(matched_labels)}")
        self.logger.debug(f"Matched count list size: {len(match_count_list)}")
        if len(matched_labels) > 0:
            return matched_labels, match_count_list
        else:
            return None, None

    def config_logger(self):
        self.logger.add(
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


def parse_predict_target(input_examples: Iterable[str],
                         case_sensitive: bool = False) -> List[str]:
    return [_parse_predict_target(input_example, case_sensitive=case_sensitive)
            for input_example in input_examples]


def _parse_predict_target(input_example: str,
                          case_sensitive: bool = False) -> str:
    return input_example if case_sensitive else input_example.lower()
