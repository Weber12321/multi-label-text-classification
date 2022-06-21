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
            self,
            filename: str = None,
            dataset_name: str = None,
            model_name: str = "rule_model",
            case_sensitive: bool = False,
            multi_output_threshold: int = 0,
            sub_set_keep: List[str] = None
    ):
        super().__init__(dataset_name, model_name)
        self.filename = filename
        self.rules = None
        self.labels = []
        self.logger = logger
        self.case_sensitive = case_sensitive
        self.multi_output_threshold = multi_output_threshold
        self.sub_set_keep = sub_set_keep
        self.config_logger()

    def initialize_model(self):
        if not self.filename:
            raise ValueError(
                'Missing input_examples or rule file name'
            )
        self.load_rules()

    def data_preprocess(self):
        pass

    def load_rules(self):
        self.rules: Dict[str, List[str]] = read_rule_json(
            self.filename, sub_set_keep=self.sub_set_keep
        )
        self.labels = [label for label in self.rules.keys()]
        self.logger.debug(f"label size: {len(self.labels)}")
        # self.logger.debug(f"{self.rules}")

    def predict(self, input_examples: Iterable[str]):
        examples = parse_predict_target(
            input_examples=input_examples,
            case_sensitive=self.case_sensitive
        )
        self.logger.debug(f"input examples size: {len(examples)}")
        if not self.rules:
            raise ValueError(
                'please perform initialize_model before predicting'
            )
        x = examples
        # matched_labels = []
        # match_count_list = []
        results = []
        for _predict_str in x:
            _match_count_list = []
            _matched_labels = []
            for label, patterns in self.rules.items():

                if not patterns:
                    continue

                _matched_count = 0
                _match_pattern = []
                for pattern in patterns:
                    if result := re.search(pattern=pattern, string=_predict_str.lower()):
                        _matched_count += 1
                        _match_pattern.append(result.group(0))

                if _matched_count > 0:
                    _match_count_list.append((label, _match_pattern))
                    _matched_labels.append(label)

            if len(_matched_labels) > self.multi_output_threshold:
                # matched_labels.append(tuple(_matched_labels))
                # match_count_list.append(_match_count_list)
                results.append(
                    {
                        'text': _predict_str,
                        'label': _matched_labels
                    }
                )

        return results
        # self.logger.debug(f"Matched labels size: {len(matched_labels)}")
        # self.logger.debug(f"Matched count list size: {len(match_count_list)}")
        # if len(matched_labels) > 0:
        #     return matched_labels, match_count_list
        # else:
        #     return None, None

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
