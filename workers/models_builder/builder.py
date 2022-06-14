from datetime import datetime
from typing import Optional

from loguru import logger

from model_class.model_interface import ModelWorker
from preprocess.raw_data_preprocess.preprocess import build_dataset
from settings import LogDir, LogVar

from utils.log_helper import get_log_name

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


class BertModelWorker(ModelWorker):
    def __init__(
            self, dataset_name: str, model_name: str,
            n_sample: Optional[int], learning_rate: float, epoch: int, max_len: int,
            batch_size: int, train_val_split_rate: float = 0.8, version: str = "small"):
        super().__init__(dataset_name, model_name)
        self.epoch = epoch
        self.batch_size = batch_size
        self.max_len = max_len
        self.learning_rate = learning_rate
        self.n_sample = n_sample if n_sample else "all"
        self.version = version
        self.model_ckpt = self.create_model_ckpt()
        self.train_val_split_rate = train_val_split_rate
        self.train_loader, self.test_loader, self.n_labels = self.data_preprocess()
        self.model = self.initialize_model()

    def initialize_model(self):
        logger.debug('building model ...')
        if self.n_labels:
            model = self.create_model_class(self.n_labels)
            model.config.problem_type = "multi_label_classification"
            # logger.debug(f"{model.config}")
            return model

        else:
            err_msg = f"number of labels is missing"
            logger.error(err_msg)
            raise ValueError(err_msg)

    def create_model_class(self, num_labels):
        ckpt = self.model_ckpt
        if ckpt == 'bert-base-uncased':
            from transformers import BertForSequenceClassification
            model = BertForSequenceClassification.from_pretrained(
                ckpt, num_labels=num_labels
            )
        elif ckpt == 'xlnet-base-cased':
            from transformers import XLNetForSequenceClassification
            model = XLNetForSequenceClassification.from_pretrained(
                ckpt, num_labels=num_labels
            )
        elif ckpt == 'roberta-base':
            from transformers import RobertaForSequenceClassification
            model = RobertaForSequenceClassification.from_pretrained(
                ckpt, num_labels=num_labels)
        elif ckpt == 'albert-base-v2':
            from transformers import AlbertForSequenceClassification
            model = AlbertForSequenceClassification.from_pretrained(
                ckpt, num_labels=num_labels
            )
        elif ckpt == 'xlm-roberta-base':
            from transformers import XLMRobertaForSequenceClassification
            model = XLMRobertaForSequenceClassification.from_pretrained(
                ckpt, num_labels=num_labels
            )
        else:
            error_message = f"model info {ckpt} is not fount"
            logger.error(error_message)
            raise ValueError(error_message)

        return model

    def data_preprocess(self):
        return build_dataset(
            dataset_name=self.dataset_name,
            token_name=self.model_name,
            batch_size=self.batch_size,
            max_len=self.max_len,
            n_sample=self.n_sample,
            train_val_split_rate=self.train_val_split_rate,
            version=self.version
        )
