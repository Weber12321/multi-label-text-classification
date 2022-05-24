import importlib
from datetime import datetime

from loguru import logger

from metrics.go_emotion import compute_metrics
from preprocess.raw_data_preprocess.preprocess import build_dataset
from settings import LogDir, LogVar, MODEL_CLASS, DEVICE

from utils.enum_helper import DatasetName
from utils.log_helper import get_log_name
from workers.build_models.go_emotion import go_emotion_model
from workers.build_trainers.train import setup_trainer

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


class BertModelWorker:
    def __init__(
            self, dataset_name: str, model_name: str,
            n_sample: int, is_trainer: int, epoch: int,
            batch_size: int, weight_decay: float):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.n_sample = n_sample
        self.is_trainer = is_trainer
        self.epoch = epoch
        self.batch_size = batch_size
        self.weight_decay = weight_decay

    def create_model_class(self):
        if self.model_name in MODEL_CLASS:
            module_path, class_name = MODEL_CLASS[self.model_name]['model'].rsplit(sep='.', maxsplit=1)
            model_ckpt = MODEL_CLASS[self.model_name]['ckpt']
            return getattr(importlib.import_module(module_path), class_name), class_name, model_ckpt
        else:
            error_message = f"model {self.model_name} not found"
            logger.error(error_message)
            raise ValueError(error_message)

    def initialize_model(self):
        logger.debug('building model ...')
        model_class, _, model_ckpt = self.create_model_class()

        logger.info('preprocess the data ...')
        req_dict = self.data_preprocess(model_ckpt)

        model = model_class.from_pretrained(model_ckpt, num_labels=req_dict.get('n_labels')).to(DEVICE)
        model.config.id2label = req_dict.get('id2label', None)
        model.config.label2id = req_dict.get('label2id', None)
        logger.info(f"{model.config}")

        if self.is_trainer == 1:
            logger.info('setting up the trainer ...')
            trainer, args = setup_trainer(
                model,
                compute_metrics,
                req_dict.get('tokenizer'),
                req_dict.get('train_dataset'),
                req_dict.get('test_dataset'),
                self.epoch,
                self.batch_size,
                self.weight_decay
            )
            return trainer, args
        else:
            # todo: add other training flow
            pass

    def data_preprocess(self, model_ckpt):
        return build_dataset(self.dataset_name, model_ckpt, self.n_sample)


