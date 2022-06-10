from datetime import datetime

from loguru import logger

from metrics.nultilabel_metrics import compute_metrics
from model_class.model_interface import ModelWorker
from preprocess.raw_data_preprocess.preprocess import build_dataset
from settings import LogDir, LogVar, MODEL_CLASS, DEVICE

from utils.log_helper import get_log_name
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


class BertModelWorker(ModelWorker):
    def __init__(
            self, dataset_name: str, model_name: str,
            n_sample: int, is_trainer: int, epoch: int,
            batch_size: int, weight_decay: float):
        super().__init__(dataset_name, model_name, n_sample)
        # self.dataset_name = dataset_name
        # self.model_name = model_name
        # self.n_sample = n_sample
        self.is_trainer = is_trainer
        self.epoch = epoch
        self.batch_size = batch_size
        self.weight_decay = weight_decay

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
            # todo: add other training method without trainer, e.g. pytorch
            raise ValueError('trainer cannot be set to none now')

    def data_preprocess(self, model_ckpt):
        """add new data preprocess flow in build_dataset"""
        return build_dataset(self.dataset_name, model_ckpt, self.n_sample)


class MLModelWorker(ModelWorker):
    def __init__(self, dataset_name: str, model_name: str, n_sample: int):
        super().__init__(dataset_name, model_name, n_sample)

    def initialize_model(self):
        logger.debug('building model ...')
        model_class, _, model_ckpt = self.create_model_class()
        logger.info('preprocess the data ...')
        req_dict = self.data_preprocess(model_ckpt)
        pass

    def data_preprocess(self, model_ckpt):
        """add new data preprocess flow in build_dataset"""
        return build_dataset(self.dataset_name, model_ckpt, self.n_sample)
