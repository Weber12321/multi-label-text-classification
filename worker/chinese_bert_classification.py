import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Any, Dict, List

from loguru import logger
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AdamW, get_scheduler
from transformers import logging as hf_logging

from config.definition import TOKEN_DIR
from data_set.data_loader import create_data_loader_from_dict
from interface.bert_model_interface import BertModelInterface
from train.evaluate_flow import eval_epoch
from train.train_flow import train_epoch
from utils.enum_helper import DatasetType
from utils.exception_helper import PreprocessError
from utils.train_helper import create_tokenizer_dir, get_dummy_input


class ChineseBertClassification(BertModelInterface):
    def __init__(self, max_len: int, ckpt: str, epochs: int, learning_rate: float,
                 batch_size: int, dataset: Iterable[Dict[str, str]],
                 label_col: List[str], model_name: str, model_version: int,
                 optimizer: Any, do_lower_case=True, is_multi_label=True):

        super().__init__(ckpt, epochs, learning_rate, batch_size, dataset,
                         optimizer, max_len)
        self.max_len = max_len
        self.ckpt = ckpt
        self.label_col = label_col
        self.model_name = model_name
        self.model_version = model_version
        self.do_lower_case = do_lower_case
        self.is_multi_label = is_multi_label
        self.num_labels = len(self.label_col)
        self.display_name = f"{self.model_name}_{self.model_version}"

        if self.is_multi_label:
            from torch.nn import BCEWithLogitsLoss
            self.loss_func = BCEWithLogitsLoss()
        else:
            from torch.nn import CrossEntropyLoss
            self.loss_func = CrossEntropyLoss()

    def preprocess(self):
        self.logger.debug(f"Creating tokenizer directory...")
        create_tokenizer_dir(self.model_name)

        self.logger.debug(f"Splitting datasets...")
        train_dataset, dev_dataset, test_dataset, dummy_input = dataset_split(self.dataset)

        if not train_dataset and not dev_dataset:
            self.logger.error('Train and dev dataset must not be empty')
            raise ValueError('Train and dev dataset must not be empty')

        tokenizer = AutoTokenizer.from_pretrained(self.ckpt, self.do_lower_case)

        self.dummy_input = get_dummy_input(
            dummy_input, tokenizer, self.max_len, device=self.device
        )

        self.logger.info('Creating data loaders...')

        self.train_loader = create_data_loader_from_dict(
            train_dataset, tokenizer, self.max_len, self.batch_size
        )
        self.dev_loader = create_data_loader_from_dict(
            dev_dataset, tokenizer, self.max_len, self.batch_size
        )

        if test_dataset:
            self.test_loader = create_data_loader_from_dict(
                test_dataset, tokenizer, self.max_len, self.batch_size
            )
        else:
            self.test_loader = None

        self.logger.info('Saving tokenizer...')
        save_model_token = Path(os.path.join(TOKEN_DIR / f"{self.model_version}/"))
        save_model_token.mkdir(exist_ok=True)
        tokenizer.save_pretrained(save_model_token)

    def init_model(self):
        try:
            self.preprocess()
        except Exception as e:
            self.logger.error(f"Something wrong when preprocessing data according to {e}")
            raise PreprocessError(f"Something wrong when preprocessing data according to {e}")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.ckpt,
            num_labels=self.num_labels,
            torchscript=True
        )
        if self.is_multi_label:
            self.model.config.problem_type = "multi_label_classification"

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            correct_bias=False
        )

        num_training_steps = self.epochs * len(self.train_loader)

        self.scheduler = get_scheduler(
            name='linear', optimizer=self.optimizer,
            num_warmup_steps=0, num_training_steps=num_training_steps
        )

        self.model.to(self.device)

    def train(self):
        return train_epoch(
            model=self.model,
            loader=self.train_loader,
            optimizer=self.optimizer,
            device=self.device,
            num_labels=self.num_labels,
            scheduler=self.scheduler,
            loss_func=self.loss_func
        )

    def evaluate(self):
        return eval_epoch(
            model=self.model,
            loader=self.dev_loader,
            device=self.device,
            num_labels=self.num_labels,
            loss_func=self.loss_func
        )

    def test(self):
        return eval_epoch(
            model=self.model,
            loader=self.test_loader,
            device=self.device,
            num_labels=self.num_labels,
            loss_func=self.loss_func
        )

    @logger.catch
    def run(self):
        hf_logging.set_verbosity_error()
        progress_bar = tqdm(range(self.epochs))
        pass




def dataset_split(dataset: Iterable[Dict[str, str]]):

    train_dataset = defaultdict(list)
    dev_dataset = defaultdict(list)
    test_dataset = defaultdict(list)

    for data in dataset:
        if data.get('dataset_type') == DatasetType.train.value:
            train_dataset['id'].append(data.get('id'))
            train_dataset['text'].append(data.get('content'))
            train_dataset['label'].append(data.get('label'))
        elif data.get('dataset_type') == DatasetType.dev.value:
            dev_dataset['id'].append(data.get('id'))
            dev_dataset['text'].append(data.get('content'))
            dev_dataset['label'].append(data.get('label'))
        else:
            test_dataset['id'].append(data.get('id'))
            test_dataset['text'].append(data.get('content'))
            test_dataset['label'].append(data.get('label'))

    dummy_input = max(train_dataset['text'])

    return train_dataset, dev_dataset, test_dataset, dummy_input


