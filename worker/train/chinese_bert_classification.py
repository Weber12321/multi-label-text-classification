import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Any, Dict, List

import torch
from loguru import logger
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AdamW, get_scheduler
from transformers import logging as hf_logging

from config.definition import MODEL_BIN_DIR
from data_set.data_loader import create_data_loader_from_dict
from interface.model_interface.bert_model_interface import BertModelInterface
from train.evaluate_flow import eval_epoch
from train.predict_flow import get_classification_report, save_pt, get_model_size
from train.train_flow import train_epoch
from utils.enum_helper import DatasetType
from utils.exception_helper import PreprocessError, ModelNotFoundError
from utils.train_helper import create_tokenizer_dir, get_dummy_input, create_model_dir


class ChineseBertClassification(BertModelInterface):
    def __init__(self, max_len: int, ckpt: str, epochs: int, learning_rate: float,
                 batch_size: int, dataset: Iterable[Dict[str, str]],
                 label_col: List[str], model_name: str, model_version: int,
                 optimizer: Any = None, do_lower_case=True, is_multi_label=1):

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
        MODEL_TOKEN_DIR = create_tokenizer_dir(self.model_name)

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
        save_model_token = Path(os.path.join(MODEL_TOKEN_DIR / f"{self.model_version}/"))
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
    def run(self) -> Dict[str, str]:

        if not self.model:
            raise ModelNotFoundError('Please init_model() before training model')

        hf_logging.set_verbosity_error()
        progress_bar = tqdm(range(self.epochs))

        self.logger.info(' **** Start training and validation flow **** ')
        best_f1_score = 0
        best_epoch = 0

        save_model_path = os.path.join(MODEL_BIN_DIR / f"{self.display_name.split('/')[-1]}.bin")
        # false_pred_path = os.path.join(MODEL_FP_DIR / f"{self.display_name.split('/')[-1]}.csv")

        for epoch in range(1, self.epochs + 1):
            self.logger.info('=' * 100)
            self.logger.info(f"Epoch: {epoch} / {self.epochs}")
            self.logger.info('-' * 100)

            train_loss, train_acc, train_time = self.train()
            self.logger.info(f"training loss {train_loss}; training time {train_time}")

            val_loss, val_acc, val_time = self.evaluate()
            self.logger.info(f"validation loss {val_loss}; validation time {val_time}")

            self.logger.info(f"training acc {train_acc}; validation acc {val_acc}")

            if val_acc['f1'] > best_f1_score:
                best_f1_score = val_acc['f1']
                best_epoch = epoch
                torch.save(self.model.state_dict(), save_model_path)
                self.logger.info(f"best f1_score is updated: {best_f1_score}")

            progress_bar.update(1)

        self.logger.info(' **** training is done !! **** ')
        self.logger.info(f"Best epoch: {best_epoch}")
        self.logger.info(f"Best validation f1_score: {best_f1_score}")

        val_report, val_fp_df = get_classification_report(
            self.model,
            self.dev_loader,
            save_model_path,
            self.device,
            self.label_col
        )

        if self.test_loader:
            test_report, test_fp_df = get_classification_report(
                self.model,
                self.test_loader,
                save_model_path,
                self.device,
                self.label_col
            )
        else:
            test_report, test_fp_df = None, None

        self.logger.info(f"Saving model to TorchScript ...")
        MODEL_PT_MODEL_NAME_DIR = create_model_dir(self.model_name)
        save_model_directory = Path(os.path.join(MODEL_PT_MODEL_NAME_DIR / f"{self.model_version}"))
        save_model_directory.mkdir(exist_ok=True)
        save_model_pt_path = os.path.join(save_model_directory / f"model.pt")

        save_pt(self.model, self.dummy_input, save_model_path, save_model_pt_path)

        model_size = get_model_size(self.model)
        self.logger.info(model_size)

        self.logger.info(val_report)
        if test_report:
            self.logger.info(test_report)

            return {
                'val_report': val_report.to_json(orient='records', force_ascii=False),
                'val_fp_df': val_fp_df.to_json(orient='records', force_ascii=False),
                'test_report': test_report.to_json(orient='records', force_ascii=False),
                'test_fp_df': test_fp_df.to_json(orient='records', force_ascii=False),
            }

        return {
            'val_report': val_report.to_json(orient='records', force_ascii=False),
            'val_fp_df': val_fp_df.to_json(orient='records', force_ascii=False),
        }


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
