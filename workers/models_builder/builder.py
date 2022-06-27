from typing import Optional
from transformers import AutoModelForSequenceClassification
from model_class.model_interface import ModelWorker
from preprocess.raw_data_preprocess.preprocess import build_dataset


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
        if self.n_labels:
            model = self.create_model_class(self.n_labels)
            model.config.problem_type = "multi_label_classification"
            return model

        else:
            err_msg = f"number of labels is missing"
            raise ValueError(err_msg)

    def create_model_class(self, num_labels):
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_ckpt, num_labels=num_labels
            )
        except Exception as e:
            raise ValueError(e)

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
