from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset

from utils.enum_helper import DatasetType


# @dataclass
# class DatasetSchema:
#     id: int
#     text: str
#     labels: List[int]
#     dataset_type: DatasetType.train


class AudienceDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len, ids=None):
        self.ids = ids
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        id_ = self.ids[idx] if self.ids else None
        review = str(self.reviews[idx])
        target = self.targets[idx]
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'id_': id_,
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target)
        }

    def __len__(self):
        return len(self.reviews)
