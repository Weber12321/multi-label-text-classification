from typing import List

import numpy as np
from torch.utils.data import DataLoader

from data_set.dataset_class import AudienceDataset


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = AudienceDataset(
        reviews=df.text.to_numpy(),
        targets=df.labels.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
    )


def     create_data_loader_pred(texts: np.array, tokenizer, max_len) -> List[dict]:
    outputs = []
    for text in texts:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        outputs.append({
            'review_text': text,
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        })

    return outputs
