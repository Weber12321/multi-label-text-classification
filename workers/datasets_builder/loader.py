from torch.utils.data import DataLoader

from datasets_class import bert_dataset


def create_data_loader(
        df, tokenizer, max_len, batch_size
):
    dataset_cls = bert_dataset.BertDataset

    ds = dataset_cls(
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



