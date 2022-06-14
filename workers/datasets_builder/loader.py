from torch.utils.data import DataLoader

from datasets_class import go_emotion
from utils.enum_helper import DatasetName


def create_data_loader(
        df, dataset_name, tokenizer, max_len, batch_size
):

    if dataset_name == DatasetName.go_emotion.value:

        dataset_cls = go_emotion.GoEmotionDataset
    else:
        raise ValueError(f"dataset_name {dataset_name} is not found in settings")

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



