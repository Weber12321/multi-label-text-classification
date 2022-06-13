import importlib
from torch.utils.data import DataLoader

from settings import DATA_CLASS


def create_data_loader(
        df, dataset_name, tokenizer, max_len, batch_size
):
    if mod_path := DATA_CLASS.get(dataset_name):
        module_path, class_name = mod_path.rsplit(sep='.', maxsplit=1)
    else:
        raise ValueError(f"dataset module {dataset_name} is unknown in settings")

    dataset_cls = getattr(
        importlib.import_module(module_path), class_name
    )

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
