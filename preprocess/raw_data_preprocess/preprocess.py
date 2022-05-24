from datetime import datetime
from datasets import load_dataset
from loguru import logger

from datasets.go_emotion import GoEmotionDataset
from settings import LogDir, LogVar
from tokenizers.tokenize import build_tokenizer
from utils.enum_helper import DatasetName
from utils.log_helper import get_log_name
from utils.preprocess_helper import custom_train_test_split, get_data_sample

logger.add(
    get_log_name(LogDir.preprocess, datetime.now()),
    level=LogVar.level,
    format=LogVar.format,
    enqueue=LogVar.enqueue,
    diagnose=LogVar.diagnose,
    catch=LogVar.catch,
    serialize=LogVar.serialize,
    backtrace=LogVar.backtrace,
    colorize=LogVar.color
)


def build_dataset(dataset_name: str, ckpt: str, n_sample: int = 500):
    if dataset_name == DatasetName.go_emotion.value:
        logger.debug('loading go_emotion dataset ...')
        df = load_dataset("go_emotions", "raw")
        df = df['train'].to_pandas()
        logger.info(f"dataset shape: {df.shape}")

        label_cols = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
                      'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
                      'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
                      'remorse', 'sadness', 'surprise', 'neutral']

        logger.debug(f"length of labels: {len(label_cols)}")

        df["labels"] = df[label_cols].values.tolist()

        # create id-label list
        id2label = {str(i): label for i, label in enumerate(label_cols)}
        label2id = {label: str(i) for i, label in enumerate(label_cols)}

        df_sample = get_data_sample(df, logger, n_sample)
        df_train, df_test = custom_train_test_split(df_sample, logger)

        # build Datasets
        train_encodings, test_encodings, train_labels, test_labels, tokenizer = build_tokenizer(
            token_name=ckpt, df_train=df_train, df_test=df_test,
            context_col="text", label_col="labels", log=logger
        )

        logger.info('building data class ...')
        train_dataset = GoEmotionDataset(train_encodings, train_labels)
        test_dataset = GoEmotionDataset(test_encodings, test_labels)

        logger.debug(f"random checking the first row of training set {train_dataset[0]}")
        logger.debug(f"random checking the first row of testing set {test_dataset[0]}")

        return {
            "train_dataset": train_dataset, "test_dataset": test_dataset,
            "id2label": id2label, "label2id": label2id, 'n_labels': len(label_cols),
            "tokenizer": tokenizer
        }
    else:
        error_message = f"dataset {dataset_name} is not found"
        logger.error(error_message)
        raise ValueError(error_message)

