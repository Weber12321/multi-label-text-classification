from typing import Dict

from loguru import logger

from model_class.distilbert import DistilBertForMultilabelSequenceClassification
from settings import DEVICE


def go_emotion_model(model_name: str,
                     n_labels: int,
                     id2label: Dict[str, str],
                     label2id: Dict[str, str],
                     log: logger,
                     model_class=DistilBertForMultilabelSequenceClassification):
    log.debug('building model ...')
    model = model_class.from_pretrained(model_name, num_labels=n_labels).to(DEVICE)
    model.config.id2label = id2label
    model.config.label2id = label2id
    log.info(f"{model.config}")

    return model
