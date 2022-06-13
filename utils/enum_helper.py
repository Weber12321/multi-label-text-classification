from enum import Enum


class TrainingStatus(str, Enum):
    training = "training"
    finished = "finished"
    failed = "failed"


class ModelName(str, Enum):
    bert_base = "bert_base"
    XLNet = "XLNet"
    roberta = "roberta"
    albert = "albert"
    XLM_roberta = "XLM_roberta"


class DatasetName(str, Enum):
    go_emotion = "go_emotion"
