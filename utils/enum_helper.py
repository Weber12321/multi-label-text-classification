from enum import Enum


class TrainingStatus(str, Enum):
    training = "training"
    finished = "finished"
    failed = "failed"


class ModelName(str, Enum):
    distil_bert = 'distil_bert'


class DatasetName(str, Enum):
    go_emotion = "go_emotion"




