from datetime import datetime
from typing import Optional, List

from sqlmodel import Field, SQLModel, Enum, Column, Relationship

from utils.enum_helper import TrainingStatus


class TrainingTask(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    dataset_name: str
    model_name: str
    status: TrainingStatus = Field(sa_column=Column(Enum(TrainingStatus)))
    create_time: datetime
    total_time: Optional[float]
    training_args: Optional[str]
    result: Optional[str]
    error_message: Optional[str]


class BertTask(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    dataset_name: str
    model_name: str
    status: TrainingStatus = Field(sa_column=Column(Enum(TrainingStatus)))
    create_time: datetime
    total_time: Optional[float]
    epoch: Optional[int]
    batch_size: Optional[int]
    max_len: Optional[int]
    learning_rate: Optional[float]
    report: Optional[str]
    model_size: Optional[float]
    error_message: Optional[str]

    bertfp: List["BertFP"] = Relationship(back_populates="berttask")


# False prediction details for bert task
class BertFP(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    text: str
    prediction: Optional[str]
    ground_truth: Optional[str]

    berttask_id: Optional[int] = Field(default=None, foreign_key="")
    berttask: Optional[BertTask] = Relationship(back_populates="bertfp")
