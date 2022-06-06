from datetime import datetime
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine, Enum, Column

from utils.enum_helper import TrainingStatus


class TrainingTask(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    dataset_name: str
    model_name: str
    status: TrainingStatus = Field(sa_column=Column(Enum(TrainingStatus)))
    create_time: datetime
    total_time: Optional[float]
    training_args: Optional[str]
    training_result: Optional[str]
    evaluate_result: Optional[str]
    error_message: Optional[str]


