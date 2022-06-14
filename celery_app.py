from datetime import datetime

from celery import Celery
from loguru import logger
from sqlmodel import create_engine, Session, select

from run import run_training_flow
from settings import CeleryConfig, DATABASE_URL, LogDir, LogVar
from utils.enum_helper import TrainingStatus
from utils.log_helper import get_log_name
from workers.dbs_builder.databases import TrainingTask
from workers.models_builder.builder import BertModelWorker

configuration = CeleryConfig()

celery_worker = Celery(name=configuration.CELERY_NAME,
                       backend=configuration.CELERY_BACKEND,
                       broker=configuration.CELERY_BROKER)

celery_worker.conf.update(enable_utc=configuration.CELERY_ENABLE_UTC)
celery_worker.conf.update(timezone=configuration.CELERY_TIMEZONE)
celery_worker.conf.update(task_track_started=configuration.CELERY_TASK_TRACK_STARTED)
celery_worker.conf.update(task_acks_late=configuration.CELERY_ACKS_LATE)

logger.add(
    get_log_name(LogDir.model, datetime.now()),
    level=LogVar.level,
    format=LogVar.format,
    enqueue=LogVar.enqueue,
    diagnose=LogVar.diagnose,
    catch=LogVar.catch,
    serialize=LogVar.serialize,
    backtrace=LogVar.backtrace,
    colorize=LogVar.color
)


@celery_worker.task(name=f'{configuration.CELERY_NAME}.trainer_training', ignore_result=True)
# def trainer_training(task_id, trainer):
def background_training(task_id, dataset_name, model_name, n_sample,
                        epoch, max_len, batch_size, split_rate, lr_rate, version):

    engine = create_engine(DATABASE_URL)
    start_time = datetime.now()

    # try:
    logger.info('init model worker ...')
    worker = BertModelWorker(
        dataset_name=dataset_name,
        model_name=model_name,
        n_sample=n_sample,
        learning_rate= lr_rate,
        epoch=epoch,
        max_len=max_len,
        batch_size=batch_size,
        train_val_split_rate=split_rate,
        version=version
    )

    args = {
        'dataset_name': dataset_name,
        'model_name': model_name,
        'n_sample': n_sample,
        'lr_rate': lr_rate,
        'epoch': epoch,
        'max_len': max_len,
        'batch_size': batch_size,
        'train_val_split_rate': split_rate,
        'version': version
    }

    logger.info(f"start training task_{task_id} ...")
    training_result = run_training_flow(worker, task_id=task_id)
    end_time = datetime.now()

    logger.info(f"writing result of training task_{task_id} ...")
    with Session(engine) as session:
        statement = select(TrainingTask).where(TrainingTask.id == task_id)
        results = session.exec(statement)
        _task = results.one()
        logger.debug(f"{_task}")
        _task.result = f"{training_result}"
        _task.training_args = f"{args}"
        _task.status = TrainingStatus.finished
        _task.total_time = (end_time - start_time).total_seconds() / 60

        session.add(_task)
        session.commit()
        session.refresh(_task)
        logger.debug(f"updated task_{task_id}")

    logger.info(f"task_{task_id} done")

    # except Exception as e:
    #     logger.error(f"task_{task_id} ...")
    #     with Session(engine) as session:
    #         statement = select(TrainingTask).where(TrainingTask.id == task_id)
    #         results = session.exec(statement)
    #         _task = results.one()
    #
    #         _task.status = TrainingStatus.failed
    #         _task.error_message = f"{e}"
    #
    #         logger.error(f"{e}")
    #         session.add(_task)
    #         session.commit()
    #         session.refresh(_task)
