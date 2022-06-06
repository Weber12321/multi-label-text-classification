from datetime import datetime

from celery import Celery
from loguru import logger
from sqlmodel import create_engine, SQLModel, Session, select

from settings import CeleryConfig, DATABASE_URL, LogDir, LogVar
from utils.enum_helper import TrainingStatus
from utils.log_helper import get_log_name
from workers.build_dbs.databases import TrainingTask
from workers.build_models.builder import BertModelWorker

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
def trainer_training(dataset_name, model_name, n_sample, is_trainer,
                     epoch, batch_size, weight_decay):
    logger.info('init model worker ...')
    model_worker = BertModelWorker(
        dataset_name, model_name,
        n_sample, is_trainer, epoch,
        batch_size, weight_decay)
    trainer, args = model_worker.initialize_model()

    logger.info('init database ...')
    engine = create_engine(DATABASE_URL)
    SQLModel.metadata.create_all(engine)

    logger.info('creating the training task ...')
    start_time = datetime.now()
    task = TrainingTask(
        dataset_name=dataset_name,
        model_name=model_name,
        status=TrainingStatus.training,
        create_time=start_time,
        training_args=f"{args}"
    )

    with Session(engine) as session:
        session.add(task)
        session.commit()
        session.refresh(task)

    task_id = task.id

    try:
        logger.info(f"start training task_{task_id} ...")
        train_result = trainer.train()
        eval_result = trainer.evaluate()
        end_time = datetime.now()

        logger.info(f"writing result of training task_{task_id} ...")
        with Session(engine) as session:
            statement = select(TrainingTask).where(TrainingTask.id == task_id)
            results = session.exec(statement)
            _task = results.one()
            logger.debug(f"{_task}")

            _task.training_result = f"{train_result}"
            _task.evaluate_result = f"{eval_result}"
            _task.status = TrainingStatus.finished
            _task.total_time = (end_time - start_time).total_seconds() / 60

            session.add(_task)
            session.commit()
            session.refresh(_task)
            logger.debug(f"updated task_{task_id}")

        logger.info(f"task_{task_id} done")

    except Exception as e:
        logger.info(f"task_{task_id} ...")
        with Session(engine) as session:
            statement = select(TrainingTask).where(TrainingTask.id == task_id)
            results = session.exec(statement)
            _task = results.one()

            logger.error(f"{_task}")
            _task.status = TrainingStatus.failed
            _task.error_message = f"{e}"

            logger.error(f"{e}")
            session.add(_task)
            session.commit()
            session.refresh(_task)
            logger.error(f"updated task_{task_id}")


