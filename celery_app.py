import json
import os
from datetime import datetime

from celery import Celery
from loguru import logger
from sqlmodel import create_engine, Session, select

from definition import DATA_DIR
from model_class.rule_model import RuleModelWorker
from run import run_training_flow
from settings import CeleryConfig, DATABASE_URL, EXCLUDE_WORDS
from utils.enum_helper import TrainingStatus
from utils.preprocess_helper import read_dataset_from_db
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


@celery_worker.task(name=f'{configuration.CELERY_NAME}.trainer_training', ignore_result=True)
def background_training(task_id, dataset_name, model_name, n_sample,
                        epoch, max_len, batch_size, split_rate, lr_rate, version):
    """

    :param task_id: task id
    :param dataset_name: dataset name, enum type, modify the selection in enum_helper.
    :param model_name: model name, enum type, modify the selection in enum_helper,
                enum value should corresponds to MODEL_CLASS in settings.py
    :param n_sample: number of sample extract from total dataset.
    :param epoch: training and validation epoch
    :param max_len: max length of data tokenizing
    :param batch_size: training and validation batch_size
    :param split_rate: train / val data split size, float number between 0 to 1
    :param lr_rate: learning rate of optimizer
    :param version: special version of data preprocess, see the readme
    :return: None
    """

    engine = create_engine(DATABASE_URL)
    start_time = datetime.now()

    try:
        logger.info('init model worker ...')
        worker = BertModelWorker(
            dataset_name=dataset_name,
            model_name=model_name,
            n_sample=n_sample,
            learning_rate=lr_rate,
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

    except Exception as e:
        logger.error(f"task_{task_id} ...")
        with Session(engine) as session:
            statement = select(TrainingTask).where(TrainingTask.id == task_id)
            results = session.exec(statement)
            _task = results.one()

            _task.status = TrainingStatus.failed
            _task.error_message = f"{e}"

            logger.error(f"{e}")
            session.add(_task)
            session.commit()
            session.refresh(_task)


@celery_worker.task(name=f'{configuration.CELERY_NAME}.auto_annotation', ignore_result=True)
def auto_annotation_flow(
        db: str,
        rule_file: str,
        n_multi_output: int,
        expect_data_size: int,
        start: str,
        end: str,
        char_length: int,
        sub_set_keep: str
):
    """
    :param sub_set_keep: filter out what tags you want to predict
    :param char_length: max len of retrieval data content
    :param db: database name
    :param rule_file: rule file, modify the ext config in settings.py
    :param n_multi_output: number of multi output threshold,
                            e.g. 1 means the output will only save the data have 2
                            or higher number of length of labels
    :param expect_data_size: expected output data limit size, processing will stop if
                            output dataset size meet this limit
    :param start: start date, it is used in batch processing in the sql query
    :param end: end date, it is used in batch processing in the sql query
    :return: None
    """

    if not start or not end:
        error_message = 'datetime params are missing'
        logger.error(error_message)
        raise ValueError(error_message)

    # logger.debug(f"{json.loads(sub_set_keep)}")

    model = RuleModelWorker(
        filename=rule_file,
        multi_output_threshold=n_multi_output,
        sub_set_keep=json.loads(sub_set_keep)
    )
    logger.debug('initializing model ...')
    model.initialize_model()

    datasets = []
    num_data = 0
    logger.info('==== start batch processing ====')
    start_date = datetime.strptime(start, "%Y-%m-%dT00:00:00")
    end_date = datetime.strptime(end, "%Y-%m-%dT00:00:00")

    for elements in read_dataset_from_db(
            db=db, start=start_date, end=end_date, char_length=char_length, filter_word_list=EXCLUDE_WORDS
    ):

        if num_data >= expect_data_size:
            logger.info(f"already meet the excepted length {num_data}")
            logger.info('==== finished processing ====')
            break

        dataset, checkpoint = elements

        # TODO: add the text split function if the text data is too long

        logger.info(f"---- checkpoint: {checkpoint} ----")
        logger.info(f"length of data: {len(dataset)}")

        # TODO: add a function which can return data that is not matched by rules

        output = model.predict(dataset)
        logger.info(f"length of output: {len(output)}")

        datasets.extend(output)
        num_data += len(output)
        logger.info(f"length of temp total results {num_data}")

    logger.info(f"collected length of data {num_data}")

    results = json.dumps(datasets, ensure_ascii=False)
    # results = pd.DataFrame(datasets)
    results_file_name = f"{db}_{n_multi_output}_{num_data}.json"
    results_file_path = os.path.join(DATA_DIR / "auto_annotation_dir" / results_file_name)

    logger.info(f"writing results to file {results_file_name}")
    with open(results_file_path, 'w') as f:
        f.write(results)

    # results.to_csv(results_file_path, encoding="utf-8")
