from datetime import datetime

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from sqlmodel import create_engine, Session, select, SQLModel

from celery_app import background_training
from settings import APIConfig, LogDir, LogVar, PostTaskData, DATABASE_URL
from utils.database_helper import orm_cls_to_dict
from utils.enum_helper import TrainingStatus, DatasetName, ModelName
from utils.log_helper import get_log_name
from workers.dbs_builder.databases import TrainingTask

configuration = APIConfig()

logger.add(
    get_log_name(LogDir.api, datetime.now()),
    level=LogVar.level,
    format=LogVar.format,
    enqueue=LogVar.enqueue,
    diagnose=LogVar.diagnose,
    catch=LogVar.catch,
    serialize=LogVar.serialize,
    backtrace=LogVar.backtrace,
    colorize=LogVar.color
)

app = FastAPI(
    title=configuration.API_TITLE,
    version=configuration.API_VERSION
)


@app.exception_handler(RequestValidationError)
def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder(f"{[err['msg'] for err in exc.errors()]}")
    )


app.add_exception_handler(RequestValidationError, request_validation_exception_handler)

engine = create_engine(DATABASE_URL)
SQLModel.metadata.create_all(engine)


@app.get('/')
def render_tasks():
    try:
        with Session(engine) as session:
            statement = select(TrainingTask)
            results = session.exec(statement)
            outputs = [orm_cls_to_dict(r) for r in results]
            logger.info(f"{status.HTTP_200_OK}: {outputs}")
            return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(outputs))

    except Exception as e:
        logger.error(f"{status.HTTP_500_INTERNAL_SERVER_ERROR}: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=jsonable_encoder(f"{e}")
        )


@app.post('/')
def post_task(
        body: PostTaskData,
        dataset_name: DatasetName,
        model_name: ModelName):
    try:
        logger.info('creating the training task ...')
        start_time = datetime.now()
        task = TrainingTask(
            dataset_name=dataset_name,
            model_name=model_name,
            status=TrainingStatus.training,
            create_time=start_time,
            training_args=""
        )

        with Session(engine) as session:
            session.add(task)
            session.commit()
            session.refresh(task)

        task_id = task.id

        background_training.apply_async(
            args=(
                task_id,
                dataset_name,
                model_name,
                body.N_SAMPLE,
                body.EPOCH,
                body.MAX_LEN,
                body.BATCH_SIZE,
                body.SPLIT_RATE,
                body.LEARNING_RATE
            ),
            queue='queue1'
        )
        logger.debug('task started ...')
        return JSONResponse(status_code=status.HTTP_200_OK, content=f"OK, task id: {task_id}")

    except Exception as e:
        err_msg = f"failed to post training task since {type(e).__class__}:{e}"
        logger.error(err_msg)
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            content=jsonable_encoder(err_msg))


@app.get('/{_id}')
def get_task(_id: int):
    try:
        with Session(engine) as session:
            statement = select(TrainingTask).where(TrainingTask.id == _id)
            result = session.exec(statement).one()
        output = orm_cls_to_dict(result)
        logger.info(f"{status.HTTP_200_OK}: {output}")
        return JSONResponse(status_code=status.HTTP_200_OK, content=jsonable_encoder(output))

    except Exception as e:
        logger.error(f"{status.HTTP_500_INTERNAL_SERVER_ERROR}: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=jsonable_encoder(f"{e}")
        )


if __name__ == '__main__':
    uvicorn.run("__main__:app", host=configuration.API_HOST, debug=True)
