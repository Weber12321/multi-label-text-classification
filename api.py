from datetime import datetime

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from sqlmodel import create_engine, Session, select, SQLModel

from celery_app import background_training, auto_annotation_flow
from settings import APIConfig, LogDir, LogVar, PostTaskData, DATABASE_URL, AutoAnnotation
from utils.database_helper import orm_cls_to_dict
from utils.enum_helper import TrainingStatus, DatasetName, ModelName, DatabaseSelection, RuleSelection
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

description = """
### Multi-label training task 
This is a quick multi-label multi-class training flow build with fastapi and celery 
which utilizes the [huggingface transformers](https://huggingface.co/docs/transformers/index) 
tool. The current dataset of this project is [go_emotions](https://huggingface.co/datasets/go_emotions) 
with models for sequence classification:  

+ aLBERT
+ BERT
+ roBERTa
+ XLNet
+ XLM-roBERTa

Users can post a task to execute a training flow by assigning the target dataset and model 
with adjusting the training parameters, and check the results of the training status. 
Noted that the training information is store in the log files in /logs directory and also 
save in a default sqlite database in debug mode, you can specify the database configuration 
in the `.env` and `settings.py` to switch the database.  
    
### Multi-label auto annotation task  
For auto-annotating, it utilizes the regular expression model.
Users can determine: 
   
+ expected size of output dataset  
+ threshold of label size  
+ start and end scrapping date  
+ source database name  
+ rule file version 
   
The output dataset will be saved as `CSV` file with the column `text` and `labels` in the data directory.
"""

app = FastAPI(
    title=configuration.API_TITLE,
    version=configuration.API_VERSION,
    description=description
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


@app.post('/task/training', description="select a dataset and model first, and then adjust the params.")
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
                body.LEARNING_RATE,
                body.VERSION
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


@app.get('/task/{_id}', description="""
return a task info by inputting a task_id, if the `status` shows `training` then the task 
is still running, you can check the result if `status` becomes `finished`, or check the 
`error_message` if the task is `failed`. 
""")
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


@app.post('/task/annotation', description="""
Execute the auto annotation flow using regular expression model.  
+ `n_multi_tresh`: the threshold of the length of output labels should return, e.g. `n_multi_tresh > 1` will only retrieve the data which contains 2 or higher number of labels.
+ `expect_output_data_length`: the number of data you expect to retrieve, if it meet the number the annotation flow will stop.  
+ `max_char_length`: the max length of the retrieval content size.   
The output of retrieved dataset will be save as CSV file in the data directory named as `<database_name>_<n_multi_tresh>_<length of output>.csv`
""")
def auto_annotation(
        body: AutoAnnotation,
        database_name: DatabaseSelection,
        rule_file_name: RuleSelection,
        n_multi_tresh: int = 0,
        expect_output_data_length: int = 1000,
        max_char_length: int = 200
):
    try:
        auto_annotation_flow.apply_async(
            args=(
                database_name,
                rule_file_name,
                n_multi_tresh,
                expect_output_data_length,
                body.START_TIME,
                body.END_TIME,
                max_char_length
            ),
            queue='queue1'
        )
        logger.info(f"{status.HTTP_200_OK}")
        return JSONResponse(status_code=status.HTTP_200_OK, content='OK')
    except Exception as e:
        logger.error(f"{status.HTTP_500_INTERNAL_SERVER_ERROR}: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=jsonable_encoder(f"{e}")
        )





if __name__ == '__main__':
    uvicorn.run("__main__:app", host=configuration.API_HOST, debug=True)
