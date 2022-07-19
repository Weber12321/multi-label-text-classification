import json
from typing import Dict

from celery import Celery

from config.settings import MODEL_CKPT, LogDir
from utils.inference_helper import chunks
from utils.log_helper import create_logger
from worker.inference.bert_triton_inference import BertInferenceWorker
from worker.train.chinese_bert_classification import ChineseBertClassification

app = Celery(
    name='bert_celery',
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1"
)

app.conf.task_routes = {
    'app.*': {'queue': 'deep_model'},
}

app.conf.update(result_expires=1)
app.conf.update(task_track_started=True)


@app.task(bind=True, queue='deep_model', name='training')
def training(
        self,
        model_name,
        version,
        dataset,
        label_col,
        learning_rate=2e-5,
        epochs=50,
        batch_size=32,
        max_len=30,
        is_multi_label=1,
        ckpt=MODEL_CKPT.get('chinese-bert-wwm')

):
    dataset = json.loads(dataset)
    label_col = json.loads(label_col)

    task_worker = ChineseBertClassification(
        max_len=max_len,
        ckpt=ckpt,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        dataset=dataset,
        label_col=label_col,
        model_name=model_name,
        model_version=version,
        is_multi_label=is_multi_label
    )

    task_worker.init_model()
    results: Dict[str, str] = task_worker.run()
    return results

    # df_train, df_test, label_col = load_dataset(
    #     TrainingFileName.dataset, TrainingFileName.labels
    # )
    #
    # report, model_size, false_pred = run(
    #     df_train= df_train,
    #     df_test= df_test,
    #     label_col=label_col,
    #     model_name=model_name,
    #     version=version,
    #     learning_rate=learning_rate,
    #     epochs=epochs,
    #     batch_size=batch_size,
    #     max_len=max_len,
    #     num_labels=len(label_col),
    #     ckpt=ckpt,
    #     dsn=dsn
    # )
    #
    # return report, model_size, false_pred


@app.task(bind=True, queue='deep_model', name='predict')
def predict(self, model_name, version, max_len, dataset):
    logger = create_logger(LogDir.inference)
    data = json.loads(dataset)

    output = []
    for idx, chunk in enumerate(chunks(data, 32)):
        logger.info(f" ==== batch: {idx} ==== ")
        infer_worker = BertInferenceWorker(
            dataset=chunk,
            model_name=model_name,
            model_version=version,
            url='localhost:8000',
            backend='pytorch',
            max_len=max_len,
            chunk_size=len(chunk)
        )
        results = infer_worker.run()
        # print(results)
        output.extend(results.tolist())

    assert len(output) == len(data)

    return json.dumps(output, ensure_ascii=False)


