import json
from typing import Dict

from celery import Celery

from settings import LogDir, ServerConfig
from utils.inference_helper import chunks
from utils.log_helper import create_logger
from worker.inference.bert_triton_inference import BertInferenceWorker
from worker.train.chinese_bert_classification import ChineseBertClassification

celery_config = ServerConfig()

app = Celery(
    name=celery_config.celery_name,
    broker=celery_config.celery_broker,
    backend=celery_config.celery_backend
)

app.conf.task_routes = {
    f"{celery_config.celery_name}.*": {'queue': celery_config.celery_queue},
}

# app.conf.update(result_expires=celery_config.celery_result_expires)
app.conf.update(task_track_started=True)


@app.task
def training(
        model_name,
        version,
        dataset,
        label_col,
        learning_rate=2e-5,
        epochs=50,
        batch_size=32,
        max_len=30,
        is_multi_label=1,
        ckpt=celery_config.celery_model_ckpt

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

    # fp_df stand for false prediction records with text, ground_truth and prediction
    # report stand for output of sklearn classification_report
    best_val_f1, best_test_f1, val_report, test_report, val_fp_df, test_fp_df = task_worker.run()
    return best_val_f1, best_test_f1, val_report, test_report, val_fp_df, test_fp_df


@app.task
def predict(model_name, version, max_len, dataset):
    logger = create_logger(LogDir.inference)
    data = json.loads(dataset)

    output = []
    for idx, chunk in enumerate(chunks(data, 32)):
        logger.info(f" ==== batch: {idx} ==== ")
        infer_worker = BertInferenceWorker(
            dataset=chunk,
            model_name=model_name,
            model_version=version,
            url=celery_config.inference_url,
            backend=celery_config.inference_backend,
            max_len=max_len,
            chunk_size=len(chunk)
        )
        results = infer_worker.run()
        output.extend(results.tolist())

    # output is a nested list contained predicted probabilities with same length of input data,
    # e.g [[0.983423, 0.234324, 0.132432, 0.235643], [...], ...]
    assert len(output) == len(data)

    return json.dumps(output, ensure_ascii=False)
