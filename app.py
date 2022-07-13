from celery import Celery

from config.settings import TrainingFileName
from train.run import run
from utils.train_helper import load_dataset

app = Celery(
    name='bert_celery',
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/1"
)

app.conf.task_routes = {
    'A.*': {'queue': 'training_queue'},
}

app.conf.update(result_expires=1)
app.conf.update(task_track_started=True)


@app.task
def training(
        model_name,
        version,
        learning_rate=2e-5,
        epochs=50,
        batch_size=32,
        max_len=30,
        ckpt="hfl/chinese-bert-wwm-ext",
        dsn="au_2234_p"
):
    df_train, df_test, label_col = load_dataset(
        TrainingFileName.dataset, TrainingFileName.labels
    )

    report, model_size, false_pred = run(
        df_train= df_train,
        df_test= df_test,
        label_col=label_col,
        model_name=model_name,
        version=version,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        max_len=max_len,
        num_labels=len(label_col),
        ckpt=ckpt,
        dsn=dsn
    )

    return report, model_size, false_pred


@app.task
def predict(model_name, version, batch_size, max_len, dataset):
    # todo: add inference function which take the dataset input and return the output labels
    pass
