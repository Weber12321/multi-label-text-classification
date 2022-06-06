from datetime import datetime
from loguru import logger
from transformers import TrainingArguments, Trainer

from settings import LogDir, LogVar
from utils.log_helper import get_log_name

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


def setup_trainer(model, compute_metrics, tokenizer,
        train_dataset, test_dataset, epoch=3, batch_size=32, weight_decay=0.01):
    logging_steps = len(train_dataset) // batch_size
    args = TrainingArguments(
        output_dir="models",
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epoch,
        weight_decay=weight_decay,
        logging_steps=logging_steps
    )

    logger.info(f"{args}")

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer)

    return trainer, args
