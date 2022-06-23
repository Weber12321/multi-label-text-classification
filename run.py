from datetime import datetime

import torch
from loguru import logger
from torch.nn import BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_scheduler

from evaluation.bert_pytorch import eval_epoch
from settings import LogDir, LogVar
from training.bert_pytorch import train_epoch
from utils.log_helper import get_log_name
from workers.models_builder.builder import BertModelWorker


logger.add(
    get_log_name(LogDir.training, datetime.now()),
    level=LogVar.level,
    format=LogVar.format,
    enqueue=LogVar.enqueue,
    diagnose=LogVar.diagnose,
    catch=LogVar.catch,
    serialize=LogVar.serialize,
    backtrace=LogVar.backtrace,
    colorize=LogVar.color
)


def run_training_flow(worker: BertModelWorker, task_id: int):

    optimizer = AdamW(
        worker.model.parameters(),
        lr=worker.learning_rate,
        correct_bias=False
    )
    num_training_steps = worker.epoch * len(worker.train_loader)
    lr_scheduler = get_scheduler(
        name='linear', optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    worker.model.to(device)

    logger.info('=' * 20)
    logger.info(f"Task_id {task_id}")
    logger.info('=' * 20)

    logger.info('Params:')
    logger.info(f"Model ckpt: {worker.model_ckpt}")
    logger.info(f"Dataset name: {worker.dataset_name}")
    logger.info(f"Number of samples: {worker.n_sample}")
    logger.info(f"Learning_rate: {worker.learning_rate}")
    logger.info(f"Batch_size: {worker.batch_size}")
    logger.info(f"Epoch: {worker.epoch}")
    logger.info(f"Max length: {worker.max_len}")

    logger.info('*** Start training flow ***')

    best_acc = 0
    best_epoch = 0
    best_train_acc_dict = {}
    best_val_acc_dict = {}

    comment = f"batch_size = {len(worker.train_loader)} lr = {worker.learning_rate}"
    tb = SummaryWriter(comment=comment)

    for epoch in range(1, worker.epoch + 1):
        logger.info('-' * 20)
        logger.info(f"Epoch {epoch} / {worker.epoch}")

        logger.info('-' * 20)
        train_loss, train_acc, train_time = train_epoch(
            model=worker.model,
            loader=worker.train_loader,
            optimizer=optimizer,
            device=device,
            lr_scheduler=lr_scheduler,
            num_labels=worker.n_labels,
            loss_func=BCEWithLogitsLoss()
        )

        logger.info(f"training loss {train_loss}; training time {train_time}")

        val_loss, val_acc, val_time = eval_epoch(
            model=worker.model,
            loader=worker.train_loader,
            device=device,
            num_labels=worker.n_labels,
            loss_func=BCEWithLogitsLoss()
        )

        logger.info(f"validation loss {val_loss}; validation time {val_time}")

        logger.info(f"training acc {train_acc}")
        logger.info(f"validation acc {val_acc}")

        if val_acc['f1'] > best_acc:
            best_acc = val_acc['f1']
            best_epoch = epoch
            best_train_acc_dict.update(train_acc)
            best_val_acc_dict.update(val_acc)
            print(f"best f1_score is updated: {best_acc}")

        tb.add_scalar('Loss/train', train_loss, epoch)
        tb.add_scalar('Loss/val', val_loss, epoch)
        tb.add_scalar('Accuracy/train', train_acc['accuracy'], epoch)
        tb.add_scalar('Accuracy/val', val_acc['accuracy'], epoch)
        tb.add_scalar('f1_score/train', train_acc['f1'], epoch)
        tb.add_scalar('f1_score/val', val_acc['f1'].epoch)
        tb.add_scalar('precision/train', train_acc['precision'], epoch)
        tb.add_scalar('precision/val', val_acc['precision'], epoch)
        tb.add_scalar('recall/train', train_acc['recall'], epoch)
        tb.add_scalar('recall/val', val_acc['recall'], epoch)

        tb.add_hparams(
            {
                "lr": worker.learning_rate,
                "bsize": len(worker.train_loader)
             },
            {
                "train_f1": train_acc['f1'],
                "val_f1": val_acc['f1'],
                "train_loss": train_loss,
                "val_loss": val_loss
            },
        )

    logger.info(' *** training is done !! *** ')
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best f1_score: {best_acc}")

    tb.close()

    return {
        'best_result': {
            'epoch': best_epoch,
            'f1_score': best_acc,
            'details': {
                'train_acc': best_train_acc_dict,
                'val_acc': best_val_acc_dict
            }
        }
    }
