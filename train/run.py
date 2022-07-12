from collections import defaultdict
from datetime import datetime

import torch
from loguru import logger
from torch.nn import BCEWithLogitsLoss
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler

from config.definition import *
from config.settings import LogDir
from data_set.data_loader import create_data_loader
from train.evaluate_flow import eval_epoch
from train.predict_flow import get_classification_report, save_pt, get_model_size
from train.train_flow import train_epoch
from utils.log_helper import create_logger
from utils.train_helper import get_dummy_input, create_model_dir

logger_handler = create_logger(LogDir.training)


@logger.catch
def run(
        df_train,
        df_test,
        label_col,
        model_name,
        version,
        learning_rate=2e-5,
        epochs=50,
        batch_size=32,
        max_len=30,
        num_labels=4,
        ckpt="roberta-base",
        dsn="au_1200_p",
        # project_name="audience_bert_4_class",
        prefix=None
):
    if not version or not isinstance(version, int):
        raise ValueError(f"version should be a non-zero integer")

    display_name = f"{ckpt}_{prefix}_{datetime.now().date()}" if prefix else f"{ckpt}_{datetime.now().date()}"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # wandb.init(project=project_name, name=display_name)
    training_config = {
        "dataset": dsn,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_len": max_len,
        "pre-trained model": ckpt,
        "device": device
    }

    logger_handler.info(training_config)

    # wandb.config.update(config)

    tokenizer = AutoTokenizer.from_pretrained(ckpt, do_lower_case=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt,
        num_labels=num_labels,
        torchscript=True
    )

    model.config.problem_type = "multi_label_classification"

    train_loader = create_data_loader(df_train, tokenizer, max_len, batch_size)
    test_loader = create_data_loader(df_test, tokenizer, max_len, batch_size)

    save_model_token = Path(os.path.join(TOKEN_DIR / f"{version}/"))
    save_model_token.mkdir(exist_ok=True)
    tokenizer.save_pretrained(save_model_token)

    optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)

    num_epochs = epochs
    num_training_steps = num_epochs * len(train_loader)

    lr_scheduler = get_scheduler(
        name='linear', optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=num_training_steps
    )

    model.to(device)

    dummy_input = get_dummy_input(df_train.text.max(), tokenizer, max_len, device=device)

    progress_bar = tqdm(range(num_epochs))

    b_history = defaultdict(list)

    logger_handler.info(' *** Start training flow *** ')
    # print(' *** Start training flow *** ')
    best_acc = 0
    best_epoch = 0

    MODEL_PT_MODEL_NAME_DIR = create_model_dir(model_name)

    save_model_path = os.path.join(MODEL_BIN_DIR / f"{display_name.split('/')[-1]}_{dsn}.bin")
    false_pred_path = os.path.join(MODEL_FP_DIR / f"{display_name.split('/')[-1]}_{dsn}.csv")
    # wandb.watch(model, log="all")

    for epoch in range(1, num_epochs + 1):
        # print('=' * 100)
        # print(f"Epoch {epoch} / {num_epochs}")
        # print('-' * 100)

        logger_handler.info('=' * 100)
        logger_handler.info(f"Epoch {epoch} / {num_epochs}")
        logger_handler.info('-' * 100)

        train_loss, train_acc, train_time = train_epoch(
            model, train_loader, optimizer, device,
            num_labels, lr_scheduler, loss_func=BCEWithLogitsLoss()
        )
        # print(f"training loss {train_loss}; training time {train_time}")
        logger_handler.info(f"training loss {train_loss}; training time {train_time}")

        val_loss, val_acc, val_time = eval_epoch(
            model, test_loader, device, num_labels,
            loss_func=BCEWithLogitsLoss()
        )
        # print(f"validation loss {val_loss}; validation time {val_time}")
        logger_handler.info(f"validation loss {val_loss}; validation time {val_time}")

        # print(f"training acc {train_acc}")
        # print(f"validation acc {val_acc}")
        logger_handler.info(f"training acc {train_acc}; validation acc {val_acc}")

        if val_acc['f1'] > best_acc:
            best_acc = val_acc['f1']
            best_epoch = epoch
            torch.save(model.state_dict(), save_model_path)
            # wandb.save(f"{ckpt.split('/')[-1]}_{dsn}_{best_acc}.bin")
            # print(f"best f1_score is updated: {best_acc}")
            logger_handler.info(f"best f1_score is updated: {best_acc}")

        b_history['train_acc'].append(train_acc)
        b_history['train_loss'].append(train_loss)
        b_history['val_acc'].append(val_acc)
        b_history['val_loss'].append(val_loss)

        # wandb.log(
        #     {
        #         "epoch": epoch,
        #         "train/loss": train_loss,
        #         "train/accuracy": train_acc['f1'],
        #         "val/loss": val_loss,
        #         "val/accuracy": val_acc['f1'],
        #     }
        # )

        progress_bar.update(1)
    # print(' *** training is done !! *** ')
    # print(f"Best epoch: {best_epoch}")
    # print(f"Best f1_score: {best_acc}")

    logger_handler.info(' *** training is done !! *** ')
    logger_handler.info(f"Best epoch: {best_epoch}")
    logger_handler.info(f"Best f1_score: {best_acc}")

    report, false_prediction_df = get_classification_report(
        model,
        test_loader,
        save_model_path,
        device,
        label_col
    )
    save_model_directory = Path(os.path.join(MODEL_PT_MODEL_NAME_DIR / f"{version}"))
    save_model_directory.mkdir(exist_ok=True)
    save_model_pt_path = os.path.join(save_model_directory / f"model.pt")

    save_pt(
        model,
        dummy_input,
        save_model_path,
        save_model_pt_path
    )

    # print(report)
    logger_handler.info(report)

    model_size = get_model_size(model)

    # print(model_size)
    logger_handler.info(model_size)

    # wandb.log({
    #     "classification_report": wandb.Table(dataframe=report),
    #     "false_prediction": wandb.Table(dataframe=false_prediction_df),
    #     "model_size_info": wandb.Table(dataframe=model_size)
    # })

    # wandb.finish()

    false_prediction_df.to_csv(false_pred_path, encoding='utf-8', index=False)

    return report.to_json(orient='records', force_ascii=False), \
           model_size.to_json(orient='records', force_ascii=False), \
           false_prediction_df.to_json(orient='records', force_ascii=False)
