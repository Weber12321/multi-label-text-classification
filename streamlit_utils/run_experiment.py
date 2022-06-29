from collections import defaultdict
from datetime import datetime

import torch
import wandb
from torch.nn import BCEWithLogitsLoss
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler

from evaluation.bert_pytorch import eval_epoch
from metrics.model_size import get_model_size
from metrics.nultilabel_metrics import get_classification_report
from training.bert_pytorch import train_epoch
from workers.datasets_builder.loader import create_data_loader


def run(
        df_train,
        df_test,
        label_col,
        learning_rate=2e-5,
        epochs=50,
        batch_size=32,
        max_len=30,
        num_labels=4,
        ckpt="roberta-base",
        dsn="au_1200_p",
        project_name="audience_bert_4_class",
        prefix=None
):
    display_name = f"{ckpt}_{prefix}_{datetime.now()}" if prefix else f"{ckpt}_{datetime.now()}"
    wandb.init(project=project_name, name=display_name)
    config = {
        "dataset": dsn,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_len": max_len,
        "pre-trained model": ckpt,
    }

    wandb.config.update(config)

    tokenizer = AutoTokenizer.from_pretrained(ckpt, do_lower_case=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt, num_labels=num_labels
    )

    model.config.problem_type = "multi_label_classification"

    train_loader = create_data_loader(df_train, tokenizer, max_len, batch_size)
    test_loader = create_data_loader(df_test, tokenizer, max_len, batch_size)

    optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)

    num_epochs = epochs
    num_training_steps = num_epochs * len(train_loader)

    lr_scheduler = get_scheduler(
        name='linear', optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_epochs))

    b_history = defaultdict(list)

    print(' *** Start training flow *** ')
    best_acc = 0
    best_epoch = 0

    wandb.watch(model, log="all")
    for epoch in range(1, num_epochs + 1):
        print('= ' * 100)
        print(f"Epoch {epoch} / {num_epochs}")

        print('- ' * 100)
        train_loss, train_acc, train_time = train_epoch(
            model, train_loader, optimizer, device, lr_scheduler, num_labels,
            loss_func=BCEWithLogitsLoss()
        )
        print(f"training loss {train_loss}; training time {train_time}")

        val_loss, val_acc, val_time = eval_epoch(
            model, test_loader, device, num_labels,
            loss_func=BCEWithLogitsLoss()
        )
        print(f"validation loss {val_loss}; validation time {val_time}")

        print(f"training acc {train_acc}")
        print(f"validation acc {val_acc}")

        if val_acc['f1'] > best_acc:
            best_acc = val_acc['f1']
            best_epoch = epoch
            torch.save(model.state_dict(), f"./model/{display_name.split('/')[-1]}_{dsn}.bin")
            wandb.save(f"{ckpt.split('/')[-1]}_{dsn}_{best_acc}.bin")
            print(f"best f1_score is updated: {best_acc}")

        b_history['train_acc'].append(train_acc)
        b_history['train_loss'].append(train_loss)
        b_history['val_acc'].append(val_acc)
        b_history['val_loss'].append(val_loss)

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc['f1'],
                "val/loss": val_loss,
                "val/accuracy": val_acc['f1'],
            }
        )

        progress_bar.update(1)
    print(' *** training is done !! *** ')
    print(f"Best epoch: {best_epoch}")
    print(f"Best f1_score: {best_acc}")

    report, false_prediction_df = get_classification_report(
        model,
        test_loader,
        f"./models/{display_name.split('/')[-1]}_{dsn}.bin",
        device,
        label_col
    )

    print(report)

    model_size = get_model_size(model)

    print(model_size)

    wandb.log({
        "classification_report": wandb.Table(dataframe=report),
        "false_prediction": wandb.Table(dataframe=false_prediction_df),
        "model_size_info": wandb.Table(dataframe=model_size)
    })

    wandb.finish()

    false_prediction_df.to_csv(f"./false_pred/{display_name.split('/')[-1]}.csv", encoding='utf-8', index=False)
