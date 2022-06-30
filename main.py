import os

import pandas as pd
import streamlit as st
from collections import defaultdict
from datetime import datetime

import torch
import wandb
from stqdm import stqdm
from torch.nn import BCEWithLogitsLoss
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler

from definition import MODEL_DIR, FALSE_PRED
from evaluation.bert_pytorch import eval_epoch
from metrics.model_size import get_model_size
from metrics.nultilabel_metrics import get_classification_report
from training.bert_pytorch import train_epoch
from workers.datasets_builder.loader import create_data_loader

from streamlit_utils.run_process import preprocess, convert_df

if __name__ == '__main__':
    st.title("Bert model quick experiment")
    st.write("Quick experiment of fine tuning the hyper-parameters of bert model")

    sidebar = st.sidebar
    dataset = sidebar.selectbox(
        'Dataset',
        ['au_200_p.json', 'au_2234_p.json']
    )
    train_display = sidebar.checkbox(
        'Display training data?',
        value=True
    )
    epoch = sidebar.slider(
        'Number of epoch?',
        min_value=2,
        max_value=50,
        value=10
    )
    label_column = sidebar.selectbox(
        'Number of label?',
        ('small (4)', 'origin (8)')
    )
    learning_rate = sidebar.number_input(
        'Insert learning rate between 1e-3 and 1e-6',
        min_value=0.000001,
        max_value=0.001,
        value=0.00002
    )
    batch_size = sidebar.slider(
        'Insert batch size between 8 and 64',
        min_value=8,
        max_value=64,
        value=32
    )
    max_len = sidebar.slider(
        'Insert max length between 30 to 100',
        min_value=30,
        max_value=100,
        value=64
    )
    ckpt = sidebar.selectbox(
        'choose a pre-trained model',
        (
            'bert-base-chinese',
            'hfl/chinese-bert-wwm-ext',
            'hfl/chinese-macbert-base',
            'hfl/chinese-roberta-wwm-ext'
        )
    )
    start = sidebar.button('start experiment')

    df, df_train, df_test = preprocess(dataset)

    if label_column == 'small (4)':
        label_col = 4
        project_name = 'audience_bert_4_class'
    else:
        label_col = 8
        project_name = 'audience_bert'

    if train_display:
        st.write(df)

    st.write(project_name)
    st.write('Model: ', ckpt)
    st.write('Epoch: ', epoch)
    st.write('Number of label: ', label_col)
    st.write('Batch size: ', batch_size)
    st.write('Max length: ', max_len)
    st.write('Learning rate: ', learning_rate)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    st.write('Device: ', device)

    if start:
        with st.spinner(text='Training in progress'):
            display_name = f"{ckpt}_{datetime.now()}"
            wandb.init(project=project_name, name=display_name)
            config = {
                "dataset": dataset.split('.')[0],
                "learning_rate": learning_rate,
                "epochs": epoch,
                "batch_size": batch_size,
                "max_len": max_len,
                "pre-trained model": ckpt,
            }

            wandb.config.update(config)

            tokenizer = AutoTokenizer.from_pretrained(ckpt, do_lower_case=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                ckpt, num_labels=label_col
            )

            model.config.problem_type = "multi_label_classification"

            train_loader = create_data_loader(df_train, tokenizer, max_len, batch_size)
            test_loader = create_data_loader(df_test, tokenizer, max_len, batch_size)

            optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)

            num_epochs = epoch
            num_training_steps = num_epochs * len(train_loader)

            lr_scheduler = get_scheduler(
                name='linear', optimizer=optimizer,
                num_warmup_steps=0, num_training_steps=num_training_steps
            )

            model.to(device)

            # progress_bar = tqdm(range(num_epochs))

            b_history = defaultdict(list)

            # print(' *** Start training flow *** ')
            best_acc = 0
            best_epoch = 0

            wandb.watch(model, log="all")
            for epoch in stqdm(range(1, num_epochs + 1)):
                # print('= ' * 100)
                # print(f"Epoch {epoch} / {num_epochs}")
                st.write(f"\n__Epoch {epoch} / {num_epochs}__\n")

                # print('- ' * 100)
                train_loss, train_acc, train_time = train_epoch(
                    model, train_loader, optimizer, device, lr_scheduler, label_col,
                    loss_func=BCEWithLogitsLoss()
                )
                # print(f"training loss {train_loss}; training time {train_time}")

                val_loss, val_acc, val_time = eval_epoch(
                    model, test_loader, device, label_col,
                    loss_func=BCEWithLogitsLoss()
                )
                # print(f"validation loss {val_loss}; validation time {val_time}")
                st.write(f"* training time: {train_time} min; validation time: {val_time} min")

                st.write(f"* training acc: {train_acc}")
                st.write(f"* validation acc: {val_acc}")
                # print(f"training acc {train_acc}")
                # print(f"validation acc {val_acc}")

                if val_acc['f1'] > best_acc:
                    best_acc = val_acc['f1']
                    best_epoch = epoch

                    torch.save(
                        model.state_dict(),
                        os.path.join(MODEL_DIR / f"{ckpt}_{dataset.split('.')[0]}.bin")
                    )
                    wandb.save(f"{ckpt.split('/')[-1]}_{dataset.split('.')[0]}_{best_acc}.bin")
                    # print(f"best f1_score is updated: {best_acc}")
                    st.write(f"* best f1_score is updated: {best_acc}")

                b_history['train_acc'].append(train_acc)
                b_history['train_loss'].append(train_loss.item())
                b_history['val_acc'].append(val_acc)
                b_history['val_loss'].append(val_loss.item())

                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "train/accuracy": train_acc['f1'],
                        "val/loss": val_loss,
                        "val/accuracy": val_acc['f1'],
                    }
                )

                # progress_bar.update(1)
            # print(' *** training is done !! *** ')
            # print(f"Best epoch: {best_epoch}")
            # print(f"Best f1_score: {best_acc}")

            st.write(f"\n__Finished__\n")
            st.write(f"* Best epoch: {best_epoch}")
            st.write(f"* Best f1_score: {best_acc}")

            report, false_prediction_df = get_classification_report(
                model,
                test_loader,
                os.path.join(MODEL_DIR / f"{ckpt}_{dataset.split('.')[0]}.bin"),
                device,
                label_col=['男性', '女性', '已婚', '未婚'] if label_col == 4 else ['男性', '女性', '已婚', '未婚', '上班族', '學生', '青年',
                                                                           '有子女']
            )

            # print(report)

            model_size = get_model_size(model)

            # print(model_size)

            wandb.log({
                "classification_report": wandb.Table(dataframe=report),
                "false_prediction": wandb.Table(dataframe=false_prediction_df),
                "model_size_info": wandb.Table(dataframe=model_size)
            })

            wandb.finish()

            false_prediction_df.to_csv(
                os.path.join(FALSE_PRED / f"{ckpt}_{dataset.split('.')[0]}_fp.csv"),
                encoding='utf-8',
                index=False
            )

            results = {
                'history': b_history,
                'report': report,
                'size': model_size,
                'false_prediction': false_prediction_df,
            }

        st.success("Training was done!")
        st.write(results['report'])
        st.write(results['size'])

        st.line_chart(pd.DataFrame(pd.DataFrame(results['history'])[['train_loss', 'val_loss']]))

        # report = convert_df(results['report'])
        # model_info = convert_df(results['size'])
        false_prediction = convert_df(results['false_prediction'])

        # st.download_button(
        #     'Press to download classification report',
        #     report,
        #     'report.csv',
        #     'text/csv',
        #     key='classification-report-download'
        # )
        #
        # st.download_button(
        #     'Press to download model information',
        #     model_info,
        #     'model_info.csv',
        #     'text/csv',
        #     key='model-info-download'
        # )

        st.download_button(
            'Press to download false prediction',
            false_prediction,
            'false_prediction.csv',
            'text/csv',
            key='false-prediction'
        )


