from collections import defaultdict
from datetime import datetime

import torch
import wandb
import streamlit as st
from stqdm import stqdm
from torch.nn import BCEWithLogitsLoss
# from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler

from evaluation.bert_pytorch import eval_epoch
from metrics.model_size import get_model_size
from metrics.nultilabel_metrics import get_classification_report
from training.bert_pytorch import train_epoch
from workers.datasets_builder.loader import create_data_loader


@st.cache
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
    pass
