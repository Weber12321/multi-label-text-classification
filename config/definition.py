import os
from pathlib import Path, PurePath

ROOT_DIR = PurePath(__file__).parent.parent

LOGS_DIR = Path(ROOT_DIR / "logs")
Path(LOGS_DIR).mkdir(exist_ok=True)

DATA_DIR = Path(ROOT_DIR / "data")
Path(DATA_DIR).mkdir(exist_ok=True)

MODEL_DIR = Path(ROOT_DIR / "model")
Path(MODEL_DIR).mkdir(exist_ok=True)

MODEL_BIN_DIR = Path(MODEL_DIR / "bin")
Path(MODEL_BIN_DIR).mkdir(exist_ok=True)

MODEL_FP_DIR = Path(MODEL_DIR / 'false_pred')
Path(MODEL_FP_DIR).mkdir(exist_ok=True)

TOKEN_DIR = Path(MODEL_DIR / "tokenizer")
Path(TOKEN_DIR).mkdir(exist_ok=True)

MODEL_PT_DIR = Path(MODEL_DIR / "torch_script")
Path(MODEL_PT_DIR).mkdir(exist_ok=True)

AUDIENCE_BERT_DIR = Path(MODEL_FP_DIR / "audience_bert")
Path(AUDIENCE_BERT_DIR).mkdir(exist_ok=True)

