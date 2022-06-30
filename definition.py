from pathlib import Path, PurePath

ROOT_DIR = PurePath(__file__).parent

LOGS_DIR = Path(ROOT_DIR / "logs")
Path(LOGS_DIR).mkdir(exist_ok=True)

DATA_DIR = Path(ROOT_DIR / "data")
Path(DATA_DIR).mkdir(exist_ok=True)

MODEL_DIR = Path(ROOT_DIR / "models")
Path(MODEL_DIR).mkdir(exist_ok=True)

SQL_DIR = Path(ROOT_DIR / "sql")
Path(SQL_DIR).mkdir(exist_ok=True)

FALSE_PRED = Path(ROOT_DIR / "false_pred")
Path(FALSE_PRED).mkdir(exist_ok=True)

REPORT = Path(ROOT_DIR / "reports")
Path(REPORT).mkdir(exist_ok=True)
