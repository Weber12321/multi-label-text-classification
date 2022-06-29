import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(file_name: str):
    df = pd.read_json(file_name)
    return train_test_split(df, test_size=0.2, random_state=42)

