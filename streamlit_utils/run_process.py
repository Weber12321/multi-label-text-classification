import os.path

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from definition import DATA_DIR


@st.cache
def preprocess(file_name: str):
    path = os.path.join(DATA_DIR / file_name)
    df = pd.read_json(path)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    return df, df_train, df_test


@st.cache
def convert_df(df: pd.DataFrame):
    return df.to_csv().encode('utf-8')
