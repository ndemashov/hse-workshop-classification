import pandas as pd
import numpy as np
import src.config as cfg
from src.config import TARGET_COLS



def drop_unnecesary_id(df: pd.DataFrame) -> pd.DataFrame:
    if 'ID_y' in df.columns:
        df = df.drop('ID_y', axis=1)
    return df


def fill_sex(df: pd.DataFrame) -> pd.DataFrame:
    most_freq = df[cfg.SEX_COL].value_counts().index[0]
    df[cfg.SEX_COL] = df[cfg.SEX_COL].fillna(most_freq)
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.CAT_COLS] = df[cfg.CAT_COLS].astype('category')
    ohe_int_cols = df[cfg.OHE_COLS].select_dtypes('number').columns
    df[ohe_int_cols] = df[ohe_int_cols].astype(np.int8)

    df[cfg.REAL_COLS] = df[cfg.REAL_COLS].astype(np.float32)
    return df


def set_idx(df: pd.DataFrame, idx_col: str) -> pd.DataFrame:
    df = df.set_index(idx_col)
    return df

def add_ord_edu(df: pd.DataFrame) -> pd.DataFrame:
    df[f'{cfg.EDU_COL}_ord'] = df[cfg.EDU_COL].str.slice(0, 1).astype(np.int8).values
    return df
    
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = set_idx(df, cfg.ID_COL)
    df = drop_unnecesary_id(df)
    df = fill_sex(df)
    df = cast_types(df)
    df = df.dropna()
    return df


def preprocess_target(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.TARGET_COLS] = df[cfg.TARGET_COLS].astype(np.int8)
    return df


def extract_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, target = df.drop(cfg.TARGET_COLS, axis=1), df[cfg.TARGET_COLS]
    return df, target
