import pandas as pd
import numpy as np
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin


class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds_ = None
        self.upper_bounds_ = None
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.to_numpy()
            X_df = X
        else:
            self.feature_names_in_ = None
            X_df = pd.DataFrame(X)
        q1 = X_df.quantile(0.25)
        q3 = X_df.quantile(0.75)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - self.factor * iqr
        self.upper_bounds_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        X_clipped = X_df.clip(self.lower_bounds_, self.upper_bounds_, axis=1)
        return X_clipped.values

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if self.feature_names_in_ is not None:
            return np.asarray(self.feature_names_in_, dtype=object)
        return np.asarray([f"x{i}" for i in range(len(self.lower_bounds_))], dtype=object)


def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None


def basic_clean(df):
    if df is None:
        return None, 0
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    return df, removed


def drop_missing_target_rows(df, target_col):
    if df is None or target_col not in df.columns:
        return df, 0
    missing_mask = df[target_col].isna()
    dropped = int(missing_mask.sum())
    if dropped:
        df = df.loc[~missing_mask].copy()
    return df, dropped


def drop_missing_target_xy(X, y):
    missing_mask = y.isna()
    dropped = int(missing_mask.sum())
    if dropped:
        X = X.loc[~missing_mask]
        y = y.loc[~missing_mask]
    return X, y, dropped


def infer_problem_type(df, target_col):
    if df is None or target_col not in df.columns:
        return "Unknown"
    target = df[target_col]
    unique_count = target.dropna().nunique()
    if target.dtype.kind in "ifu":
        if unique_count <= 10:
            return "Classification"
        return "Regression"
    return "Classification"


def detect_columns(df, target_col):
    numeric_cols = df.drop(columns=[target_col]).select_dtypes(
        include=[np.number]).columns.tolist()
    categorical_cols = df.drop(columns=[target_col]).select_dtypes(
        exclude=[np.number]).columns.tolist()

    # Treat low-cardinality numeric columns as categorical if user wants
    low_card = []
    for col in numeric_cols:
        if df[col].nunique() < 10:
            low_card.append(col)
    return numeric_cols, categorical_cols, low_card
