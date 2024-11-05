import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load dataset from the specified file path.
    """
    return pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

def clean_data(df):
    """
    Clean the dataset by handling missing values.
    Fill missing values using forward fill method.
    """
    df.fillna(method='ffill', inplace=True)
    return df

def add_time_features(df):
    """
    Add time-based features to the dataset for time-series analysis.
    """
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    return df

def preprocess_data(file_path):
    """
    Full preprocessing pipeline: load, clean, and add features.
    """
    df = load_data(file_path)
    df = clean_data(df)
    df = add_time_features(df)
    return df

