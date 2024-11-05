import pandas as pd
from sklearn.model_selection import train_test_split

def train_test_split_time_series(df, target_column, test_size=0.2):
    """
    Split the dataset into training and testing sets for time-series data.
    """
    split_index = int(len(df) * (1 - test_size))
    train_df = df[:split_index]
    test_df = df[split_index:]
    X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]
    X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]
    return X_train, X_test, y_train, y_test

def save_metrics(metrics, file_path="metrics.json"):
    """
    Save evaluation metrics as a JSON file.
    """
    import json
    with open(file_path, "w") as f:
        json.dump(metrics, f)

def load_metrics(file_path="metrics.json"):
    """
    Load evaluation metrics from a JSON file.
    """
    import json
    with open(file_path, "r") as f:
        return json.load(f)

