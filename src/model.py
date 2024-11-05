from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data and return RMSE.
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse, y_pred

def save_model(model, file_path="model.pkl"):
    """
    Save the trained model to a file.
    """
    with open(file_path, "wb") as f:
        pickle.dump(model, f)

def load_model(file_path="model.pkl"):
    """
    Load a saved model from a file.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)

