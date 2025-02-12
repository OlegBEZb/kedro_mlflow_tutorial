import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def select_training_columns(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Selects columns from multiple DataFrames.

    Args:
        datasets: Input DataFrames.
        features: List of columns to select.
    Returns:
        DataFrame with selected columns.
    """
    return df[features]


def extract_target(data: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Extracts the target column from the DataFrame.

    Args:
        data: Input DataFrame.
    Returns:
        Series with target values.
    """
    return data[[target_column]]


def fit_numerical_features_scaler(data: pd.DataFrame) -> MinMaxScaler:
    """Fits a MinMaxScaler to normalize all numerical features.

    Args:
        data: Input DataFrame.

    Returns:
        Fitted MinMaxScaler.
    """
    numerical_features = data.select_dtypes(include=["number"]).columns.tolist()

    scaler = MinMaxScaler()
    scaler.fit(data[numerical_features])
    return scaler


def transform_numerical_features(data: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    """Applies normalization to all detected numerical features using a fitted MinMaxScaler.

    Args:
        data: Input DataFrame.
        scaler: Fitted MinMaxScaler.

    Returns:
        DataFrame with normalized numerical features.
    """
    numerical_features = data.select_dtypes(include=["number"]).columns.tolist()

    # Transform numerical features
    normalized_data = scaler.transform(data[numerical_features])
    normalized_df = pd.DataFrame(normalized_data, columns=numerical_features, index=data.index)

    # Update the original DataFrame with normalized values
    data.update(normalized_df)
    return data


def generate_numerical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Generates square, log, and square root of all numerical features.

    Args:
        data: Input DataFrame.

    Returns:
        DataFrame with additional numerical features.
    """
    numerical_features = data.select_dtypes(include=["number"]).columns.tolist()

    for feature in numerical_features:
        data[f"{feature}_squared"] = (data[feature] ** 2).round(3)
        data[f"{feature}_log"] = data[feature].apply(lambda x: x if x == 0 else np.log(x)).round(3)
        data[f"{feature}_sqrt"] = (data[feature] ** 0.5).round(3)
    return data


def split_data(X: pd.DataFrame, y: pd.DataFrame, split_parameters: dict) -> tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
    Returns:
        Split data.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_parameters["test_size"], random_state=split_parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict) -> LinearRegression:
    """Trains the linear regression model with given parameters.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.
        parameters: Dictionary of model parameters.

    Returns:
        Trained model.
    """
    regressor = LinearRegression(**parameters)
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(
    regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
    return round(score, 3)


def predict_node(
    regressor: LinearRegression,
    new_data_to_predict: pd.DataFrame,
) -> pd.DataFrame:
    return regressor.predict(new_data_to_predict)
