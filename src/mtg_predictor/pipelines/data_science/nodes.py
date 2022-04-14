"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.0
"""
import logging
from collections import Counter
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def expand_labels(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(colorIdentity=lambda x: x["colorIdentity"].apply(list))


def make_counters(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["name", "colorIdentity", "manaValue"])
        .agg({"word": list, "count": list})
        .apply(lambda x: Counter(zip(x["word"], x["count"])), axis=1)
    )


def split_data(data: pd.DataFrame, parameters: dict) -> Tuple:
    """This function splits the data into training and test data.

    Args:
        data: Data containing word counts and colour identity
        parameters: Parameters defined in parameters/data_science.yml.

    Returns:
        X_train: Training data
        X_test: Test data
        y_train: Training labels
        y_test: Test labels
    """
    logging.info(data.head())
    data = make_counters(data)
    logging.info(data.head())
    data = expand_labels(data)
    logging.info(data.head())
    X = data.drop(columns=["colorIdentity"])
    y = data["colorIdentity"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: dict
) -> Tuple:
    pass
