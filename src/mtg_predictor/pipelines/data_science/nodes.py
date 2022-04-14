"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.0
"""
import logging
from collections import Counter
from typing import Tuple

import pandas as pd
from nltk import MultiClassifierI, ClassifierI
from sklearn.model_selection import train_test_split
from nltk.classify import NaiveBayesClassifier


def expand_labels(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(colorIdentity=lambda x: x["colorIdentity"].apply(list))


def make_counters(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["name", "colorIdentity", "manaValue"])
        .agg({"word": list, "count": list})
        .apply(lambda x: Counter(zip(x["word"], x["count"])), axis=1)
        .rename("word_counts")
        .reset_index()
    )


def split_data(data: pd.DataFrame, parameters: dict) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
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
    data = make_counters(data)
    # data = expand_labels(data)
    X = data["word_counts"]
    y = data["colorIdentity"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    logging.info(f"y_test type: {type(y_test)}")
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.Series, y_train: pd.Series) -> ClassifierI:
    logging.info(f"X_train:\n{X_train}")
    logging.info(f"y_train:\n{y_train}")
    return NaiveBayesClassifier.train(zip(X_train, y_train))


def predict(model: ClassifierI, X_test: pd.Series, y_test: pd.Series) -> pd.Series:
    prediction = model.classify_many(X_test)
    X_test = X_test.to_frame("X_test").assign(prediction=prediction, actual=y_test)
    logging.info(f"Prediction results:\n{X_test}")
    return pd.Series(prediction, name="colorIdentity")


def evaluate_model(prediction: pd.Series, y_test: pd.Series) -> float:
    logging.info(f"prediction:\n{prediction}")
    logging.info(f"y_test:\n{y_test}")
    correct_proportion = sum(prediction.to_numpy() == y_test.to_numpy()) / len(prediction)
    logging.info(f"Correct proportion: {correct_proportion}")
    return correct_proportion

