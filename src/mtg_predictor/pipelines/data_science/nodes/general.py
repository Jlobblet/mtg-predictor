import logging
from collections import Counter
from typing import List, Tuple

import pandas as pd
from nltk import ClassifierI, NaiveBayesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def make_counters(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["name", "colorIdentity", "manaValue", "type"])
        .agg({"word": list, "count": list})
        .apply(lambda x: Counter(zip(x["word"], x["count"])), axis=1)
        .rename("word_counts")
        .reset_index()
    )


def split_data(
    data: pd.DataFrame, target: str, parameters: dict
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """This function splits the data into training and test data.

    Args:
        data: Data containing word counts and colour identity.
        target: The name of the column containing the target values.
        parameters: Parameters defined in parameters/data_science.yml.

    Returns:
        X_train: Training data
        X_test: Test data
        y_train: Training labels
        y_test: Test labels
    """
    logging.info(f"Columns:\n{data.columns}")
    X = data["word_counts"]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    logging.info(f"y_test type: {type(y_test)}")
    return X_train, X_test, y_train, y_test


def train_naive_bayes(X_train: pd.Series, y_train: pd.Series) -> ClassifierI:
    logging.info(f"X_train:\n{X_train}")
    logging.info(f"y_train:\n{y_train}")
    return NaiveBayesClassifier.train(list(zip(X_train, y_train)))


def predict(model: ClassifierI, X_test: pd.Series, name: str) -> pd.Series:
    prediction = model.classify_many(X_test)
    prediction = pd.Series(prediction, index=X_test.index, name=name)
    logging.info(f"Prediction:\n{prediction}")
    return prediction


def evaluate_model(prediction: pd.Series, X_test: pd.Series, y_test: pd.Series):
    logging.info("Combining series")
    X_test = pd.concat([X_test, prediction, y_test], axis=1)
    logging.info(f"Prediction results:\n{X_test}")
    logging.info(f"Classification report:\n{classification_report(y_test, prediction)}")
    logging.info(f"Confusion matrix:\n{confusion_matrix(y_test, prediction)}")
