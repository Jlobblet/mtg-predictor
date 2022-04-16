import logging
from collections import Counter
from importlib import import_module
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import unique_labels


def test_train_split(
    data: pd.DataFrame, target_column: str, params: dict
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"],
        data[target_column],
        test_size=params["test_size"],
        random_state=params["random_state"],
    )
    return X_train, X_test, y_train, y_test


def make_vectoriser(params: dict) -> TfidfVectorizer:
    return TfidfVectorizer(**params)


def make_selector(params: dict) -> SelectKBest:
    return SelectKBest(chi2, **params)


def make_classifier(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(**params)


def make_pipeline(vectoriser, selector, classifier) -> Pipeline:
    return Pipeline(
        [("vectoriser", vectoriser), ("selector", selector), ("classifier", classifier)]
    )


def fit_pipeline(
    pipeline: Pipeline, X_train: pd.Series, y_train: pd.Series
) -> Pipeline:
    return pipeline.fit(X_train, y_train)


def make_prediction(pipeline: Pipeline, X_test: pd.Series) -> pd.Series:
    return pd.Series(pipeline.predict(X_test), index=X_test.index, name="prediction")


def evaluate_model(prediction: pd.Series, X_test: pd.Series, y_test: pd.Series):
    df = pd.concat([X_test, prediction, y_test], axis=1)
    labels = unique_labels(y_test, prediction)

    logging.info(f"Prediction results:\n{df}")

    logging.info(f"Classification report:\n{classification_report(y_test, prediction, labels=labels)}")

    cm = confusion_matrix(y_test, prediction, normalize="true", labels=labels)
    logging.info(f"Confusion matrix:\n{cm}")
    return ConfusionMatrixDisplay(cm, display_labels=labels).plot().figure_
