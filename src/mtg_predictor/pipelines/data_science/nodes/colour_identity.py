import logging

import pandas as pd
from nltk import ClassifierI


def expand_labels(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(colorIdentity=lambda x: x["colorIdentity"].apply(list))


def predict_prob(
    model: ClassifierI, X_test: pd.Series, y_test: pd.Series
) -> pd.DataFrame:
    prediction = model.prob_classify_many(X_test)
    X_test = X_test.to_frame("X_test").assign(
        prediction=[{s: f"{p.prob(s):.3f}" for s in p.samples()} for p in prediction],
        actual=y_test,
    )
    with pd.option_context("display.max_colwidth", 100):
        logging.info(f"Prediction results:\n{X_test}")
    return X_test
