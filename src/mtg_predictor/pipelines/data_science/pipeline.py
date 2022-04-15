"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.0
"""
from functools import partial, update_wrapper

from kedro.pipeline import Pipeline, node, pipeline

from .nodes.general import (
    evaluate_model,
    fit_pipeline,
    make_classifier,
    make_pipeline,
    make_prediction,
    make_selector,
    make_vectoriser,
    test_train_split,
)


def create_pipeline(**kwargs) -> Pipeline:
    target = "type"
    return pipeline(
        [
            node(
                func=lambda df, params: test_train_split(df, target, params),
                inputs=["atomic_cards", "params:split"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="test_train_split",
                tags=target,
            ),
            node(
                func=make_vectoriser,
                inputs="params:vectoriser",
                outputs="vectoriser",
                name="make_vectoriser",
                tags=target,
            ),
            node(
                func=make_selector,
                inputs="params:selector",
                outputs="selector",
                name="make_selector",
                tags=target,
            ),
            node(
                func=make_classifier,
                inputs="params:classifier",
                outputs="classifier",
                name="make_classifier",
                tags=target,
            ),
            node(
                func=make_pipeline,
                inputs=["vectoriser", "selector", "classifier"],
                outputs="pipeline",
                name="make_pipeline",
                tags=target,
            ),
            node(
                func=fit_pipeline,
                inputs=["pipeline", "X_train", "y_train"],
                outputs="pipeline_fitted",
                name="fit_pipeline",
                tags=target,
            ),
            node(
                func=make_prediction,
                inputs=["pipeline_fitted", "X_test"],
                outputs="predictions",
                name="make_prediction",
                tags=target,
            ),
            node(
                func=evaluate_model,
                inputs=["predictions", "X_test", "y_test"],
                outputs="confusion_matrix_display",
                name="evaluate_model",
                tags=target,
            ),
        ]
    )
