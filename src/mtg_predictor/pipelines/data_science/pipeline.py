"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.0
"""
from functools import partial, update_wrapper

from kedro.pipeline import Pipeline, node, pipeline

from .nodes.colour_identity import predict_prob
from .nodes.general import (
    evaluate_model,
    make_counters,
    predict,
    split_data,
    train_naive_bayes,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=lambda df: make_counters(df).set_index("name"),
                inputs="word_counts",
                outputs="word_counters",
                name="make_counters",
            ),
            node(
                func=lambda df, params: split_data(df, "colorIdentity", params),
                inputs=["word_counters", "params:split_ratio"],
                outputs=[
                    "colour_identity_X_train",
                    "colour_identity_X_test",
                    "colour_identity_y_train",
                    "colour_identity_y_test",
                ],
                name="colour_identity_split_data",
                tags="colour_identity",
            ),
            node(
                func=train_naive_bayes,
                inputs=["colour_identity_X_train", "colour_identity_y_train"],
                outputs="colour_identity_model",
                name="colour_identity_train_model",
                tags="colour_identity",
            ),
            node(
                func=lambda df, params: split_data(df, "type", params),
                inputs=["word_counters", "params:split_ratio"],
                outputs=["type_X_train", "type_X_test", "type_y_train", "type_y_test"],
                name="type_split_data",
                tags="type",
            ),
            node(
                func=train_naive_bayes,
                inputs=["type_X_train", "type_y_train"],
                outputs="type_model",
                name="type_train_model",
                tags="type",
            ),
            node(
                func=lambda m, x: predict(m, x, "colorIdentity"),
                inputs=["colour_identity_model", "colour_identity_X_test"],
                outputs="colour_identity_predictions",
                name="colour_identity_predict",
                tags="colour_identity",
            ),
            node(
                func=lambda m, x: predict(m, x, "type"),
                inputs=["type_model", "type_X_test"],
                outputs="type_predictions",
                name="type_predict",
                tags="type",
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "colour_identity_predictions",
                    "colour_identity_X_test",
                    "colour_identity_y_test",
                ],
                outputs=None,
                name="colour_identity_evaluate_model",
                tags="colour_identity",
            ),
            node(
                func=evaluate_model,
                inputs=["type_predictions", "type_X_test", "type_y_test"],
                outputs=None,
                name="type_evaluate_model",
                tags="type",
            ),
        ]
    )
