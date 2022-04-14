"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.0
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data, evaluate_model, train_model, predict


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["word_counts", "params:split_ratio"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="model",
                name="train_model",
            ),
            node(
                func=predict,
                inputs=["model", "X_test", "y_test"],
                outputs="predictions",
                name="predict",
            ),
            node(
                func=evaluate_model,
                inputs=["predictions", "y_test"],
                outputs="accuracy",
                name="evaluate_model",
            )
        ]
    )
