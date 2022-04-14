"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.0
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["word_counts", "params:split_ratio"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data",
            )
        ]
    )
