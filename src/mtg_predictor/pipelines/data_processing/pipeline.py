"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.0
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_mtg_json, process_atomic_cards


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_mtg_json,
                inputs="mtg_json",
                outputs="atomic_cards_json@json",
                name="preprocess_mtg_json",
            ),
            node(
                func=process_atomic_cards,
                inputs="atomic_cards_json@pandas",
                outputs="word_counts",
                name="word_counts",
            ),
        ]
    )
