"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from .pipelines import classifier_model, data_processing


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    dp = data_processing.create_pipeline()
    classifier = classifier_model.create_pipeline
    colours = pipeline(
        pipe=classifier(target="colorIdentity"),
        inputs="atomic_cards",
        namespace="colorIdentity",
    )
    types = pipeline(
        pipe=classifier(target="type"),
        inputs="atomic_cards",
        namespace="type",
    )

    return {
        "__default__": dp + colours + types,
        "data_processing": dp,
        "colours": colours,
        "types": types,
    }
