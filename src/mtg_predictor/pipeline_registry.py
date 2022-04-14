"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from .pipelines import data_processing


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    dp = data_processing.create_pipeline()

    return {"__default__": dp, "data_processing": dp}