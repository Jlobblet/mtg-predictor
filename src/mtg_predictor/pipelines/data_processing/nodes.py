"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.0
"""
import logging
from typing import Any, List

import pandas as pd


def _filter_entries(card: dict) -> dict:
    """Remove entries from `card` that are not needed for the analysis."""
    return {
        k: v
        for k, v in card.items()
        if k in ["name", "colorIdentity", "text", "manaValue"]
    }


def _split_words(text: str) -> List[str]:
    """Remove select punctuation, convert to folded case, and split into words."""
    if not isinstance(text, str):
        return []

    punctuation = ".,:;'\"()[]!?-_"
    maketrans = str.maketrans("", "", punctuation)
    return text.translate(maketrans).lower().split()


def _inspect(x: Any) -> Any:
    """Log an object and then return it unchanged."""
    logging.info(x)
    return x


def preprocess_mtg_json(mtg_json: dict) -> dict:
    return {k: _filter_entries(v[0]) for k, v in mtg_json["data"].items()}


def process_atomic_cards(atomic_cards: pd.DataFrame) -> pd.DataFrame:
    """For each row in `atomic_cards`, count the number of occurrences of each word in its `text` column.

    Args:
        atomic_cards: A pandas DataFrame containing atomic card data from MtG JSON.

    Returns:
        A pandas DataFrame containing the same data as `atomic_cards`, with a new column containing the word counts.
    """
    atomic_cards = atomic_cards.copy()
    atomic_cards["colorIdentity"] = atomic_cards["colorIdentity"].apply(
        lambda x: "".join(x)
    )
    atomic_cards["word_counts"] = (
        atomic_cards["text"]
        .str.replace(r"[.,:;\"()\[\]!?\-_]", "", regex=True)
        .str.lower()
        .str.split()
    )
    return (
        atomic_cards.explode("word_counts")
        .groupby(by=["name", "colorIdentity", "manaValue"])["word_counts"]
        .value_counts()
        .rename("count")
        .reset_index()
        .rename(columns={"word_counts": "word"})
    )
