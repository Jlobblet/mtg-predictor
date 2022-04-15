"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.0
"""
import logging
from typing import Any, List

import pandas as pd
from nltk import PorterStemmer
from nltk.corpus import stopwords


def _filter_entries(card: dict) -> dict:
    """Remove entries from `card` that are not needed for the analysis."""
    return {
        k: v
        for k, v in card.items()
        if k == "text" or k in ["name", "colorIdentity", "manaValue", "types"]
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
    logging.info(f"\n{x}" if isinstance(x, (pd.DataFrame, pd.Series)) else x)
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
    # Join the colour identity into a single string
    atomic_cards["colorIdentity"] = atomic_cards["colorIdentity"].apply(
        lambda x: "".join(x)
    )

    # Common words to remove
    stop_words = set(stopwords.words("english"))

    # Stemming
    stemmer = PorterStemmer()

    # Sanitise the text
    atomic_cards["word_counts"] = (
        atomic_cards["text"]
        .str.replace(r"[.,:;\"()\[\]!?\-_]", "", regex=True)
        .str.lower()
        .str.split()
        .dropna()  # Drop cards with no text
        .apply(lambda x: [stemmer.stem(word) for word in x if word not in stop_words])
    )

    # Remove cards with no text
    atomic_cards = atomic_cards[atomic_cards["word_counts"].notna()]

    # Filter to monocolour or colourless cards cards
    atomic_cards["colorIdentity"] = atomic_cards["colorIdentity"].apply(
        lambda c: c or "C"
    )
    atomic_cards["color_counts"] = atomic_cards["colorIdentity"].str.len()
    atomic_cards = atomic_cards[atomic_cards["color_counts"] == 1]
    atomic_cards["colorIdentity"] = atomic_cards["colorIdentity"].str[0]

    # Convert the types column to a set
    atomic_cards["types"] = atomic_cards["types"].apply(
        lambda ts: set(t.lower() for t in ts)
    )
    # Filter to types we care about
    atomic_cards["types"] = atomic_cards["types"].apply(
        lambda s: s
        & {
            "creature",
            "instant",
            "sorcery",
            "enchantment",
            "artifact",
            "land",
            "planeswalker",
        }
    )
    # Get the first element of types
    atomic_cards["type"] = atomic_cards["types"].apply(
        lambda x: x.pop() if x else pd.NA
    )
    # Remove cards with no matching type
    atomic_cards = atomic_cards[atomic_cards["type"].notna()]

    return atomic_cards
