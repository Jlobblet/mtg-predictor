"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.0
"""
colour_mapping = {
    "W": 1,
    "U": 2,
    "B": 4,
    "R": 8,
    "G": 16,
}


def _filter_entries(card: dict) -> dict:
    return {k: v for k, v in card.items() if k in ["name", "colorIdentity", "text"]}


def _parse_colour_identity(card: dict) -> dict:
    card: dict = card.copy()
    card["colorIdentity"] = sum(colour_mapping[c] for c in card["colorIdentity"])
    return card


def preprocess_mtg_json(mtg_json: dict) -> dict:
    return {
        k: _parse_colour_identity(_filter_entries(v[0]))
        for k, v in mtg_json["data"].items()
    }
