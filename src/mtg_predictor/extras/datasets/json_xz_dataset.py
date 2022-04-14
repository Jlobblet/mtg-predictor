import json
from lzma import compress, decompress
from pathlib import Path
from typing import Any, Dict

import fsspec
from kedro.io import AbstractDataSet
from kedro.io.core import get_filepath_str, get_protocol_and_path


class JsonXzDataSet(AbstractDataSet):
    def __init__(self, filepath: str):
        """Creates a new instance of the JsonXzDataSet to load / save .json.xz files for a given filepath.

        Args:
            filepath: The filepath to the .json.xz file.
        """
        # Parse the filepath into protocol and path
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = Path(path)
        self._fs = fsspec.filesystem(self._protocol)

    def _load(self) -> dict:
        """Loads the data from the filepath.

        Returns:
            The loaded data as a Python dictionary.
        """
        load_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(load_path, "rb") as f:
            return json.loads(decompress(f.read()))

    def _save(self, data: Any) -> None:
        save_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(save_path, "wb") as f:
            f.write(compress(json.dumps(data).encode()))

    def _describe(self) -> Dict[str, Any]:
        return {
            "protocol": self._protocol,
            "filepath": self._filepath,
        }
