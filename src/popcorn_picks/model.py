"""Persistence for trusted Surprise models, including the existing model.pkl."""

import os
import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .config import DEFAULT_MODEL_PATH


def load_model(path: Path = DEFAULT_MODEL_PATH) -> Any:
    """Load a trusted local pickle; pickle files can execute code when opened."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing model: {path}. Run 'popcorn-picks train --model-path {path}'."
        )
    with path.open("rb") as stream:
        if stream.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"{path} is a Git LFS pointer. Download the model first.")
        stream.seek(0)
        model = pickle.load(stream)
    if not callable(getattr(model, "predict", None)):
        raise ValueError(f"{path} does not contain a recommendation model.")
    return model


def save_model(model: Any, path: Path = DEFAULT_MODEL_PATH) -> None:
    """Write atomically so an interrupted save does not corrupt an existing model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
