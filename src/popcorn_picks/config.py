"""Default paths for an editable source checkout and environment overrides."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path(os.getenv("POPCORN_PICKS_DATA_DIR", PROJECT_ROOT / "datasets"))
DEFAULT_MODEL_PATH = Path(
    os.getenv("POPCORN_PICKS_MODEL_PATH", PROJECT_ROOT / "model.pkl")
)

GENRE_RATING_COLUMNS = {
    "Action": "action_rating",
    "Comedy": "comedy_rating",
    "Sci-Fi": "sci_fi_rating",
}
