"""Load and validate MovieLens CSVs without loading unused columns."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DEFAULT_DATA_DIR, GENRE_RATING_COLUMNS


@dataclass(frozen=True)
class MovieData:
    """Prepared catalog, ratings, and optional survey preferences."""

    movies: pd.DataFrame
    ratings: pd.DataFrame
    survey: pd.DataFrame


def _require_csv(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset: {path}")
    with path.open("rb") as stream:
        if stream.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(
                f"{path} is a Git LFS pointer. Download the dataset first."
            )


def load_movies(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load movie IDs, titles, and genres as lists; discard the no-genre marker."""
    path = Path(data_dir) / "ml-25m" / "movies.csv"
    _require_csv(path)
    movies = pd.read_csv(path, usecols=["movieId", "title", "genres"])
    if movies.empty or movies["movieId"].isna().any():
        raise ValueError("The movie catalog must contain valid movie IDs.")
    if movies["movieId"].duplicated().any():
        raise ValueError("The movie catalog contains duplicate movie IDs.")
    movies["genres"] = (
        movies["genres"]
        .fillna("")
        .map(
            lambda value: [
                genre
                for genre in value.split("|")
                if genre and genre != "(no genres listed)"
            ]
        )
    )
    return movies


def load_ratings(
    data_dir: Path = DEFAULT_DATA_DIR, *, max_rows: int | None = None
) -> pd.DataFrame:
    """Read only the three columns needed for training and recommendations."""
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive.")
    path = Path(data_dir) / "ml-25m" / "ratings.csv"
    _require_csv(path)
    ratings = pd.read_csv(
        path,
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
        nrows=max_rows,
    )
    if ratings.empty or not ratings["rating"].between(0.5, 5.0).all():
        raise ValueError("Ratings must be nonempty and between 0.5 and 5.0.")
    return ratings


def load_survey(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Accept the supplied favorite-genre survey or numeric genre preferences."""
    path = Path(data_dir) / "survey.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["userId", "fav_genre"])
    _require_csv(path)
    survey = pd.read_csv(path)
    if "userId" not in survey or survey["userId"].isna().any():
        raise ValueError("Survey data must contain valid userId values.")
    if survey["userId"].duplicated().any():
        raise ValueError("Survey data must have at most one row per user.")
    weight_columns = set(GENRE_RATING_COLUMNS.values()).intersection(survey.columns)
    if "fav_genre" not in survey and not weight_columns:
        raise ValueError("Survey data needs fav_genre or numeric genre rating columns.")
    for column in weight_columns:
        survey[column] = pd.to_numeric(survey[column], errors="raise")
        if not survey[column].dropna().between(0, 5).all():
            raise ValueError(f"Survey column {column} must contain scores from 0 to 5.")
    return survey


def filter_active_users(ratings: pd.DataFrame, min_ratings: int = 50) -> pd.DataFrame:
    """Keep users with at least min_ratings observations, as in the notebook."""
    if min_ratings < 1:
        raise ValueError("min_ratings must be positive.")
    counts = ratings["userId"].value_counts()
    active_users = counts[counts >= min_ratings].index
    return ratings.loc[ratings["userId"].isin(active_users)].copy()


def load_data(data_dir: Path = DEFAULT_DATA_DIR) -> MovieData:
    """Prepare a catalog with actual average ratings and rating counts."""
    movies = load_movies(data_dir)
    ratings = load_ratings(data_dir)
    ratings = ratings.loc[ratings["movieId"].isin(movies["movieId"])]
    if ratings.empty:
        raise ValueError("No ratings match the movie catalog.")
    summary = ratings.groupby("movieId")["rating"].agg(
        avg_rating="mean", rating_count="size"
    )
    movies = movies.merge(summary, on="movieId", how="left", validate="one_to_one")
    movies["rating_count"] = movies["rating_count"].fillna(0).astype("int32")
    return MovieData(movies, ratings, load_survey(data_dir))
