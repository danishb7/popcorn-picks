"""Hybrid ranking with predictions and genre scores aligned by movie ID."""

from typing import Any

import pandas as pd

from .config import GENRE_RATING_COLUMNS
from .data import MovieData


def genre_preferences(user_id: int, survey: pd.DataFrame) -> dict[str, float]:
    """Map favorite genres to 5/5 and accept optional explicit 0-5 genre scores."""
    rows = survey.loc[survey["userId"] == user_id]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    weights = {
        genre: float(row[column])
        for genre, column in GENRE_RATING_COLUMNS.items()
        if column in row and pd.notna(row[column])
    }
    favorite = row.get("fav_genre")
    if pd.notna(favorite) and favorite:
        weights.setdefault(str(favorite), 5.0)
    return weights


def recommend_movies(
    user_id: int,
    model: Any,
    data: MovieData,
    *,
    n: int = 5,
    genre: str | None = None,
    min_average_rating: float = 0.0,
    collaborative_weight: float = 0.6,
) -> pd.DataFrame:
    """Rank unrated movies using SVD plus the strongest matching genre preference.

    Both component scores use a 0-5 scale. Without survey preferences, use the
    collaborative prediction alone. Ties are resolved by movie ID for repeatability.
    """
    if user_id < 1 or n < 1:
        raise ValueError("user_id and n must be positive.")
    if not 0 <= collaborative_weight <= 1:
        raise ValueError("collaborative_weight must be between 0 and 1.")
    if not 0 <= min_average_rating <= 5:
        raise ValueError("min_average_rating must be between 0 and 5.")

    rated_ids = data.ratings.loc[data.ratings["userId"] == user_id, "movieId"]
    candidates = data.movies.loc[
        ~data.movies["movieId"].isin(rated_ids)
        & data.movies["avg_rating"].ge(min_average_rating)
    ].copy()
    if genre is not None:
        candidates = candidates.loc[candidates["genres"].map(lambda g: genre in g)]

    # Predict in catalog order, rather than assuming ratings and catalog rows align.
    candidates["predicted_rating"] = [
        model.predict(int(user_id), int(movie_id)).est
        for movie_id in candidates["movieId"]
    ]
    weights = genre_preferences(user_id, data.survey)
    candidates["genre_score"] = candidates["genres"].map(
        lambda genres: max((weights.get(g, 0.0) for g in genres), default=0.0)
    )
    if weights:
        candidates["score"] = (
            collaborative_weight * candidates["predicted_rating"]
            + (1 - collaborative_weight) * candidates["genre_score"]
        )
    else:
        candidates["score"] = candidates["predicted_rating"]
    return (
        candidates.sort_values(["score", "movieId"], ascending=[False, True])
        .head(n)
        .reset_index(drop=True)
    )
