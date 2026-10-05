"""Generate a repeatable demo survey, replacing notebook-only sample data code."""

import numpy as np
import pandas as pd


def generate_survey(
    movies: pd.DataFrame,
    *,
    users: int = 50,
    first_user_id: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Return synthetic preferences; IDs must be aligned with ratings by the caller."""
    if users < 1 or first_user_id < 1 or movies.empty:
        raise ValueError(
            "users, first_user_id, and the movie catalog must be nonempty."
        )
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "userId": np.arange(first_user_id, first_user_id + users),
            "age": rng.integers(18, 65, users),
            "fav_genre": rng.choice(["Action", "Comedy", "Sci-Fi"], users),
            "fav_movie_id": rng.choice(movies["movieId"].to_numpy(), users),
        }
    )
