"""Small MovieLens-shaped fixtures that do not depend on repository datasets."""

import pandas as pd
import pytest

from popcorn_picks.data import load_data


@pytest.fixture
def data_dir(tmp_path):
    directory = tmp_path / "datasets"
    (directory / "ml-25m").mkdir(parents=True)
    pd.DataFrame(
        {
            "movieId": [30, 10, 40, 20, 50],
            "title": ["Comedy", "Seen movie", "Drama", "Action", "Unrated"],
            "genres": [
                "Comedy",
                "Action|Drama",
                "Drama",
                "Action",
                "(no genres listed)",
            ],
        }
    ).to_csv(directory / "ml-25m" / "movies.csv", index=False)
    pd.DataFrame(
        [
            (1, 10, 4.0),
            (2, 30, 4.0),
            (2, 10, 3.0),
            (2, 40, 3.5),
            (2, 20, 4.5),
            (3, 30, 4.0),
            (3, 10, 2.5),
            (3, 40, 4.0),
            (3, 20, 5.0),
        ],
        columns=["userId", "movieId", "rating"],
    ).assign(timestamp=123).to_csv(directory / "ml-25m" / "ratings.csv", index=False)
    pd.DataFrame({"userId": [1], "fav_genre": ["Action"]}).to_csv(
        directory / "survey.csv", index=False
    )
    return directory


@pytest.fixture
def movie_data(data_dir):
    return load_data(data_dir)
