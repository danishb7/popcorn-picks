from types import SimpleNamespace

import pandas as pd
import pytest

from popcorn_picks.data import MovieData
from popcorn_picks.recommender import genre_preferences, recommend_movies


class FixedModel:
    """Distinct predictions reveal accidental positional alignment with ratings."""

    def predict(self, user_id, movie_id):
        return SimpleNamespace(est={10: 5.0, 20: 3.0, 30: 4.0, 40: 2.0}[movie_id])


def test_ranking_aligns_predictions_and_excludes_seen_movies(movie_data):
    result = recommend_movies(1, FixedModel(), movie_data)
    assert result["movieId"].tolist() == [20, 30, 40]
    assert result["title"].tolist() == ["Action", "Comedy", "Drama"]
    assert result["score"].tolist() == pytest.approx([3.8, 2.4, 1.2])
    assert len(movie_data.movies) == 5


def test_no_survey_uses_collaborative_scores(movie_data):
    result = recommend_movies(99, FixedModel(), movie_data, n=2)
    assert result["movieId"].tolist() == [10, 30]
    assert result["score"].tolist() == [5.0, 4.0]


def test_filters_apply_before_ranking(movie_data):
    result = recommend_movies(
        1, FixedModel(), movie_data, genre="Action", min_average_rating=4.0
    )
    assert result["movieId"].tolist() == [20]
    assert recommend_movies(1, FixedModel(), movie_data, genre="Fantasy").empty


def test_zero_recommendations_is_rejected(movie_data):
    with pytest.raises(ValueError, match="must be positive"):
        recommend_movies(1, FixedModel(), movie_data, n=0)


def test_explicit_preferences_keep_hybrid_scores_on_rating_scale(movie_data):
    survey = pd.DataFrame(
        {"userId": [1], "action_rating": [4.0], "comedy_rating": [2.0]}
    )
    assert genre_preferences(1, survey) == {"Action": 4.0, "Comedy": 2.0}
    result = recommend_movies(
        1, FixedModel(), MovieData(movie_data.movies, movie_data.ratings, survey)
    )
    assert result["score"].between(0, 5).all()


def test_ties_are_sorted_by_movie_id(movie_data):
    class TiedModel:
        def predict(self, user_id, movie_id):
            return SimpleNamespace(est=3.0)

    result = recommend_movies(99, TiedModel(), movie_data)
    assert result["movieId"].tolist() == [10, 20, 30, 40]
