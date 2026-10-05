import pandas as pd
import pytest

from popcorn_picks.data import filter_active_users, load_data, load_ratings, load_survey


def test_catalog_uses_real_statistics_and_genres(movie_data):
    catalog = movie_data.movies.set_index("movieId")
    assert catalog.loc[10, "genres"] == ["Action", "Drama"]
    assert catalog.loc[10, "avg_rating"] == pytest.approx(9.5 / 3)
    assert catalog.loc[10, "rating_count"] == 3
    assert catalog.loc[50, "genres"] == []
    assert pd.isna(catalog.loc[50, "avg_rating"])
    assert catalog.loc[50, "rating_count"] == 0
    assert set(movie_data.ratings.columns) == {"userId", "movieId", "rating"}


def test_active_user_threshold_is_inclusive(movie_data):
    filtered = filter_active_users(movie_data.ratings, min_ratings=4)
    assert set(filtered["userId"]) == {2, 3}
    assert len(filtered) == 8


def test_missing_survey_is_optional(data_dir):
    (data_dir / "survey.csv").unlink()
    assert load_survey(data_dir).empty


def test_lfs_pointer_has_actionable_error(data_dir):
    (data_dir / "ml-25m" / "ratings.csv").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:example\nsize 100\n"
    )
    with pytest.raises(ValueError, match="Git LFS pointer"):
        load_ratings(data_dir)


def test_rating_range_is_validated(data_dir):
    path = data_dir / "ml-25m" / "ratings.csv"
    ratings = pd.read_csv(path)
    ratings.loc[0, "rating"] = 8
    ratings.to_csv(path, index=False)
    with pytest.raises(ValueError, match="between 0.5 and 5.0"):
        load_ratings(data_dir)


def test_loading_is_independent_of_current_directory(data_dir, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert len(load_data(data_dir).movies) == 5
    assert len(load_ratings(data_dir, max_rows=2)) == 2


def test_explicit_survey_scores_are_validated(data_dir):
    pd.DataFrame({"userId": [1], "action_rating": [6]}).to_csv(
        data_dir / "survey.csv", index=False
    )
    with pytest.raises(ValueError, match="scores from 0 to 5"):
        load_survey(data_dir)
