import json

import pandas as pd
import pytest

from popcorn_picks.cli import main
from popcorn_picks.model import load_model
from popcorn_picks.survey import generate_survey
from popcorn_picks.training import TrainingConfig, save_training_result, train_model


def test_evaluation_refit_and_serialization(movie_data, tmp_path):
    config = TrainingConfig(min_ratings=1, n_factors=3, n_epochs=2)
    model, metrics = train_model(movie_data.ratings, config)
    assert metrics["evaluation_train_ratings"] + metrics["evaluation_test_ratings"] == 9
    assert model.trainset.n_ratings == metrics["final_training_ratings"] == 9
    assert 0 <= metrics["rmse"] <= 5
    path = tmp_path / "artifacts" / "model.pkl"
    report_path = save_training_result(model, metrics, path)
    loaded = load_model(path)
    assert loaded.predict(1, 20).est == pytest.approx(model.predict(1, 20).est)
    assert json.loads(report_path.read_text())["config"]["seed"] == 42
    repeated, repeated_metrics = train_model(movie_data.ratings, config)
    assert repeated_metrics == metrics
    assert repeated.predict(1, 20).est == pytest.approx(model.predict(1, 20).est)


def test_training_with_no_active_users_is_actionable(movie_data):
    with pytest.raises(ValueError, match="Lower --min-ratings"):
        train_model(movie_data.ratings)


def test_cli_training_and_recommendation(data_dir, tmp_path, capsys):
    path = tmp_path / "model.pkl"
    common = ["--data-dir", str(data_dir), "--model-path", str(path)]
    assert main(["train", *common, "--min-ratings", "1", "--n-epochs", "2"]) == 0
    assert path.is_file()
    assert main(["recommend", *common, "--user-id", "1", "--genre", "Action"]) == 0
    output = capsys.readouterr().out
    assert "Evaluation report saved" in output
    assert "Action" in output
    assert "Seen movie" not in output


def test_cli_missing_data_returns_error(tmp_path, capsys):
    assert main(["train", "--data-dir", str(tmp_path)]) == 1
    assert "Missing dataset" in capsys.readouterr().err


def test_synthetic_survey_is_reproducible_and_does_not_overwrite(
    movie_data, data_dir, tmp_path, capsys
):
    pd.testing.assert_frame_equal(
        generate_survey(movie_data.movies), generate_survey(movie_data.movies)
    )
    path = tmp_path / "survey.csv"
    args = ["generate-survey", "--data-dir", str(data_dir), "--output", str(path)]
    assert main(args) == 0
    original = path.read_bytes()
    assert main(args) == 1
    assert "Output already exists" in capsys.readouterr().err
    assert path.read_bytes() == original
