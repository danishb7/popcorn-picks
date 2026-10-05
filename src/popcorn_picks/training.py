"""Reproducible SVD training and held-out evaluation, formerly in Project.ipynb."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

from .data import filter_active_users
from .model import save_model


@dataclass(frozen=True)
class TrainingConfig:
    min_ratings: int = 50
    n_factors: int = 100
    n_epochs: int = 20
    test_size: float = 0.2
    seed: int = 42


def train_model(
    ratings: pd.DataFrame, config: TrainingConfig = TrainingConfig()
) -> tuple[Any, dict]:
    """Evaluate on held-out ratings, then refit on all retained ratings."""
    if config.n_factors < 1 or config.n_epochs < 1:
        raise ValueError("n_factors and n_epochs must be positive.")
    if not 0 < config.test_size < 1:
        raise ValueError("test_size must be between 0 and 1, exclusive.")
    filtered = filter_active_users(ratings, config.min_ratings)
    if len(filtered) < 2:
        raise ValueError("Too few ratings after filtering. Lower --min-ratings.")
    dataset = Dataset.load_from_df(
        filtered[["userId", "movieId", "rating"]], Reader(rating_scale=(0.5, 5.0))
    )
    trainset, testset = train_test_split(
        dataset, test_size=config.test_size, random_state=config.seed
    )
    if trainset.n_ratings == 0 or not testset:
        raise ValueError(
            "The train/test split must contain ratings in both partitions."
        )
    parameters = {
        "n_factors": config.n_factors,
        "n_epochs": config.n_epochs,
        "random_state": config.seed,
    }
    evaluation_model = SVD(**parameters).fit(trainset)
    predictions = evaluation_model.test(testset)
    metrics = {
        "rmse": accuracy.rmse(predictions, verbose=False),
        "mae": accuracy.mae(predictions, verbose=False),
        "evaluation_train_ratings": trainset.n_ratings,
        "evaluation_test_ratings": len(testset),
        "final_training_ratings": len(filtered),
        "users": int(filtered["userId"].nunique()),
        "movies": int(filtered["movieId"].nunique()),
        "config": asdict(config),
    }
    model = SVD(**parameters).fit(dataset.build_full_trainset())
    return model, metrics


def save_training_result(model: Any, metrics: dict, path: Path) -> Path:
    """Save the fitted model and a separate, readable evaluation report."""
    save_model(model, path)
    report_path = Path(path).with_suffix(".metrics.json")
    report_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return report_path
