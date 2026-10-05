"""Command-line training, recommendation, and demo survey workflows."""

import argparse
import json
import pickle
import sys
from pathlib import Path

from .config import DEFAULT_DATA_DIR, DEFAULT_MODEL_PATH, PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Popcorn Picks movie recommender")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Evaluate and train an SVD model")
    train.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    train.add_argument("--min-ratings", type=int, default=50)
    train.add_argument("--n-factors", type=int, default=100)
    train.add_argument("--n-epochs", type=int, default=20)
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--max-rows",
        type=int,
        help="Read only the first N ratings for a quick smoke run",
    )

    recommend = commands.add_parser("recommend", help="Rank unrated movies for a user")
    recommend.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    recommend.add_argument("--user-id", type=int, required=True)
    recommend.add_argument("--n", type=int, default=5)
    recommend.add_argument("--genre")
    recommend.add_argument("--min-average-rating", type=float, default=0.0)

    survey = commands.add_parser(
        "generate-survey", help="Create a synthetic demo survey"
    )
    survey.add_argument("--users", type=int, default=50)
    survey.add_argument("--first-user-id", type=int, default=10000)
    survey.add_argument("--seed", type=int, default=42)
    survey.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "artifacts" / "survey.csv"
    )
    for command in (train, recommend, survey):
        command.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser


def run_command(args: argparse.Namespace) -> None:
    from .data import load_data, load_movies, load_ratings

    if args.command == "train":
        from .training import TrainingConfig, save_training_result, train_model

        movies = load_movies(args.data_dir)
        ratings = load_ratings(args.data_dir, max_rows=args.max_rows)
        ratings = ratings.loc[ratings["movieId"].isin(movies["movieId"])]
        config = TrainingConfig(
            min_ratings=args.min_ratings,
            n_factors=args.n_factors,
            n_epochs=args.n_epochs,
            test_size=args.test_size,
            seed=args.seed,
        )
        model, metrics = train_model(ratings, config)
        report = save_training_result(model, metrics, args.model_path)
        print(json.dumps(metrics, indent=2))
        print(f"Model saved to {args.model_path}\nEvaluation report saved to {report}")
    elif args.command == "recommend":
        from .model import load_model
        from .recommender import recommend_movies

        model = load_model(args.model_path)
        data = load_data(args.data_dir)
        recommendations = recommend_movies(
            args.user_id,
            model,
            data,
            n=args.n,
            genre=args.genre,
            min_average_rating=args.min_average_rating,
        )
        if recommendations.empty:
            print("No unrated movies match these filters.")
        else:
            columns = ["movieId", "title", "predicted_rating", "genre_score", "score"]
            print(recommendations[columns].to_string(index=False, float_format="%.3f"))
    else:
        from .survey import generate_survey

        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        survey = generate_survey(
            load_movies(args.data_dir),
            users=args.users,
            first_user_id=args.first_user_id,
            seed=args.seed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        survey.to_csv(args.output, index=False)
        print(f"Synthetic survey saved to {args.output}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_command(args)
    except (
        OSError,
        ValueError,
        ImportError,
        pickle.UnpicklingError,
        EOFError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0
