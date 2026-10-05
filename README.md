# Popcorn Picks

A movie recommendation project built with Python, MovieLens, Surprise SVD, and
Streamlit. Train and evaluate a model from the command line, generate personalized
recommendations, or explore the catalog in the web app. No notebooks are required.

## Project structure

```text
popcorn-picks/
├── app/
│   └── app.py                # Streamlit interface
├── src/popcorn_picks/
│   ├── cli.py                # Train, recommend, and generate-survey commands
│   ├── config.py             # Paths and survey column mapping
│   ├── data.py               # CSV validation and preprocessing
│   ├── model.py              # Model loading and atomic saving
│   ├── recommender.py        # Hybrid ranking
│   ├── survey.py             # Reproducible synthetic survey generation
│   └── training.py           # SVD training and held-out evaluation
├── tests/                    # Small fixtures; no 25M training run
├── datasets/
│   ├── ml-25m/               # Original CSVs and dataset documentation
│   └── survey.csv            # Synthetic favorite-genre preferences
├── model.pkl                 # Existing pretrained model (Git LFS)
├── pyproject.toml            # Package, dependencies, CLI, and development tools
└── requirements.txt          # Editable install of the package
```

## Setup

Use **Python 3.11 or 3.12**. Run these commands from the repository root.
The CSVs and pretrained model use Git LFS; enable it before downloading the
repository with [GitHub CLI](https://cli.github.com/).

```bash
git lfs install
gh repo clone danishb7/popcorn-picks
cd popcorn-picks
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the native build prerequisites first, then the package:

```bash
python -m pip install --upgrade pip
python -m pip install "setuptools>=68" wheel "Cython>=3,<4" "numpy>=1.26,<2"
python -m pip install --no-build-isolation -r requirements.txt
```

Surprise 1.1.4 builds a native extension. A C/C++ compiler is required, such as
Microsoft C++ Build Tools on Windows. The separate prerequisite step and
`--no-build-isolation` ensure it builds against the NumPy version used at runtime.
NumPy stays below version 2 for compatibility with the existing model. See the
[Surprise installation documentation](https://pypi.org/project/scikit-surprise/1.1.4/).

## Run the app

```bash
python -m streamlit run app/app.py
```

Enter a MovieLens user ID, select a genre or minimum community rating, and choose
**Find movies**. The app shows predicted ratings, hybrid scores, and real catalog
genres and average ratings. Its browse tab lists the most rated movies.

The app and CLI share the same data and recommendation functions. The app caches
the loaded resources; the full MovieLens dataset and pretrained model take time
to load and require substantial memory.

## Train and evaluate

Train a model without changing the supplied pretrained model:

```bash
popcorn-picks train --model-path artifacts/model.pkl
```

The pipeline validates the data, keeps users with at least 50 ratings, and
evaluates SVD on a reproducible 80/20 split. It reports **RMSE** and **MAE**, then
refits a fresh model on all retained ratings. Defaults match the original
notebook: 100 factors, 20 epochs, and random seed 42.

The command saves `artifacts/model.pkl` and `artifacts/model.metrics.json`.
For a quick run using the first 5,000 rating rows:

```bash
popcorn-picks train --max-rows 5000 --min-ratings 5 --n-factors 10 --n-epochs 2 --model-path artifacts/smoke-model.pkl
```

`--max-rows` takes the first rows, which are ordered by user in MovieLens. Use it
for smoke checks, not representative model evaluation. Run
`popcorn-picks train --help` for all options. With no `--model-path`, training
replaces `model.pkl`.

## Get recommendations from the CLI

```bash
popcorn-picks recommend --user-id 1 --n 5
popcorn-picks recommend --user-id 10000 --genre Action --min-average-rating 3.5
popcorn-picks recommend --user-id 1 --model-path artifacts/model.pkl
```

Every command also works as `python -m popcorn_picks`, for example:

```bash
python -m popcorn_picks recommend --user-id 1
```

The ranker excludes already-rated films and aligns predictions with catalog
movie IDs. For users with survey preferences, the score is 60% predicted rating
and 40% genre alignment. A favorite genre contributes 5/5; explicit
`action_rating`, `comedy_rating`, and `sci_fi_rating` columns can instead provide
0–5 weights. The strongest matching preference supplies the genre score, keeping
it on the same scale as predictions. Users without survey preferences use the
collaborative prediction alone. Average rating and genre filters apply before
ranking; ties are resolved by movie ID.

## Data and model paths

Default paths point to `datasets/` and `model.pkl` in the source checkout,
independent of the directory where the CLI runs. Use `--data-dir` and
`--model-path` to override them. For the app or shared defaults, set:

```powershell
$env:POPCORN_PICKS_DATA_DIR = "C:\path\to\datasets"
$env:POPCORN_PICKS_MODEL_PATH = "C:\path\to\artifacts\model.pkl"
python -m streamlit run app/app.py
```

```bash
POPCORN_PICKS_MODEL_PATH=artifacts/model.pkl python -m streamlit run app/app.py
```

Required files are `ml-25m/movies.csv` (`movieId,title,genres`) and
`ml-25m/ratings.csv` (`userId,movieId,rating`; timestamps are ignored).
`survey.csv` is optional. The supplied survey was generated synthetically and
does not establish real-world identity correspondence with MovieLens users.
Preferences are attached by `userId`, so meaningful preferences require matching
IDs.

Generate another repeatable demo survey without replacing the supplied file:

```bash
popcorn-picks generate-survey --users 50 --seed 42 --output artifacts/survey.csv
```

Only load trusted model files: Python pickle can execute code when loaded.
The original model format remains supported; retrain if an old pickle is
incompatible with your installed environment.

## Development

```bash
python -m pip install --no-build-isolation -e ".[dev]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

Tests use temporary CSVs and small SVD models to cover loading, filtering,
ranking, evaluation, serialization, CLI commands, and Streamlit interactions.
They do not retrain on the full MovieLens dataset or overwrite the supplied
model. Generated models, reports, bytecode, and virtual environments are ignored.

## Data sources

- [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) from GroupLens.
  Dataset terms and details are retained in `datasets/ml-25m/README.txt`.
- The local synthetic survey is for demonstration only.

Built with [Surprise](https://surprise.readthedocs.io/) and
[Streamlit](https://docs.streamlit.io/).
