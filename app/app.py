"""Streamlit presentation layer for the shared Popcorn Picks recommender."""

import pickle
from pathlib import Path

import streamlit as st

from popcorn_picks.config import DEFAULT_DATA_DIR, DEFAULT_MODEL_PATH
from popcorn_picks.data import MovieData, load_data
from popcorn_picks.model import load_model
from popcorn_picks.recommender import recommend_movies


@st.cache_resource
def cached_data(data_dir: str) -> MovieData:
    """Keep the large ratings table in memory without copying it on each rerun."""
    return load_data(Path(data_dir))


@st.cache_resource
def cached_model(model_path: str):
    return load_model(Path(model_path))


def render_recommendations(recommendations) -> None:
    if recommendations.empty:
        st.info("No unrated movies match these filters. Try another genre or rating.")
        return
    for movie in recommendations.itertuples(index=False):
        with st.container(border=True):
            st.subheader(movie.title)
            st.caption(" · ".join(movie.genres) or "Genres unavailable")
            left, middle, right = st.columns(3)
            left.metric("Predicted rating", f"{movie.predicted_rating:.2f} / 5")
            middle.metric("Recommendation score", f"{movie.score:.2f} / 5")
            right.metric("Community rating", f"{movie.avg_rating:.2f} / 5")


def main() -> None:
    st.set_page_config(page_title="Popcorn Picks", page_icon="🍿", layout="wide")
    st.title("🍿 Popcorn Picks")
    st.write("Find your next movie with ratings and genre preferences.")
    try:
        with st.spinner("Loading the movie catalog and recommendation model…"):
            model = cached_model(str(DEFAULT_MODEL_PATH))
            data = cached_data(str(DEFAULT_DATA_DIR))
    except (
        OSError,
        ValueError,
        ImportError,
        pickle.UnpicklingError,
        EOFError,
    ) as error:
        st.error(f"Could not load Popcorn Picks: {error}")
        st.stop()

    genres = sorted({genre for items in data.movies["genres"] for genre in items})
    with st.sidebar:
        st.header("Your preferences")
        with st.form("preferences"):
            user_id = st.number_input("MovieLens user ID", min_value=1, value=1, step=1)
            count = st.slider("Number of recommendations", 1, 20, 5)
            selected_genre = st.selectbox("Genre", ["All", *genres])
            minimum = st.slider("Minimum community rating", 0.0, 5.0, 0.0, 0.5)
            submitted = st.form_submit_button("Find movies")
        st.caption(
            "Movies you have rated are excluded. Users without rating history receive "
            "catalog predictions, with a genre boost when survey preferences exist."
        )

    personalized, catalog = st.tabs(["For you", "Browse movies"])
    with personalized:
        if submitted:
            with st.spinner("Finding movies for you…"):
                recommendations = recommend_movies(
                    int(user_id),
                    model,
                    data,
                    n=count,
                    genre=None if selected_genre == "All" else selected_genre,
                    min_average_rating=minimum,
                )
            render_recommendations(recommendations)
        else:
            st.info("Choose your preferences and select Find movies to get started.")
    with catalog:
        st.subheader("Most rated movies")
        popular = data.movies.sort_values("rating_count", ascending=False).head(25)
        display = popular[["title", "genres", "avg_rating", "rating_count"]].copy()
        display["genres"] = display["genres"].map(", ".join)
        display.columns = ["Movie", "Genres", "Community rating", "Ratings"]
        st.dataframe(display, hide_index=True)


if __name__ == "__main__":
    main()
