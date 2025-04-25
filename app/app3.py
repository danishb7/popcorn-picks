import streamlit as st
import pandas as pd
from utils3 import load_data, load_model, hybrid_recommend
from carousel import render_carousel  # Import the carousel function

# Custom CSS for styling (minimal for list view)
def add_custom_css():
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(to right, #ff7e5f, #feb47b);
            color: #fff;
        }
        .movie-card {
            margin-bottom: 15px;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 8px;
            background: #2c2f33;
            color: #fff;
            text-align: left;
        }
        .movie-card img {
            float: left;
            margin-right: 15px;
            width: 80px;
            height: 120px;
            border-radius: 5px;
        }
        .movie-card h4 {
            margin: 0;
            font-size: 18px;
        }
        .movie-card p {
            margin: 5px 0;
            font-size: 14px;
            color: #aaa;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_recommendations_list(recommendations):
    """
    Render recommendations in a vertical list view.
    """
    st.subheader("🎥 Recommendations for you!")

    default_poster = "https://via.placeholder.com/80x120"
    for movie in recommendations:
        st.markdown(
            f"""
            <div class="movie-card">
                <img src="{default_poster}" alt="{movie} Poster">
                <div>
                    <h4>{movie}</h4>
                    <p>⭐ Rating: 4.5</p>
                    <p>Genre: Action, Adventure</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_filters_and_apply(movies_df):
    st.subheader("🔍 Filter Movies")
    
    # Filter widgets
    genre = st.selectbox("Select Genre", ["All"] + sorted(set(genre for genres in movies_df["genres"] for genre in genres)))
    min_rating = st.slider("Select Minimum Rating", 0.0, 5.0, 3.0)

    # Apply filters
    filtered_movies = movies_df.copy()
    if genre != "All":
        filtered_movies = filtered_movies[movies_df["genres"].apply(lambda x: genre in x)]
    filtered_movies = filtered_movies[movies_df["avg_rating"] >= min_rating]

    return filtered_movies

def render_movie_cards(movies_df, limit=10):
    st.subheader("🎬 Movie Recommendations")
    st.markdown('<div style="display: flex; flex-wrap: wrap; justify-content: center;">', unsafe_allow_html=True)
    
    default_poster = "https://via.placeholder.com/150"
    for _, movie in movies_df.head(limit).iterrows():
        poster_url = movie['poster'] if pd.notna(movie['poster']) else default_poster
        st.markdown(
            f"""
            <div class="movie-card">
                <img src="{poster_url}" alt="{movie['title']} Poster">
                <h4>{movie['title']}</h4>
                <p>Genre: {", ".join(movie['genres'])}</p>
                <p>Rating: {'⭐' * int(movie['avg_rating'])}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.title("🎬 DB's Popcorn Picks!")
    add_custom_css()

    # Load data and model
    movies_df, ratings_df, survey_data = load_data()
    model = load_model()

    # --- Render Personalized Recommendations ---
    default_user_id = 1  # Set a fixed user ID for recommendations
    with st.spinner('Generating recommendations...'):
        recommendations = hybrid_recommend(
            user_id=default_user_id,
            model=model,
            ratings_df=ratings_df,
            movies_df=movies_df,
            survey_df=survey_data,
            n=5
        )
    render_recommendations_list(recommendations)
    # -----------------------------------------------------

    # Render filters and apply them to the movies dataset
    filtered_movies = render_filters_and_apply(movies_df)

    # Render filtered movie cards
    render_movie_cards(filtered_movies, limit=10)

    # Render carousel for recently released movies
    render_carousel()

if __name__ == "__main__":
    main()