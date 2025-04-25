import streamlit as st
import pandas as pd
from utils3 import load_data, load_model, hybrid_recommend
from carousel import render_carousel  # Import the carousel function

# Custom CSS for styling
def add_custom_css():
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(to right, #ff7e5f, #feb47b);
            color: #fff;
        }
        .movie-card {
            margin: 10px;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 10px;
            text-align: center;
            transition: transform 0.2s ease-in-out;
        }
        .movie-card img {
            width: 100%;
            border-radius: 5px;
        }
        .movie-card:hover {
            transform: scale(1.05);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Filter movies based on dropdown and sliders
def render_filters_and_apply(movies_df):
    st.subheader("🔍 Filter Movies")
    
    # Filter widgets
    genre = st.selectbox("Select Genre", ["All"] + sorted(set(genre for genres in movies_df["genres"] for genre in genres)))
    year_range = st.slider("Select Year Range", int(movies_df["release_year"].min()), int(movies_df["release_year"].max()), (2010, 2025))
    min_rating = st.slider("Select Minimum Rating", 0.0, 5.0, 3.0)

    # Apply filters
    filtered_movies = movies_df.copy()
    if genre != "All":
        filtered_movies = filtered_movies[filtered_movies["genres"].apply(lambda x: genre in x)]
    filtered_movies = filtered_movies[(filtered_movies["release_year"] >= year_range[0]) & (filtered_movies["release_year"] <= year_range[1])]
    filtered_movies = filtered_movies[filtered_movies["avg_rating"] >= min_rating]

    return filtered_movies

# Render movie cards dynamically based on the filtered dataset
def render_movie_cards(movies_df):
    st.subheader("🎬 Movie Recommendations")
    st.markdown('<div style="display: flex; flex-wrap: wrap; justify-content: center;">', unsafe_allow_html=True)
    
    default_poster = "https://via.placeholder.com/150"
    for _, movie in movies_df.iterrows():
        poster_url = movie['poster'] if pd.notna(movie['poster']) else default_poster
        st.markdown(
            f"""
            <div class="movie-card">
                <img src="{poster_url}" alt="{movie['title']} Poster">
                <h4>{movie['title']}</h4>
                <p>Genre: {", ".join(movie['genres'])}</p>
                <p>Year: {movie['release_year']}</p>
                <p>Rating: {'⭐' * int(movie['avg_rating'])}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.title("🎬 Enhanced Movie Recommender App")
    add_custom_css()

    # Load data and model
    movies_df, ratings_df, survey_data = load_data()
    model = load_model()

    # Render filters and apply them to the movies dataset
    filtered_movies = render_filters_and_apply(movies_df)

    # Render filtered movie cards
    render_movie_cards(filtered_movies)

    # Render carousel for recently released movies
    render_carousel()

    # Hybrid Recommendation Section
    st.subheader("🔍 Personalized Recommendations")
    default_user_id = 1  # Set a fixed user ID for recommendations
    st.write(f"Recommendations for User ID: {default_user_id}")

    with st.spinner('Generating recommendations...'):
        recommendations = hybrid_recommend(
            user_id=default_user_id,
            model=model,
            ratings_df=ratings_df,
            movies_df=movies_df,
            survey_df=survey_data,
            n=5
        )
    st.success("Top Recommendations:")
    for movie in recommendations:
        st.write(f"- {movie}")

if __name__ == "__main__":
    main()