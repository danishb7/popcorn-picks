import streamlit as st
import pandas as pd
import pickle
from carousel import render_carousel  # Import the carousel function
from utils import hybrid_recommend  # Import the hybrid recommendation function

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

# Cache data loading to optimize performance
@st.cache_data
def load_data():
    movies = pd.read_csv("datasets/ml-25m/movies.csv")
    ratings = pd.read_csv("datasets/ml-25m/ratings.csv")
    survey_data = pd.read_csv("datasets/survey.csv")
    
    # Preprocess genres into lists
    movies['genres'] = movies['genres'].str.split('|')
    
    return movies, ratings, survey_data

# Cache the model loading
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

# Placeholder data for additional features
movies = [
    {"title": "Movie 1", "genre": "Action", "rating": 4, "image": "https://via.placeholder.com/150"},
    {"title": "Movie 2", "genre": "Drama", "rating": 5, "image": "https://via.placeholder.com/150"},
    {"title": "Movie 3", "genre": "Comedy", "rating": 3, "image": "https://via.placeholder.com/150"},
]

def render_movie_cards():
    # Card layout for movie recommendations
    st.markdown('<div style="display: flex; flex-wrap: wrap; justify-content: center;">', unsafe_allow_html=True)
    for movie in movies:
        st.markdown(
            f"""
            <div class="movie-card">
                <img src="{movie['image']}" alt="{movie['title']} Poster">
                <h4>{movie['title']}</h4>
                <p>Genre: {movie['genre']}</p>
                <p>Rating: {'⭐' * movie['rating']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

def render_genre_sections():
    # Thematic sections for genre-based recommendations
    st.subheader("🎭 Drama Recommendations")
    st.image("https://via.placeholder.com/200x300")
    st.write("Movie 1: Description or Rating")

    st.subheader("💥 Action Recommendations")
    st.image("https://via.placeholder.com/200x300")
    st.write("Movie 2: Description or Rating")

def render_filters():
    # Dynamic filters for genre, year, and rating
    genre = st.selectbox("Select Genre", ["All", "Action", "Drama", "Comedy", "Sci-Fi"])
    year = st.slider("Select Year", 2000, 2025, (2010, 2025))
    rating = st.slider("Select Minimum Rating", 1, 5, 3)
    st.write(f"Filtering by: Genre={genre}, Year={year}, Rating={rating}")

def main():
    st.title("🎬 Enhanced Movie Recommender App")
    add_custom_css()

    # Hybrid Recommendation Section (Move this section up)
    st.subheader("🔍 Personalized Recommendations")
    # Load data and model
    movies_df, ratings_df, survey_data = load_data()
    model = load_model()
    
    # User input for personalized recommendations
    user_id = st.number_input("Enter User ID", 
                              min_value=1, 
                              max_value=ratings_df['userId'].max(), 
                              value=1)

    if st.button("Get Recommendations"):
        with st.spinner('Generating recommendations...'):
            recommendations = hybrid_recommend(
                user_id=user_id,
                model=model,
                ratings_df=ratings_df,
                movies_df=movies_df,
                survey_df=survey_data,
                n=5
            )
        st.success("Top Recommendations:")
        for movie in recommendations:
            st.write(f"- {movie}")

    # Render filters
    render_filters()

    # Render carousel for recently released movies
    render_carousel()  # Carousel function called here

    # Render movie cards
    render_movie_cards()

    # Render genre sections
    render_genre_sections()

    # Interactive ratings
    st.subheader("⭐ Rate Your Favorite Movie")
    st.slider("Rate Movie 1", 1, 5, value=3)

if __name__ == "__main__":
    main()