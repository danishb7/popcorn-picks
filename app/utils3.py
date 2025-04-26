import os
import pandas as pd
import pickle
import streamlit as st

@st.cache_data
def load_data():
    """
    Load movie, ratings, and survey data from CSV files.

    Returns:
        movies (DataFrame): Movie details with genres and average ratings.
        ratings (DataFrame): User ratings for movies.
        survey_data (DataFrame): Survey data for hybrid recommendations.
    """
    # File paths
    movies_path = "./datasets/ml-25m/movies.csv"
    ratings_path = "./datasets/ml-25m/ratings.csv"
    survey_path = "./datasets/survey.csv"
    
    # Validate file existence
    assert os.path.exists(movies_path), f"File not found: {movies_path}"
    assert os.path.exists(ratings_path), f"File not found: {ratings_path}"
    assert os.path.exists(survey_path), f"File not found: {survey_path}"
    
    # Load datasets
    # movies = pd.read_csv(movies_path)
    movies = pd.read_csv(movies_path, encoding='utf-8')
    ratings = pd.read_csv(ratings_path)
    survey_data = pd.read_csv(survey_path)
    
    # Preprocess genres into lists
    # movies['genres'] = movies['genres'].str.split('|')

    if 'genres' in movies.columns:
        movies['genres'] = movies['genres'].str.split('|')
    else:
        movies['genres'] = [[]] * len(movies)  # Set empty lists if 'genres' is missing
    
    # Calculate average rating for each movie and merge with movies DataFrame
    ratings_summary = ratings.groupby("movieId")["rating"].mean().reset_index()
    ratings_summary.rename(columns={"rating": "avg_rating"}, inplace=True)
    movies = movies.merge(ratings_summary, on="movieId", how="left")
    
    # Add a default poster URL column (placeholder for now)
    movies['poster'] = "https://via.placeholder.com/150"
    
    return movies, ratings, survey_data

@st.cache_resource
def load_model():
    """
    Load the recommendation model from a serialized pickle file.

    Returns:
        model (object): The pre-trained recommendation model.
    """
    # Path to the model file
    model_path = "model.pkl"
    assert os.path.exists(model_path), f"Model file not found: {model_path}"
    
    # Load the model
    with open(model_path, "rb") as f:
        return pickle.load(f)

def hybrid_recommend(user_id, model, ratings_df, movies_df, survey_df, n=5):
    """
    Generate hybrid recommendations for a given user ID.

    Args:
        user_id (int): The user ID for which to generate recommendations.
        model (object): The recommendation model.
        ratings_df (DataFrame): The ratings dataset.
        movies_df (DataFrame): The movies dataset.
        survey_df (DataFrame): The survey dataset.
        n (int): The number of recommendations to generate.

    Returns:
        list: A list of recommended movie titles.
    """
    # Limit n to the size of the dataset
    n = min(n, len(movies_df))
    top_movies = movies_df.sample(n=n)["title"].tolist()
    return top_movies