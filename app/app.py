import streamlit as st
import pickle
import pandas as pd
from utils import hybrid_recommend  # Import from utils.py

# Cache data loading to avoid reloading on every interaction
@st.cache_data
def load_data():
    movies = pd.read_csv("datasets/ml-25m/movies.csv")
    ratings = pd.read_csv("datasets/ml-25m/ratings.csv")
    survey_data = pd.read_csv("datasets/survey.csv")
    
    # Preprocess genres into lists (if not already done in utils.py)
    movies['genres'] = movies['genres'].str.split('|')
    
    return movies, ratings, survey_data

# Cache model loading for performance
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

def main():
    st.title("🎬 Hybrid Movie Recommender")
    
    # Load data and model
    movies, ratings, survey_data = load_data()
    model = load_model()
    
    # User input
    user_id = st.number_input("Enter User ID", 
                            min_value=1,
                            max_value=ratings['userId'].max(),
                            value=1)
    
    if st.button("Get Recommendations"):
        with st.spinner('Generating recommendations...'):
            recommendations = hybrid_recommend(
                user_id=user_id,
                model=model,
                ratings_df=ratings,
                movies_df=movies,
                survey_df=survey_data,
                n=5
            )
        
        st.success("Top Recommendations:")
        for movie in recommendations:
            st.write(f"- {movie}")

if __name__ == "__main__":
    main()