import numpy as np
from surprise import Prediction

def hybrid_recommend(user_id, model, ratings_df, movies_df, survey_df, n=5):
    """
    Args:
        user_id: Target user ID
        model: Trained Surprise model (SVD)
        ratings_df: DataFrame of ratings (columns: userId, movieId, rating)
        movies_df: DataFrame of movies (columns: movieId, title, genres)
        survey_df: Survey data (columns: userId, action_rating, comedy_rating, etc.)
        n: Number of recommendations
    """
    # Get all unique movie IDs
    all_movies = ratings_df['movieId'].unique()
    
    # Step 1: Collaborative Filtering predictions
    cf_predictions = [model.predict(user_id, mid).est for mid in all_movies]
    
    # Step 2: Content-based boosting (if user exists in survey)
    if user_id in survey_df['userId'].values:
        user_row = survey_df[survey_df['userId'] == user_id].iloc[0]
        genre_weights = {
            'Action': user_row['action_rating'],
            'Comedy': user_row['comedy_rating'],
            'Sci-Fi': user_row['sci_fi_rating']
        }
        # Score movies by genre alignment (assuming movies_df['genres'] is pre-split into lists)
        genre_scores = movies_df['genres'].apply(
            lambda g: sum(genre_weights.get(genre, 0) for genre in g)
        )
    else:
        genre_scores = 0  # Fallback to pure CF
    
    # Step 3: Combine scores (60% CF + 40% genre)
    combined_scores = 0.6 * np.array(cf_predictions) + 0.4 * genre_scores
    top_indices = np.argsort(combined_scores)[-n:][::-1]
    
    return movies_df.iloc[top_indices]['title'].tolist()