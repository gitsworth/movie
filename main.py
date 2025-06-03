import streamlit as st
import pandas as pd

# Load MovieLens 100K ratings data CSV from local or URL (you need to have these CSVs)
@st.cache_data
def load_data():
    # Ratings file with columns: userId, movieId, rating, timestamp
    ratings = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.data', sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])
    # Movie info file with columns: movieId|title|...
    movies = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.item', sep='|', encoding='latin-1', header=None, usecols=[0,1], names=['movieId', 'title'])
    return ratings, movies

ratings, movies = load_data()

# Convert IDs to string for consistency
ratings['movieId'] = ratings['movieId'].astype(str)
movies['movieId'] = movies['movieId'].astype(str)

st.title("Simple Movie Recommender")

user_id = st.number_input("Enter your user ID (1-943):", min_value=1, max_value=943, value=1)

if st.button("Get Recommendations"):
    user_id_str = str(user_id)

    # Movies the user has already rated
    user_movies = ratings[ratings['userId'] == user_id][['movieId']]

    # Calculate average rating for each movie
    avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()

    # Filter out movies the user already rated
    rec_movies = avg_ratings[~avg_ratings['movieId'].isin(user_movies['movieId'])]

    # Get top 10 movies by average rating
    top_movies = rec_movies.sort_values('rating', ascending=False).head(10)

    # Join with movie titles
    top_movies = top_movies.merge(movies, on='movieId')

    st.write(f"Top 10 movie recommendations for user {user_id} based on average ratings:")

    for idx, row in top_movies.iterrows():
        st.write(f"{row['title']} — Average Rating: {row['rating']:.2f}")
