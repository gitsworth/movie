import streamlit as st
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pandas as pd

# Load the MovieLens 100K dataset
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.25)

# Train the SVD model
algo = SVD()
algo.fit(trainset)

# Function to get top N recommendations for a user
def get_top_n(user_id, n=10):
    # Get a list of all movie ids
    all_movie_ids = trainset.all_items()
    # Get the list of movies the user has already rated
    rated_movies = [trainset.to_raw_iid(i) for i in trainset.ur[user_id]]
    # Predict ratings for all movies not rated by the user
    predictions = [algo.predict(user_id, movie_id) for movie_id in all_movie_ids if movie_id not in rated_movies]
    # Sort the predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)
    # Get the top N recommendations
    top_n = predictions[:n]
    return [(pred.iid, pred.est) for pred in top_n]

# Streamlit UI
st.title("Movie Recommender System")
user_id = st.number_input("Enter your user ID (1-943):", min_value=1, max_value=943, value=1)
if st.button("Get Recommendations"):
    recommendations = get_top_n(user_id)
    st.write("Top 10 movie recommendations for you:")
    for movie_id, rating in recommendations:
        st.write(f"Movie ID: {movie_id}, Estimated Rating: {rating:.2f}")
