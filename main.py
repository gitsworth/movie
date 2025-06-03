import streamlit as st
from surprise import Dataset, SVD
from surprise.model_selection import train_test_split
import pandas as pd
import os

# Load movie titles
def load_movie_titles():
    # Path to the u.item file from MovieLens 100k dataset
    # This file should be in the dataset folder, so we extract its location via surprise dataset folder
    data_folder = Dataset.load_builtin('ml-100k').raw_ratings_file().rsplit('/', 1)[0]
    item_file = os.path.join(data_folder, 'u.item')

    # u.item format: movie id | movie title | ... (| separated)
    movies = pd.read_csv(item_file, sep='|', header=None, encoding='latin-1', usecols=[0,1], names=['movie_id', 'title'])
    # Convert movie_id to string to match surprise IDs
    movies['movie_id'] = movies['movie_id'].astype(str)
    return movies.set_index('movie_id')['title'].to_dict()

movie_titles = load_movie_titles()

# Load the MovieLens 100K dataset
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=0.25)

# Train the SVD model
algo = SVD()
algo.fit(trainset)

# Function to get top N recommendations for a user
def get_top_n(user_id, n=10):
    try:
        inner_uid = trainset.to_inner_uid(str(user_id))
    except ValueError:
        st.error(f"User ID {user_id} not found in the dataset.")
        return []

    rated_movies = [trainset.to_raw_iid(iid) for (iid, _) in trainset.ur[inner_uid]]

    all_inner_ids = trainset.all_items()
    predictions = []
    for inner_iid in all_inner_ids:
        raw_iid = trainset.to_raw_iid(inner_iid)
        if raw_iid not in rated_movies:
            pred = algo.predict(str(user_id), raw_iid)
            predictions.append(pred)

    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n = predictions[:n]

    # Return movie titles and estimated ratings
    return [(movie_titles.get(pred.iid, pred.iid), pred.est) for pred in top_n]

# Streamlit UI
st.title("Movie Recommender System with Titles")
user_id = st.number_input("Enter your user ID (1-943):", min_value=1, max_value=943, value=1)

if st.button("Get Recommendations"):
    recommendations = get_top_n(user_id)
    if recommendations:
        st.write("Top 10 movie recommendations for you:")
        for title, rating in recommendations:
            st.write(f"{title} — Estimated Rating: {rating:.2f}")
