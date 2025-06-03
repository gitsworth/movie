import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_data():
    ratings = pd.read_csv("https://files.grouplens.org/datasets/movielens/ml-100k/u.data", sep="\t",
                          names=["userId", "movieId", "rating", "timestamp"])
    movies = pd.read_csv("https://files.grouplens.org/datasets/movielens/ml-100k/u.item", sep="|",
                         encoding="latin-1", header=None, usecols=[0, 1], names=["movieId", "title"])
    return ratings, movies

ratings, movies = load_data()

# Create user-item rating matrix
user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

st.title("Collaborative Movie Recommender (User-Based)")

user_id = st.number_input("Enter your user ID (1-943):", min_value=1, max_value=943, value=1)

def get_recommendations(user_id, top_n=10):
    if user_id not in user_movie_matrix.index:
        return []

    # Similarity scores
    sim_scores = user_similarity_df[user_id].sort_values(ascending=False)
    sim_scores = sim_scores.drop(user_id)  # Drop self

    # Get top 10 similar users
    top_users = sim_scores.head(10).index

    # Ratings from similar users
    similar_users_ratings = user_movie_matrix.loc[top_users]

    # Weighted average of their ratings
    weighted_ratings = similar_users_ratings.T.dot(sim_scores.loc[top_users])
    similarity_sum = sim_scores.loc[top_users].sum()
    weighted_avg_ratings = weighted_ratings / similarity_sum

    # Remove movies the current user has already rated
    already_rated = user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index
    weighted_avg_ratings = weighted_avg_ratings.drop(already_rated, errors='ignore')

    # Get top N recommendations
    top_recs = weighted_avg_ratings.sort_values(ascending=False).head(top_n)
    recs = pd.DataFrame({'movieId': top_recs.index, 'score': top_recs.values})
    recs = recs.merge(movies, on='movieId')
    return recs

if st.button("Get Recommendations"):
    recs = get_recommendations(user_id)
    if recs.empty:
        st.write("No recommendations available.")
    else:
        st.write(f"Top recommendations for User {user_id}:")
        for i, row in recs.iterrows():
            st.write(f"{row['title']} — Predicted Score: {row['score']:.2f}")
