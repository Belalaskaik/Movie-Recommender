from flask import Flask, jsonify, request, render_template
import os
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack  
from sklearn.preprocessing import MultiLabelBinarizer
from fuzzywuzzy import process
import pandas as pd
import random
from joblib import load


app = Flask(__name__)

# Loading datasets
movies = pd.read_csv('./movies.csv')
ratings = pd.read_csv('./ratings.csv')
tags = pd.read_csv('./tags.csv')

# Setting 'movieId' as the index for the movies DataFrame
movies.set_index('movieId', inplace=True)

# Preprocess movies data
movies['genres'] = movies['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])

# # Binarizing genres
# mlb = MultiLabelBinarizer(sparse_output=True)  # Enable sparse_output to reduce memory usage
# genres_binarized = mlb.fit_transform(movies['genres'])

# # Preprocess tags data
# tags_grouped = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(map(str, x))).reset_index()
# vectorizer = CountVectorizer()
# tags_binarized = vectorizer.fit_transform(tags_grouped['tag'])

# # Ensure tags_df has the same indices as movies
# tags_df = pd.DataFrame(tags_binarized.toarray(), columns=vectorizer.get_feature_names_out(), index=tags_grouped['movieId'])
# tags_df = tags_df.reindex(movies.index, fill_value=0)

# # Combine genres and tags data into combined_features
# combined_features = hstack([genres_binarized, csr_matrix(tags_df)])

# # Nearest Neighbors Model - updated to use combined features
# model = NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
# model.fit(combined_features)

# # Save the model to a file
# dump(model, 'nearest_neighbors_model.joblib')
model = load('./nearest_neighbors_model.joblib')


@app.route('/')
def index():
    return render_template('movieRecommender.html')

@app.route('/recommend', methods=['GET'])
def recommend():
    movie_title = request.args.get('title', '')
    recommended_movies = recommend_movies(movie_title)
    return jsonify(recommended_movies)

@app.route('/fan-favorites', methods=['GET'])
def fan_favorites():
    favorites = recommend_fan_favorites()
    return jsonify(favorites)

if __name__ == '__main__':
    app.run(debug=True)






# Function to find closest movie title
def find_closest_title(title):
    all_titles = movies['title'].tolist()
    closest_title = process.extractOne(title, all_titles)[0]
    return closest_title

# Function to recommend movies based on favorite
def recommend_movies(input_title):
    movie_title = find_closest_title(input_title)
    movie_idx = movies.index[movies['title'] == movie_title].tolist()
    
    if not movie_idx:
        return "Movie not found."
    movie_idx = movie_idx[0]
    chosen_movie = {
        "title": movies.iloc[movie_idx]['title'],
        "genre": ", ".join(movies.iloc[movie_idx]['genres'])
    }

    # Ensure the input for the model has the same feature names as during fitting
    input_features = genres_df.iloc[[movie_idx]]  # Make it a dataframe with the same structure
    distances, indices = model.kneighbors(input_features)
    recommended_movie_indices = indices[0][1:]  # Exclude the input movie itself

    recommended_movies = []
    for idx in recommended_movie_indices:
        movie_info = movies.iloc[idx]
        recommended_movies.append({
            "title": movie_info['title'],
            "genre": ", ".join(movie_info['genres'])
        })

    return {"chosenMovie": chosen_movie, "recommendations": recommended_movies}


# Function to recommend random fan favorites (5-star ratings)
def recommend_fan_favorites():
    high_rated = ratings[ratings['rating'] == 5.0]
    top_movies = high_rated['MovieId'].unique()
    random_top_movies = random.sample(list(top_movies), 5)
    fan_favorites = []
    for movie_id in random_top_movies:
        movie_info = movies.loc[movie_id]
        fan_favorites.append({
            "title": movie_info['title'],
            "genre": movie_info['genres']
        })
    return fan_favorites


# Example usage
user_favorite_movie = 'Toy Story'  # User input
recommended_movies = recommend_movies(user_favorite_movie)
print("Movies recommended based on", user_favorite_movie, ":\n", recommended_movies)

# Recommend random fan favorites
fan_favorites = recommend_fan_favorites()
print("Random fan favorite movies:\n", fan_favorites)
