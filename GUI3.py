import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

# Assuming you have loaded your data and created the user_item_matrix
# For this example, I'm assuming user_item_matrix, movies DataFrame, and cf_knn_model are already defined

# Sample data (you should replace this with your actual data)
# movies = pd.read_csv('path_to_movies.csv')
# user_item_matrix = pd.read_csv('path_to_user_item_matrix.csv')

# Function to find the closest movie title
def find_closest_title(input_title, movie_titles):
    closest_title, _ = process.extractOne(input_title, movie_titles)
    return closest_title

# Define the KNN model for collaborative filtering
cf_knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
cf_knn_model.fit(user_item_matrix)

def movie_recommender_engine(movie_name, matrix, cf_model, n_recs):
    # Find the closest matching movie title
    closest_title = find_closest_title(movie_name, movies['title'].tolist())
    if not closest_title:
        messagebox.showinfo("No Match", f"No matching movie found for '{movie_name}'.")
        return pd.DataFrame()

    # Find the movie ID of the closest matching movie
    movie_id = movies[movies['title'] == closest_title]['movieId'].values
    if len(movie_id) == 0:
        messagebox.showinfo("Error", f"Movie ID for '{closest_title}' not found in the dataset.")
        return pd.DataFrame()

    movie_id = movie_id[0]

    # Ensure the movie ID is in the matrix columns
    if movie_id not in matrix.columns:
        messagebox.showinfo("Error", f"Movie ID {movie_id} not found in the user-item matrix.")
        return pd.DataFrame()

    # Create a query vector for the movie
    query_vector = np.zeros(matrix.shape[1])
    if movie_id in matrix.columns:
        query_vector[matrix.columns.get_loc(movie_id)] = 1

    # Calculate neighbor distances
    distances, indices = cf_model.kneighbors(query_vector.reshape(1, -1), n_neighbors=n_recs)

    # List to store recommendations
    cf_recs = []
    for i in indices.squeeze().tolist():
        if i < len(matrix.columns):
            movie_id_rec = matrix.columns[i]
            rec_title = movies[movies['movieId'] == movie_id_rec]['title'].values[0]
            cf_recs.append({'Title': rec_title, 'Distance': distances.squeeze().tolist()[indices.squeeze().tolist().index(i)]})

    # Return a DataFrame of recommendations
    df = pd.DataFrame(cf_recs, index=range(1, len(cf_recs) + 1))

    return df

# GUI Setup
def get_recommendations():
    movie_name = movie_entry.get()
    if not movie_name:
        messagebox.showerror("Input Error", "Please enter a movie name.")
        return
    
    cf_recommendations = movie_recommender_engine(movie_name, user_item_matrix, cf_knn_model, 10)
    if not cf_recommendations.empty:
        result_text.set(cf_recommendations.to_string(index=False))

# Initialize the Tkinter GUI
root = tk.Tk()
root.title("Movie Recommendation System")

mainframe = ttk.Frame(root, padding="10")
mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

# Movie entry
movie_label = ttk.Label(mainframe, text="Enter Movie Name:")
movie_label.grid(column=0, row=0, sticky=tk.W)
movie_entry = ttk.Entry(mainframe, width=40)
movie_entry.grid(column=1, row=0, sticky=(tk.W, tk.E))

# Recommend button
recommend_button = ttk.Button(mainframe, text="Get Recommendations", command=get_recommendations)
recommend_button.grid(column=2, row=0)

# Result text area
result_text = tk.StringVar()
result_label = ttk.Label(mainframe, textvariable=result_text)
result_label.grid(column=0, row=1, columnspan=3, sticky=tk.W)

root.mainloop()
