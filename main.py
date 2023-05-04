import pickle
import random

import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, mean_squared_error
from scipy.sparse import csr_matrix

# Load the MovieLens 1M dataset
ratings_df = pd.read_csv("C:\\Users\\abedf\\Downloads\\ratings.csv")
movies_df = pd.read_csv("C:\\Users\\abedf\\Downloads\\movies.csv")

ratings_df.drop(columns=['timestamp'])

# Show the ratings dataframe
# print(ratings_df.head())

# Show the movies dataframe
# print(movies_df.head())

# Merge the ratings and movies dataframes
movie2 = movies_df.loc[:, ["movieId", "title"]]
rating = ratings_df.loc[:, ["userId", "movieId", "rating"]]

# pivot ratings into movie features
data = pd.merge(movie2, rating, on='movieId')
data = data.iloc[:1000000, :]

# How are we dealing with duplicate values on pivoting?
merged_df = data.pivot_table(
    index=['title'],  # previously movieId
    columns=['userId'],
    values='rating',
    aggfunc='mean'
).fillna(0)

# print(merged_df.head())

# convert dataframe of movie features to scipy sparse matrix
movie_matrix = csr_matrix(merged_df.values)

# print(movie_matrix)

# Set k as the number of similar movies to recommend
k = 20

# PASTE HERE

# Evaluations Metrics
# Do an 80-20 train-test split against the merged_df CHANGE TO SPLIT ON USERS
all_userIds = ratings_df['userId'].unique().tolist()

# Do an 80-20 on our matrix
movie_array = movie_matrix.toarray()

# print(movie_array)
# print("Movie Array Length: ", len(movie_array))

# Create a list of values from 1 to 610 (inclusive)
values = list(range(1, 611))

# Shuffle the values in random order
random.shuffle(values)

# Split the shuffled list into two lists, one containing 80% and another containing 20%
split_idx = int(len(values) * 0.8)
train_values = values[:split_idx]
test_values = values[split_idx:]

# Print the length of each list
# print(f"Length of train set: {len(train_values)}")
# print(f"Length of test set: {len(test_values)}")

# Split the merged_df dataframe into train and test dataframes
train_df = merged_df.loc[:, merged_df.columns.isin(train_values)]
test_df = merged_df.loc[:, merged_df.columns.isin(test_values)]

# Convert the train and test arrays back to csr matrices
train_matrix = csr_matrix(train_df)
test_matrix = csr_matrix(test_df)

# print(train_matrix.shape[0])  # should be 488
# print(test_matrix.shape[0])  # should be 122
#
# print(train_matrix.shape[1])  # should be 488
# print(test_matrix.shape[1])  # should be 122

# Create our kNN model and fit it to our movie matrix
knn_eval = NearestNeighbors(n_neighbors=k, leaf_size=100, metric="euclidean", algorithm='auto')
knn_eval.fit(test_matrix)

precision = []
recall = []
f1_score = []

for movie in range(0, test_matrix.shape[0]):

    # Retrieve the distances and indices of k recommendations
    distances, indices = knn_eval.kneighbors(test_df.iloc[movie, :].values.reshape(1, -1), n_neighbors=k)

    # print(distances)
    # print(indices)

    # Define a movie list to hold our recommendations
    movie_ls = []

    # Iterate through our k recommendations
    for i in range(0, len(distances.flatten())):

        # If we are not on the first distance (skip over the first which is the movie itself)
        if i != 0:
            # Add our movie to the list of recommendations
            # print(train_data.index[indices.flatten()[i]])

            movie_ls.append(indices.flatten()[i])  # train_data.index[indices.flatten()[i]]

    # Calculate precision, recall, and f1-score
    true_positives = len(set(movie_ls) & set(test_df.iloc[movie, :][test_df.iloc[movie, :] != 0].index.values))
    print("movie list: ")
    print(set(movie_ls))  # returns titles
    print()
    print("TEST VALS: ")
    print(set(test_df.iloc[movie, :][test_df.iloc[movie, :] != 0].index.values))  # returns test matrix indices (A to Z titles)
    false_positives = len(movie_ls) - true_positives
    # print(false_positives)
    false_negatives = len(set(test_df.iloc[movie, :][test_df.iloc[movie, :] != 0].index.values)) - true_positives
    # print(false_negatives)
    true_negatives = 0

    precision.append(true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else 0
    recall.append(true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0
    f1_score.append(2 * (precision[-1] * recall[-1]) / (precision[-1] + recall[-1])) if (precision[-1] + recall[-1]) > 0 else 0

mean_precision = np.mean(precision)
mean_recall = np.mean(recall)
mean_f1_score = np.mean(f1_score)

print()
print("Mean Precision: ", mean_precision)
print("Mean Recall: ", mean_recall)
print("Mean F1 Score: ", mean_f1_score)
