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
knn_eval = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm='auto')
knn_eval.fit(movie_matrix)

accuracy = []
precision = []
recall = []
f1_score = []

for movie in range(0, movie_matrix.shape[0]):

    # Retrieve the distances and indices of k recommendations
    distances, indices = knn_eval.kneighbors(merged_df.iloc[movie, :].values.reshape(1, -1), n_neighbors=k)

    # print(distances)
    # print(indices)

    # Define a movie list to hold our recommendations
    movie_ls = []

    # Retrieve the k number of recommendations
    for i in range(0, len(distances.flatten())):

        # The movie does not equal the movie name we are evaluating
        if movie != indices.flatten()[i]:

            # Add our movie to the list of recommendations
            movie_ls.append(indices.flatten()[i])  # train_data.index[indices.flatten()[i]]

    # Calculate precision, recall, and f1-score
    # our recommendations versus WHAT?
    # print()
    # print("CHECK:")
    # print(merged_df.iloc[movie, :][merged_df.iloc[movie, :] >= 3]) # List of movies in matrix mor than rating of 3

    # The movie in the matrix has a rating >= 3 AND has been recommended
    # The intersection of actual (in the movie matrix) versus predicted (kNN)
    true_positives = len(set(movie_ls) & set(merged_df.iloc[movie, :][merged_df.iloc[movie, :] >= 3].index.values))

    # The movie in the matrix has a rating < 3 BUT has been recommended
    # The difference between predicted (kNN) and actual (in the movie matrix) when P=1 and A=0
    false_positives = len(set(movie_ls) & set(merged_df.iloc[movie, :][(merged_df.iloc[movie, :] < 3) & (merged_df.iloc[movie, :] > 0)].index.values))

    # The movie in the matrix has a rating >= 3 BUT has not been recommended
    # The difference between predicted (kNN) and actual (in the movie matrix) when P=0 and A=1
    false_negatives = len(set(merged_df.iloc[movie, :][merged_df.iloc[movie, :] >= 3].index.values)) - true_positives

    # The movie in the matrix has a rating < 3 AND has not been recommended
    # The intersection of actual (in the movie matrix) and predicted (kNN) when both are equal to 0.
    true_negatives = len(set(merged_df.iloc[movie, :][(merged_df.iloc[movie, :] < 3) & (merged_df.iloc[movie, :] > 0)].index.values)) - false_positives

    acc = true_positives / (true_positives + false_positives + false_negatives + true_negatives) if (true_positives + false_positives) > 0 else 0
    accuracy.append(acc)

    pre = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    precision.append(pre)

    rec = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    recall.append(rec)

    f1 = (2 * (precision[-1] * recall[-1]) / (precision[-1] + recall[-1])) if (precision[-1] + recall[-1]) > 0 else 0
    f1_score.append(f1)

mean_accuracy = np.mean(accuracy)
mean_precision = np.mean(precision)
mean_recall = np.mean(recall)
mean_f1_score = np.mean(f1_score)

print()
print("Mean Accuracy: ", mean_accuracy)
print("Mean Precision: ", mean_precision)
print("Mean Recall: ", mean_recall)
print("Mean F1 Score: ", mean_f1_score)
