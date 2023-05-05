import pickle
import random

import numpy as np
import pandas as pd
import seaborn as sn
from statistics import mean
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
k = 10

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

col_arr = movie_matrix.tocoo().col
# print(col_arr)
row_arr = movie_matrix.tocoo().row
print(merged_df)

# i = 0
# for movie in range(0, movie_matrix.shape[0]):
for x in range(0, movie_matrix.shape[1]):
    # print(x)
    col_ind = col_arr[x]  # int(x.indices[0])  # column index
    curr_userId = int(merged_df.columns[col_ind-1])  # current userId
    # actual_rating = float(x.data[0])  # actual rating
    # row_ind = row_arr[i]  # int(x.indptr[0])  # row index
    # print()
    # print(x)
    # print(row_ind)
    # print()

    # curr_movie_title = merged_df.index.values[row_ind]  # current movie title
    # print(curr_movie_title)

    # if actual_rating > 0:
    # Retrieve the distances and indices of k recommendations
    # print(d.loc[:,'All'])
    # print(merged_df.iloc[:, col_ind].values.reshape(9719, -1))
    print()

    my_df = merged_df.copy()
    ans = []
    actual_rating_ls = []
    actual_rating = 0

    for idx, col in enumerate(my_df.columns):
        if x != idx:
            my_df[col].values[:] = 0
        else:
            my_df.loc[my_df[col] < 3, col] = 0
            ans.append(my_df.index[my_df[col_ind] > 0].tolist())
            actual_rating_ls = my_df[col].values
            actual_rating_ls = actual_rating_ls.tolist()
            actual_rating_ls = [i for i in actual_rating_ls if i != 0]
            actual_rating = mean(actual_rating_ls)

    print(my_df)

    distances, indices = knn_eval.kneighbors(my_df, n_neighbors=k)  # put the movie index
    # distances, indices = knn_eval.kneighbors(merged_df.iloc[col_ind, :].values.reshape(1, -1), n_neighbors=k)  # put the movie index

    # print(distances)
    # print(indices)

    # Define a movie list to hold our recommendations
    movie_ls = []

    # Retrieve the k number of recommendations
    for i in range(0, len(distances.flatten())):

        # The movie does not equal the movie name we are evaluating
        if indices.flatten()[i] not in ans:  # list of row values having more than 0 ratings:

            # Add our movie to the list of recommendations
            movie_ls.append(indices.flatten()[i])  # train_data.index[indices.flatten()[i]]

    pred_rating_list = []

    for movie_ind in movie_ls:

        # Average all actual ratings in the movie matrix and store it
        pred_rating_list.append(ratings_df[ratings_df['movieId'] == movie_ind]['rating'].mean())
        if len(pred_rating_list) == 0:
            pred_rating_list.append(0)


    # Drop all nans from pred list
    pred_rating_list = [x for x in pred_rating_list if str(x) != 'nan']
    print(pred_rating_list)
    # Calculate our final predicted rating
    predicted_rating = mean(pred_rating_list)
    print(predicted_rating)

    # Create lists to tally our evaluation metric results
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    if (actual_rating >= 3) & (predicted_rating >= 3):
        true_positives += 1
    elif (actual_rating < 3) & (predicted_rating >= 3):
        false_positives += 1
    elif (actual_rating >= 3) & (predicted_rating < 3):
        false_negatives += 1
    else:
        true_negatives += 1

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
