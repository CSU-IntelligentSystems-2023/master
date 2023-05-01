import pickle
import random
#adding some comment
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load the MovieLens 1M dataset
ratings_df = pd.read_csv("C:\\Users\\aidah\\OneDrive\\Desktop\\CSU\\Intelligent Systems\\movieLens25mArchive\\ml-25m\\ratings.csv")
movies_df = pd.read_csv("C:\\Users\\aidah\\OneDrive\\Desktop\\CSU\\Intelligent Systems\\movieLens25mArchive\\ml-25m\\movies.csv")

# Show the ratings dataframe
print(ratings_df.head())

# Show the movies dataframe
print(movies_df.head())

# Split the data into training and test sets
all_userIds = ratings_df['userId'].unique().tolist()

train_users = random.sample(all_userIds, 130033)

X = ratings_df.drop(columns=['rating', 'timestamp'])
X_train = X[X['userId'].isin(train_users)]
X_test = X[~X['userId'].isin(train_users)]

print("X_train: ")
print(X_train.head())

y = ratings_df.drop(columns=['movieId', 'timestamp'])
y_train = y[y['userId'].isin(train_users)]
y_test = y[~y['userId'].isin(train_users)]

y_train = y_train.drop(columns=['userId'])
y_test = y_test.drop(columns=['userId'])

print("y_train: ")
print(y_train.head())

# Set k as the number of similar movies to recommend
k = 10

# EITHER LOAD AN EXISTING MODEL OR CREATE A NEW ONE

# Load the model from disk
# Algorithm:  Ball Tree
# Metric: Euclidean
knn = pickle.load(open('C:\\Users\\aidah\\OneDrive\\Desktop\\CSU\\Intelligent Systems\\knnpickle_file_bt_eu', 'rb'))

# Create a new model
# Fit a kNN model using the training set
# knn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm='ball_tree')

# knn.fit(X_train.values, y_train)

# Create a new file to store the model in
# knnPickle = open('knnpickle_file_bt_eu', 'wb')

# Specify the source and destination
# pickle.dump(knn, knnPickle)

# Close the file
# knnPickle.close()

# Create our Confusion Matrix
# Create empty confusion matrix
conf_matrix = [[0, 0], [0, 0]]



# Loop over each user in the test set
for user_id in X_test['userId'].unique():
    print(user_id)
    all_user_rows = ratings_df[ratings_df['userId'] == user_id]
    user_ratings = all_user_rows['rating'].values
    print(user_ratings)
    rated_movies = all_user_rows['movieId']
    print(rated_movies)
    user_id_list_copy = [user_id] * len(rated_movies)
    X_test_user_df = pd.DataFrame({
        'userId': user_id_list_copy,
        'movieId': rated_movies})
    print(X_test_user_df)

    # Get the k most similar movies to the user's rated movies
    distances, indices = knn.kneighbors(X_test_user_df.values, return_distance=True)
    similar_movies = [ratings_df.loc[idx, 'movieId'] for idx in indices[0] if idx not in rated_movies][:k]
    print(similar_movies)

    # Find the actual ratings of the recommended movies
    # actual_ratings = []
    for movie_id in similar_movies:
        print(movie_id)
        sm_actual_ratings_df = ratings_df.loc[ratings_df['movieId'] == movie_id, ['rating']].copy()
        print(sm_actual_ratings_df.head())
        sm_actual_ratings_df['actual_ratings'] = np.where(
            (pd.isna(sm_actual_ratings_df['rating']) is False) &
            (sm_actual_ratings_df['rating'] >= 3), 1, 0)
        # if not pd.isna(actual_rating):
        #     actual_ratings.append(1 if actual_rating >= 3 else 0) # rating of 3 or higher
        actual_ratings = sm_actual_ratings_df['actual_ratings'].values.tolist()

        # UPDATE THE CONFUSION MATRIX USING THE THRESHOLD APPROACH

    # The approach below is known as a "threshold-based" approach
    # to making predictions or recommendations. It involves setting
    # a fixed threshold and using it to make binary decisions.
    # This approach has been used in various fields such as classification,
    # recommendation systems, and medical testing. It is simple and easy to
    # implement, but it may not always be the most accurate or optimal approach,
    # particularly when the distribution of data is imbalanced or the
    # threshold needs to be adjusted for different scenarios.
    # Source: Xiaoyuan Su and Taghi M. Khoshgoftaar. "A survey of collaborative
    # filtering techniques." Advances in Artificial Intelligence 2009 (2009): 4.
    # PDF Website: https://downloads.hindawi.com/archive/2009/421425.pdf

    # The first line below checks if there are any
    # actual ratings provided by the user for a particular movie.
    # If there are no ratings provided, then the
    # code skips updating the confusion matrix:
    if len(actual_ratings) > 0:
        # The next line calculates the predicted rating
        # based on the actual ratings. It first sums up all the
        # actual ratings for the movie and then compares it to
        # half the number of actual ratings. If the sum of actual ratings
        # is greater than or equal to half the number of actual ratings,
        # then the predicted rating is set to 1, otherwise, it is set to 0:
        predicted_rating = 1 if sum(actual_ratings) >= len(actual_ratings) / 2 else 0
        # The third line of code sets the actual rating as 1 if at
        # least one actual rating is 1, otherwise, it is set to 0:
        actual_rating = 1 if sum(actual_ratings) > 0 else 0
        # The last line of code updates the confusion matrix by
        # incrementing the count in the corresponding cell based on the actual
        # and predicted ratings. The conf_matrix variable is a 2x2 matrix that
        # keeps track of the number of true positives, false positives,
        # true negatives, and false negatives:
        conf_matrix[actual_rating][predicted_rating] += 1

# Calculate accuracy and precision from the confusion matrix
tp = conf_matrix[1][1]  # True Positive
tn = conf_matrix[0][0]  # True Negative
fp = conf_matrix[0][1]  # False Positive
fn = conf_matrix[1][0]  # False Negative

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)

print("Confusion matrix: ", conf_matrix)
print("Accuracy: ", accuracy)
print("Precision: ", precision)

# Plot our confusion matrix
# conf_matrix.plot()
# plt.show()
