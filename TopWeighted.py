import random
import pandas as pd

class TopWeighted():
    
    def top_weighted_movies(genre=None, blacklist=None ):


        movies_df = pd.read_csv('.\\movies.csv')
        ratings_df = pd.read_csv('.\\ratings.csv')
        # IMDB weighted avg formula:
        # Weighted Rating(WR)=[vR/(v+m)]+[mC/(v+m)]

        # v is the number of votes for the movie
        vote_count = ratings_df.groupby('movieId')['rating'].count().reset_index()
        vote_count.columns = ['movieId', 'vote_count']

        # m is the minimum votes required to be listed in the chart
        m = 10 #hardcoded value

        # C is the mean vote across the whole report
        C = ratings_df['rating'].mean()

        # R is the average rating of the movie
        avg_rating = ratings_df.groupby('movieId')['rating'].mean().reset_index()
        avg_rating.columns = ['movieId', 'avg_rating']

        # merging the vote count and average rating dataframes with the movies dataframe
        movies_df = pd.merge(movies_df, vote_count, on='movieId')
        movies_df = pd.merge(movies_df, avg_rating, on='movieId')
        
        # calculating the Weighted Rating for each movie
        movies_df['weighted_rating'] = ((movies_df['vote_count'] * movies_df['avg_rating']) / (movies_df['vote_count'] + m)) + ((m * C) / (movies_df['vote_count'] + m))

        # filtering the movies by genre if it is provided:
        if genre:
            movies_df = movies_df[movies_df['genres'].str.contains(genre)]

        # excluding blacklisted movies, which are the titles with user rating of 1 or 2
        if blacklist:
            movies_df = movies_df[~movies_df['title'].isin(blacklist)]  

        # shuffling the movies dataframe and select the top 10 movies
        movies_df = movies_df.sample(frac=1)  # shuffling the dataframe
        top_movies = movies_df.sort_values('weighted_rating', ascending=False).head(10)

        # Randomly select a subset of the top movies
        num_movies = min(10, len(top_movies))  # Select at most 5 movies
        random_movies = random.sample(top_movies['title'].tolist(), num_movies)
        #adding titles into a list to pass to main app
        titles_list = []
        for item in random_movies:
            title,sep, year = item.partition(' (')
            titles_list.append(title)
        print("Top 10 movies based on Weighted Ratings:")
        print (titles_list)
        return titles_list
    