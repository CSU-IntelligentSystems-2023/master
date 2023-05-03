import pandas as pd

class TopWeighted():
    def top_weighted_movies(genre=None):

        movies_df = pd.read_csv('C:\\Users\\aidah\\OneDrive\\Desktop\\CSU\\Intelligent Systems\\ml-latest-small\\ml-latest-small\\movies.csv')
        ratings_df = pd.read_csv('C:\\Users\\aidah\\OneDrive\\Desktop\\CSU\\Intelligent Systems\\ml-latest-small\\ml-latest-small\\ratings.csv')
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

        # sort the movies based on their Weighted Ratings
        top_movies = movies_df.sort_values('weighted_rating', ascending=False)

        # dropping the release date from the titles
        list = top_movies['title'].head(10).tolist()
        titles_list = []
        for item in list:
            title,sep, year = item.partition(' (')
            titles_list.append(title)

        # printing and returning the top 10 movies based on their Weighted Ratings
        print("Top 10 movies based on Weighted Ratings:")
        print (titles_list)
        return titles_list
