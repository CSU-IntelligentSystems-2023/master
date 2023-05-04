import tkinter as tk
import os
import pandas as pd
import pickle
import random 
import pandas as pd

from tkinter import filedialog
from tkinter import *
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from TopWeighted import *


#paths -- grabbed latest small from: https://grouplens.org/datasets/movielens/

movies_filename = '.\\movies.csv'
ratings_filename = '.\\ratings.csv'

#read the data
df_movies = pd.read_csv(movies_filename,
    usecols = ['movieId', 'title'],
    dtype = {'movieId': 'int32', 'title': 'str'})

df_ratings = pd.read_csv(ratings_filename,
    usecols = ['userId', 'movieId', 'rating'],      
    dtype = {'userId' : 'int32', 'movieId': 'int32', 'rating':'float32'})

df_genres = pd.read_csv(movies_filename,
    usecols = ['movieId','genres'],
    dtype = {'movieId': 'int32', 'genres': 'str'})

rated_df = pd.DataFrame({'userId':  [162543],'movieId': [0]})

coldstart_list = TopWeighted.top_weighted_movies()

recommendations = [["", 0], ["", 0], ["", 0], 
                ["", 0], ["", 0], ["", 0], 
                ["", 0], ["", 0], ["", 0],
                ["", 0],]

recommendations = list(recommendations)

#distance dictionaries
distance_dict = {
  1: "euclidean",
  2: "minkowski",
  3: "cosine",
  4: "manhattan",
  5: "suggest title based on popuarity"
}

#set up the ratings as the features for the movies 
#df_pivoted = pd.merge(df_movies, df_genres, on='movieId')
df_pivoted = df_ratings.pivot(index = 'movieId', columns = 'userId', values = 'rating')

df_pivoted.fillna(0, inplace=True)

# Set k as the number of similar movies to recommend (knn still returns 10 for some reason)
k = 25
# Create a new model
# Set up variousu kNN model and fit our pivoted df -- euclidean is default
knn = knn_euc = NearestNeighbors(n_neighbors=k, leaf_size = 100, metric="euclidean", algorithm='auto')
knn_cos = NearestNeighbors(n_neighbors=k, leaf_size = 100, metric="cosine", algorithm='auto')
knn_mink = NearestNeighbors(n_neighbors=k, leaf_size = 100, metric="minkowski", algorithm='auto')
knn_manhattan = NearestNeighbors(n_neighbors=k, leaf_size = 100, metric="manhattan", algorithm='auto')

knn.fit(df_pivoted)

# Dictionary of the genre of the the movies user has picked 
genreDict = {"Action" : 0, "Adventure" : 0, "Animation":0, "Children's":0, "Comedy":0, "Crime":0, "Documentary":0, "Drama":0, "Fantasy":0, "Film-Noir":0, "Horror":0, "Musical":0, "Mystery":0, "Romance":0, "Sci-Fi":0, "Thriller":0, "War":0, "Western":0, "(no genres listed)":0, "IMAX":0}

#blacklist to remove user selected titles
blacklist = []

# Create a new file to store the model in
#knnPickle = open('knnpickle_file_bt_eu', 'wb')

#Specify the source and destination
#pickle.dump(knn, knnPickle)

# Close the file
#knnPickle.close()

#Main App
class RecommenderApp(tk.Tk):
    def __init__(self):
        super().__init__()
  
	#window
        self.title("Recommender System")
        self.geometry("900x450")
        self.resizable(False, False)        
        self.create_widgets()
    #widgets    
    def create_widgets(self):
        buttonwidth = 40
        offsetx = 50
        offsety = 220
        self.movie_entry = tk.Entry(self, width=30, font=('Verdana 11'))
        self.movie_entry.grid(row=0, column=0, padx =10, pady = 10)
        self.movie_button = tk.Button(self, font=('Verdana'), text = "Search", command = self.search_movie)
        self.movie_button.grid(row=0, column = 1)
        self.movie_name_text = Text(self, height = 1, width = 30, font=('Verdana 11'), state = 'disabled', background = "#efefef")
        self.movie_name_text.grid(row=2, column = 0, )
        self.movie_id_text = Text(self, height = 1, width = 10, font=('Verdana 11'), state = 'disabled', background = "#efefef")
        self.movie_id_text.grid(row=2, column = 1, )
        self.movie_genre_text = None
        self.movie_name_label = Label(self, font=('Verdana 11'), text="ID")
        self.movie_name_label.grid(row=3, column = 1, )        
        self.movie_name_label = Label(self, font=('Verdana 11'), text="Movie Name")
        self.movie_name_label.grid(row=3, column = 0, )
        self.movie_rating_text_label = Label(self, font=('Verdana 11'), text="Movie Rating:")
        self.movie_rating_text_label.grid(row=5, column = 0, padx = 0, pady = 10)        
        self.movie_rating_label = Label(self, font=('Verdana 11'), text="5")
        self.movie_rating_label.grid(row=5, column = 1, padx =0, pady = 10)
        self.rating = IntVar()
        self.R1 = Radiobutton(self, text="1", variable=self.rating, value=1, command=self.select)
        self.R1.place(x=(offsetx), y=(offsety)) 
        self.R2 = Radiobutton(self, text="2", variable=self.rating, value=2, command=self.select)
        self.R2.place(x=(offsetx*2), y=(offsety)) 
        self.R3 = Radiobutton(self, text="3", variable=self.rating, value=3, command=self.select)
        self.R3.place(x=(offsetx*3), y=(offsety)) 
        self.R4 = Radiobutton(self, text="4", variable=self.rating, value=4, command=self.select)
        self.R4.place(x=(offsetx*4), y=(offsety)) 
        self.R5 = Radiobutton(self, text="5", variable=self.rating, value=5, command=self.select)
        self.R5.place(x=(offsetx*5), y=(offsety))
        self.rating.set(5)
        self.distance_method = IntVar()
        self.R6 = Radiobutton(self, text="euclidian", variable=self.distance_method, value=1, command=self.select)
        self.R6.place(x=(offsetx-10), y=(offsety+130)) 
        self.R7 = Radiobutton(self, text="minkowski", variable=self.distance_method, value=2, command=self.select)
        self.R7.place(x=(offsetx*3-10), y=(offsety+130)) 
        self.R8 = Radiobutton(self, text="manhattan", variable=self.distance_method, value=4, command=self.select)
        self.R8.place(x=(offsetx*5-10), y=(offsety+130)) 
        self.R9 = Radiobutton(self, text="cosine", variable=self.distance_method, value=3, command=self.select)
        self.R9.place(x=(offsetx*7-10), y=(offsety+130)) 
        self.R10 = Radiobutton(self, text="suggest title based on popuarity", variable=self.distance_method, value=5, command=self.select)
        self.R10.place(x=(offsetx-10), y=(offsety+170)) 
        self.distance_method.set(1)
        self.distance_method_text_label = Label(self, font=('Verdana 11'), text="Distance Method:")
        self.distance_method_text_label.place(x = offsetx*2, y = offsety+100) 
        self.movie_button = tk.Button(self, font=('Verdana'), text = "Submit", command = self.submit_rating)
        self.movie_button.grid(row=6, column = 1, padx = 0, pady= 10)

        #Rate one of these movies or search for one to rate
        self.recommended_label = Label(self, font=('Verdana 11'), text="Initial recomendations:")
        self.recommended_label.grid(row=0, column = 4, )

        #Cold-Start Recommendations
        for i in range(len(recommendations)):
            recommendations[i]=self.search_initial_movie(coldstart_list[i])

        self.buttons = []

        for r in range(0, 10):
            btn = tk.Button(self, text=recommendations[r][0],height='1', width='50', font=("Verdana", 11))
            btn.grid(row=r+1,column=4)
            btn.bind('<Button-1>', self.recommendation_button)
            self.buttons.append(btn)

    def recommendation_button(self, event):
        self.movie_entry.delete(0,tk.END)
        self.movie_entry.insert (tk.END, event.widget['text'])
        self.search_movie()
        
#   returns values to populate initial buttons
    def search_recommendation(self, button_string):
        self.movie_entry.delete(0,tk.END)
        self.movie_entry.configure(text = button_string)
        
#   returns values to populate initial buttons
    def search_initial_movie(self, movie_string):

        df_results = df_movies[df_movies['title'].str.contains(movie_string)].merge(df_genres[df_genres['genres'].str.contains(self.movie_id_text.get("1.0",'end-1c'))])
        
    #   return the top result only
        df_results = df_results.head(1)
        
        movie_name = (df_results['title'].to_string(index=False))
        movie_id = (df_results['movieId'].to_string(index=False))
        return movie_name, movie_id

#   TODO Improve search to prioritize exact matches
    def search_movie(self):
        
        if not self.movie_entry.get():
            tk.messagebox.showwarning(title= "Warning", message="Search field empty")
            return 
        
        df_results = df_movies[df_movies['title'].str.contains(self.movie_entry.get().split('(')[0])]
        
	#   return the top result only
        df_results = df_results.head(1)
        
        movie_name = (df_results['title'].to_string(index=False))
        movie_id = (df_results['movieId'].to_string(index=False))
        
        self.movie_name_text.configure(state='normal')
        self.movie_name_text.delete("1.0","end")        
        self.movie_name_text.insert(tk.END, movie_name)
        self.movie_name_text.configure(state='disabled')

        self.movie_id_text.configure(state='normal')
        self.movie_id_text.delete("1.0","end")
        self.movie_id_text.configure(state='normal')
        self.movie_id_text.insert(tk.END, movie_id)
        self.movie_id_text.configure(state='disabled')

    def select(self):
        global knn
        global knn_euc
        global knn_mink
        global knn_cos
        global knn_manhattan

        self.movie_rating_label.config(text = str(self.rating.get()))        
        if self.distance_method.get() == 1:
            knn = knn_euc
        if self.distance_method.get() == 2:
            knn = knn_mink            
        if self.distance_method.get() == 3:
            knn = knn_cos 
        if self.distance_method.get() == 4:
            knn = knn_manhattan 

        knn.fit(df_pivoted)

    def submit_rating(self):
        global rated_df

        rating = self.rating.get()
        movieid = int(self.movie_id_text.get("1.0",'end-1c'))
        movie_name = self.movie_name_text.get("1.0",'end-1c')
        movie_row = df_genres.loc[df_genres['movieId'] == movieid]
        
        movieGenreFull = movie_row['genres'].iloc[0]
        #list of movie's genre(s)
        movieGenreFullList = movieGenreFull.split("|")

        #prioritizing/deprioritizing genres based on user's rating
        for movieGenre in movieGenreFullList:
            if rating == 1:
                genreDict[movieGenre] -=2
            elif rating == 2:
                genreDict[movieGenre] -=1
            elif rating == 4:
                genreDict[movieGenre] +=1
            elif rating == 5:
                genreDict[movieGenre] +=2
        
        #reseting any genre with scores +20 or -20
        for key,value in genreDict.items():
            if value > 20 or value < -20 :
                genreDict[key] = 0

        #topGenresSelected = nlargest(2, genreDict, key=genreDict.get)
        
        #we're picking the most popular and the most unpopular user selected genres and calculate popular list based on these two
        movieGenre = max(genreDict, key=genreDict.get)+"|"+min(genreDict, key=genreDict.get)

        # removing selected title from the pool
        blacklist.append(movie_name)

        new_df = pd.DataFrame({'userId':  [162543], 'movieId': [movieid]})      
        if rated_df.values[0][1]==0:
            rated_df = new_df 

        else:
            rated_df = pd.concat([rated_df, new_df])

        self.recommended_label.configure(text = "New Recommendations")

        # Get the Popular movies based on user's selected movie genre
        if self.distance_method.get() == 5:
            list = TopWeighted.top_weighted_movies(movieGenre, blacklist)
            for r in range(0, 10):
                name = list[r]
                self.buttons[r].config(text = name)
                recommendations[r]=name

        else :
            # Get the k most similar movies to the user's rated movies
            distances, indices = knn.kneighbors(df_pivoted.loc[[movieid]], return_distance=True)
            similar_movies = [df_pivoted.index[idx] for idx in indices[0]]

            for r in range(0, 10):
                r_movie_id = similar_movies[r]            
                r_movie_name = df_movies.loc[df_movies['movieId']== r_movie_id, 'title']
                self.buttons[r].config(text = r_movie_name.to_string(index=False))
                recommendations[r] = r_movie_id, r_movie_name
    
    
if __name__ == "__main__":
    app = RecommenderApp()
    app.mainloop()
