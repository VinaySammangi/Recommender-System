"""
Content-based filtering for netflix movie recommendation.
"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Importing data
movies_data = pd.read_csv("resources/data/movies_data.csv")


def clean_data(df,features):
    df_subset = df[features].copy()
    df_subset['main_column'] = ""
    for feature in features:
        if feature!="description":
            df_subset[feature] = df_subset[feature].apply(lambda x: str.lower(str(x).replace(" ", "")))
        df_subset["main_column"] = df_subset["main_column"] + ' ' + df_subset[feature]
    return df_subset


def get_recommendations_new(title,n,features):
    """
    Find the similar movies to a given movie
    Args:
        title: movie title to which we find recommendations
        cosine_sim: cosine similarity matrix for finding similar movies
        n: number of movies to recommend
    Returns:
        results_df: returns a dataframe containing the list of recommended movies with rowids
        and their similarity score
    """ 
    
    all_features = ['title','director','cast','listed_in','country']
    
    movies_data_subset1 = clean_data(movies_data,all_features)
    movies_data_subset2 = clean_data(movies_data,features)
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_data_subset2['main_column'])
   
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    movies_data_subset1 = movies_data_subset1.reset_index()
    indices = pd.Series(movies_data_subset1.index, index=movies_data_subset1['title'])
    title = title.replace(' ','').lower()
    idx = indices[title]

    #pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    #sort the movies based on cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top n most similar movies
    sim_scores = sim_scores[1:(n+1)]
    # Get their movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top n most similar movies
    results_df = pd.DataFrame(movies_data['title'].iloc[movie_indices])
    results_df["score"] = np.round(np.array(sim_scores)[:,1],2)
    results_df = results_df.reset_index(drop=False)
    results_df.columns = ["RowID","Recommended Movie","Similarity Score"]
    return results_df



# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
def content_model(movie,top_n,features):
    """Performs Content filtering based upon a list of movies supplied
        by the app user.

    Parameters
    ----------
    movie : list (str)
        Favorite movie chosen by the netflix app user.
    top_n : type
        Number of top recommendations to return to the user.
    features: 
        What all features to be used by recommendation engine
    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.
    """
    recommendations_df = get_recommendations_new(movie,top_n,features)   
    return list(recommendations_df["Recommended Movie"])
