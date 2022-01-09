"""
Helper functions for data loading and manipulation.
"""
# Data handling dependencies
import pandas as pd
import numpy as np

def load_movie_titles(path_to_movies):
    """Load movie titles from database records.
    Returns
    list[str]
        Movie titles.
    """
    df = pd.read_csv(path_to_movies)
    df = df.dropna()
    movie_list = list(df['title'])
    return movie_list
