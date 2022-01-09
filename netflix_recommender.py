"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.content_based import content_model

# Data Loading
movies_data = pd.read_csv("resources/data/movies_data.csv")

# App declaration
def main():

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    # Header contents
    st.write('# Netflix Movie Recommendation System')
    st.image('resources/imgs/Image_header.png',use_column_width=True)

    # User-based preferences
    movie = st.selectbox('Enter your favorite movie',movies_data["title"],)
    n_movies = st.select_slider('How many movies to recommend?',options=range(1,11))
    # features = st.multiselect('Important Variables', ['title','director','cast','listed_in','country'])
    container = st.container()
    all = st.checkbox("Select All Variables")
     
    if all:
        selected_variables = container.multiselect("Important Variables",
                    ['title','director','cast','listed_in','country'],['title','director','cast','listed_in','country'])
    else:
        selected_variables =  container.multiselect("Important Variables",
            ['title','director','cast','listed_in','country'])
    
    # Perform top-10 movie recommendation generation
    if st.button("Recommend"):
        try:
            with st.spinner('Crunching the numbers...'):
                top_recommendations = content_model(movie=movie,top_n = n_movies,features=selected_variables)
            st.title("We think you'll like the below movie(s):")
            for i,j in enumerate(top_recommendations):
                st.header(str(i+1)+'. '+j)
        except:
            st.error("Oops! Looks like this algorithm does't work.\
                      Please check the provided parameters again!")


if __name__ == '__main__':
    main()
