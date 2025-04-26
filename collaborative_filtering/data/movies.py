"""
Movies Module.
Provides functions for loading movie data,
extracting genres, and retrieving movie details.

@author: Nicola Maritan
"""
__author__ = "Nicola Maritan"

import pandas as pd
from collections import Counter

NO_GENRES_LISTED_VALUE = "(no genres listed)"

def load_movies(movies_path: str) -> pd.DataFrame:
    """
    Load movies from a file into a Pandas DataFrame.

    Parameters:
    - movies_path (str): The file path to the movies dataset.

    Returns:
    - pd.DataFrame: A DataFrame containing movies indexed by `movieId`.
    """
    movies_df = pd.read_csv(
        movies_path, 
        sep=",", 
        engine="python",
    )

    # Convert genre strings into lists for easy access
    movies_df["genres"] = movies_df["genres"].str.split("|")

    return movies_df.set_index("movieId")

def get_movie(movies_df: pd.DataFrame, movieId: int) -> dict:
    """
    Get movie details for a given movie ID.

    Parameters:
    - movies_df (pd.DataFrame): The DataFrame containing movies.
    - movieId (int): The ID of the movie.

    Returns:
    - dict: A dictionary containing movie details.

    Raises:
    - KeyError: If the movie ID is not found in the dataset.
    """
    if movieId not in movies_df.index:
        raise KeyError(f"Movie ID {movieId} not found.")
    return movies_df.loc[movieId].to_dict()