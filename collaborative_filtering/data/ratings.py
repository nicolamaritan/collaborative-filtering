"""
Ratings Module.
Provides functions for loading rating data,
accessing user ratings, and processing ratings.

@author: Nicola Maritan
"""
__author__ = "Nicola Maritan"

import pandas as pd

def load_ratings(ratings_path: str) -> pd.DataFrame:
    """
    Load ratings from a file into a Pandas DataFrame.

    Parameters:
    - ratings_path (str): The file path to the ratings dataset.

    Returns:
    - pd.DataFrame: A DataFrame containing user ratings.
    """
    ratings_df = pd.read_csv(
        ratings_path, 
        sep=",", 
        engine="python",
        dtype={"userId": int, "movieId": int}
    )
    return ratings_df