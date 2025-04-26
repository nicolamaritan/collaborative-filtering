"""
This module implements a hybrid recommender system that combines User-User
Collaborative Filtering (UUCF) and Item-Item Collaborative Filtering (IICF)
to provide personalized movie recommendations.

The hybrid approach averages predictions from both UUCF and IICF to improve
recommendation accuracy.

@author: Nicola Maritan
"""
__author__ = "Nicola Maritan"

from collaborative_filtering.recommendations.uucf import UserUserCollaborativeFilter 
from collaborative_filtering.recommendations.iicf import ItemItemCollaborativeFilter 
import heapq
import pandas as pd
from collections import defaultdict

class ItemItemUserUserHybridRecommender:
    """
    A hybrid recommender system that combines User-User Collaborative Filtering (UUCF)
    and Item-Item Collaborative Filtering (IICF) to provide personalized recommendations.
    """

    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        """
        Initializes the hybrid recommender with movie and rating data.

        Args:
            movies_df (pd.DataFrame): DataFrame containing movie information.
            ratings_df (pd.DataFrame): DataFrame containing user ratings for movies.
        """

        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.uucf_rec = UserUserCollaborativeFilter(movies_df, ratings_df)
        self.iicf_rec = ItemItemCollaborativeFilter(movies_df, ratings_df)
        self._init_user_movie_ratings_dict()

    def _init_user_movie_ratings_dict(self):
        """
        Initializes a dictionary mapping users to their movie ratings.
        """

        self.ratings_dict = defaultdict(dict)
        for _, row in self.ratings_df.iterrows():
            self.ratings_dict[int(row['userId'])][int(row['movieId'])] = row['rating']

    def predict_rating(self, userId: int, movieId: int, k: int, gamma: float = None):
        """
        Predicts the rating for a specific movie by a user using the hybrid approach.

        Args:
            userId (int): ID of the target user.
            movieId (int): ID of the target movie.
            k (int): Number of top neighbors or similar items to consider.
            gamma (float, optional): Weighting factor for UUCF (optional).

        Returns:
            float: Predicted rating as the average of UUCF and IICF predictions.
        """

        # Validate input parameters
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        if gamma is not None and gamma <= 0:
            raise ValueError("gamma must be a positive number.")
        
        return (self.uucf_rec.predict_rating(userId, movieId, k, gamma) + self.iicf_rec.predict_rating(userId, movieId, k)) / 2

    def top_n_recommendations(self, userId: int, n: int, k: int, gamma: float = None):
        """
        Provides the top-N movie recommendations for a user using the hybrid approach.

        Args:
            userId (int): ID of the target user.
            n (int): Number of recommendations to retrieve.
            k (int): Number of top neighbors or similar items to consider.
            gamma (float, optional): Weighting factor for UUCF (optional).

        Returns:
            list: List of top-N recommended movies and their predicted ratings.
        """

        # Validate input parameters
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer.")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        if gamma is not None and gamma <= 0:
            raise ValueError("gamma must be a positive number.")

        # Builds predictions and filters out movies already rated by the user ("if movieId not in self.ratings_dict[userId]" at the end).
        predictions = [(movieId, self.predict_rating(userId, movieId, k, gamma)) for movieId, _ in self.movies_df.iterrows() if movieId not in self.ratings_dict[userId]]

        # Compute the top-n recommendations
        top_n = heapq.nsmallest(
            n,
            predictions,
            key=lambda x: (-x[1], x[0])  # Negate score to simulate descending order
        )

        return top_n

