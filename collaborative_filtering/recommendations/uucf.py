"""
This module implements the User-User Collaborative Filtering (UUCF) algorithm
for personalized movie recommendations. It predicts user ratings based on
similarities between users and provides top-N recommendations.

@author: Nicola Maritan
"""
__author__ = "Nicola Maritan"

import pandas as pd
import numpy as np
import heapq
from collections import defaultdict

class UserUserCollaborativeFilter:
    """
    A class to implement User-User Collaborative Filtering (UUCF) for movie recommendations.
    """

    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        """
        Initializes the UUCF model with movie and rating data.

        Args:
            movies_df (pd.DataFrame): DataFrame containing movie information.
            ratings_df (pd.DataFrame): DataFrame containing user ratings for movies.
        """

        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self._init_user_movie_ratings_dict()
        self._init_mean_ratings_and_normalize_ratings_dict()
        
    def _init_user_movie_ratings_dict(self):
        """
        Initializes a dictionary mapping users to their movie ratings.
        """

        self.ratings_dict = defaultdict(dict)
        for _, row in self.ratings_df.iterrows():
            self.ratings_dict[int(row['userId'])][int(row['movieId'])] = row['rating']
        
    def _init_mean_ratings_and_normalize_ratings_dict(self):
        """
        Computes and stores mean ratings for each user and normalizes their ratings.
        """

        self.user_mean_rating = dict()
        for userId in self.ratings_dict.keys():
            mean = np.mean([rating for rating in self.ratings_dict[userId].values()])

            # Store for rating prediction
            self.user_mean_rating[userId] = mean
            
            # Store normalized ratings
            for movieId in self.ratings_dict[userId]:
                self.ratings_dict[userId][movieId] -= mean

    def get_overlapping_movie_ids(self, userId_1: int, userId_2: int):
        """
        Finds movies rated by both users.

        Args:
            userId_1 (int): ID of the first user.
            userId_2 (int): ID of the second user.

        Returns:
            set: Set of overlapping movie IDs.
        """

        if userId_1 not in self.ratings_dict:
            raise ValueError("userId_1 not found during initialization. Invalid user ID.")
        if userId_2 not in self.ratings_dict:
            raise ValueError("userId_2 not found during initialization. Invalid user ID.")
        return set(self.ratings_dict[userId_1]) & set(self.ratings_dict[userId_2]) 
    
    def pearson_correlation(self, userId_1: int, userId_2: int, gamma: float = None):
        """
        Computes the Pearson correlation between two users.

        Args:
            userId_1 (int): ID of the first user.
            userId_2 (int): ID of the second user.
            gamma (float, optional): Weighting factor.

        Returns:
            float: Pearson correlation coefficient.
        """

        if userId_1 not in self.ratings_dict:
            raise ValueError("userId_1 not found during initialization. Invalid user ID.")
        if userId_2 not in self.ratings_dict:
            raise ValueError("userId_2 not found during initialization. Invalid user ID.")
        if gamma is not None and gamma <= 0:
            raise ValueError("gamma must be a positive number.")

        overlapping_movie_ids = self.get_overlapping_movie_ids(userId_1, userId_2)
        if len(overlapping_movie_ids) <= 1:
            return 0

        covariance = sum([self.ratings_dict[userId_1][movieId] * self.ratings_dict[userId_2][movieId] for movieId in overlapping_movie_ids])
        std_1 = np.linalg.norm([self.ratings_dict[userId_1][movieId] for movieId in overlapping_movie_ids])
        std_2 = np.linalg.norm([self.ratings_dict[userId_2][movieId] for movieId in overlapping_movie_ids])
        if std_1 == 0 or std_2 == 0:
            return 0

        correlation = covariance / (std_1 * std_2)

        return correlation * min(gamma, len(overlapping_movie_ids)) / gamma if gamma is not None else correlation 

    def get_neighbors(self, userId: int, movieId: int, gamma: float = None):
        """
        Finds neighbors who have rated a specific movie.

        Args:
            userId (int): ID of the target user.
            movieId (int): ID of the target movie.
            gamma (float, optional): Weighting factor.

        Returns:
            dict: Dictionary of neighbors and their similarity scores.
        """

        # Validate input parameters
        if userId not in self.ratings_dict:
            raise ValueError("userId not found during initialization. Invalid user ID.")
        if movieId not in self.movies_df.index:
            raise ValueError("movieId not found during initialization. Invalid movie ID.")
        if gamma is not None and gamma <= 0:
            raise ValueError("gamma must be a positive number.")
        
        neighbors = dict() 
        for other_userId in self.ratings_dict:
            # First assure movieId is rated by both
            if movieId not in self.ratings_dict[other_userId]: 
                continue

            # If rated by both, compute Pearson Correlation
            similarity = self.pearson_correlation(userId, other_userId, gamma)

            if similarity > 0:
                neighbors[other_userId] = similarity

        return neighbors

    def top_k_neighbors(self, userId: int, movieId: int, k: int, gamma: float = None):
        """
        Finds the top-K most similar neighbors for a user and movie.

        Args:
            userId (int): ID of the target user.
            movieId (int): ID of the target movie.
            k (int): Number of top neighbors to retrieve.
            gamma (float, optional): Weighting factor.

        Returns:
            list: List of top-K neighbors and their similarity scores.
        """

        # Validate input parameters
        if userId not in self.ratings_dict:
            raise ValueError("userId not found during initialization. Invalid user ID.")
        if movieId not in self.movies_df.index:
            raise ValueError("movieId not found during initialization. Invalid movie ID.")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        if gamma is not None and gamma <= 0:
            raise ValueError("gamma must be a positive number.")

        neighbors = self.get_neighbors(userId, movieId, gamma)

        similarities = (
            (userId, similarity)
            for userId, similarity in neighbors.items()
        )

        top_k = heapq.nsmallest(
            k,
            similarities,
            key=lambda x: (-x[1], x[0])  # Negate score to simulate descending order
        )

        return top_k

    def get_deviation_from_mean_rating(self, userId: int, movieId: int, k: int, gamma: float = None):
        """
        Computes the deviation from the mean rating for a specific movie.

        Args:
            userId (int): ID of the target user.
            movieId (int): ID of the target movie.
            k (int): Number of top neighbors to consider.
            gamma (float, optional): Weighting factor.

        Returns:
            float: Deviation from the mean rating.
        """

        # Validate input parameters
        if userId not in self.ratings_dict:
            raise ValueError("userId not found during initialization. Invalid user ID.")
        if movieId not in self.movies_df.index:
            raise ValueError("movieId not found during initialization. Invalid movie ID.")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        if gamma is not None and gamma <= 0:
            raise ValueError("gamma must be a positive number.")
        
        neighbors = self.top_k_neighbors(userId, movieId, k, gamma)

        if len(neighbors) == 0:
            return 0

        # At this point, since all similarities are > 0, if len != 0 the denominator will
        # always be positive, so I don't do any particular check for it.

        numerator = sum([self.ratings_dict[neighbor_userId][movieId] * similarity for neighbor_userId, similarity in neighbors])
        denominator = sum([similarity for _, similarity in neighbors])
        return numerator / denominator

    def predict_rating(self, userId: int, movieId: int, k: int, gamma: float = None):
        """
        Predicts the rating for a specific movie by a user.

        Args:
            userId (int): ID of the target user.
            movieId (int): ID of the target movie.
            k (int): Number of top neighbors to consider.
            gamma (float, optional): Weighting factor.

        Returns:
            float: Predicted rating.
        """

        # Validate input parameters
        if userId not in self.ratings_dict:
            raise ValueError("userId not found during initialization. Invalid user ID.")
        if movieId not in self.movies_df.index:
            raise ValueError("movieId not found during initialization. Invalid movie ID.")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        if gamma is not None and gamma <= 0:
            raise ValueError("gamma must be a positive number.")

        deviation = self.get_deviation_from_mean_rating(userId, movieId, k, gamma)
        return deviation + self.user_mean_rating[userId]

    def top_n_recommendations(self, userId: int, n: int, k: int, gamma: float = None):
        """
        Provides the top-N movie recommendations for a user.

        Args:
            userId (int): ID of the target user.
            n (int): Number of recommendations to retrieve.
            k (int): Number of top neighbors to consider.
            gamma (float, optional): Weighting factor.

        Returns:
            list: List of top-N recommended movies and their predicted ratings.
        """

        # Validate input parameters
        if userId not in self.ratings_dict:
            raise ValueError("userId not found during initialization. Invalid user ID.")
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