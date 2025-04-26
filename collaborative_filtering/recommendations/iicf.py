"""
This module implements the Item-Item Collaborative Filtering (IICF) algorithm
for personalized movie recommendations. It predicts user ratings based on
similarities between items and provides top-N recommendations.

@author: Nicola Maritan
"""
__author__ = "Nicola Maritan"

from collaborative_filtering.recommendations.similarity_model import SimilarityModel
import pandas as pd
import heapq
from collections import defaultdict

class ItemItemCollaborativeFilter:
    """
    A class to implement Item-Item Collaborative Filtering (IICF) for movie recommendations.
    This implementation only considers positive similarities between movies.
    """

    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        """
        Initializes the IICF model with movie and rating data.

        Args:
            movies_df (pd.DataFrame): DataFrame containing movie information.
            ratings_df (pd.DataFrame): DataFrame containing user ratings for movies.
        """

        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.similarities = SimilarityModel(self.movies_df, self.ratings_df)
        self._init_user_movie_ratings_dict()

    def _init_user_movie_ratings_dict(self):
        """
        Initializes a dictionary mapping users to their movie ratings.
        """

        self.ratings_dict = defaultdict(dict)
        for _, row in self.ratings_df.iterrows():
            self.ratings_dict[int(row['userId'])][int(row['movieId'])] = row['rating']

    def get_rated_positive_similarities(self, userId: int, movieId: int):
        """
        Retrieves positive similarities between a target movie and movies rated by a user.

        Args:
            userId (int): ID of the target user.
            movieId (int): ID of the target movie.

        Returns:
            list: List of tuples containing movie IDs and their positive similarity scores.
        """

        # Validate input parameter
        if userId not in self.ratings_dict:
            raise ValueError("userId not found during initialization. Invalid user ID.")
        if movieId not in self.movies_df.index:
            raise ValueError("movieId not found during initialization. Invalid movie ID.")

        rated_similarities = []
        for other_movieId in self.ratings_dict[userId]:
            similarity = self.similarities.get_positive_similarity(movieId, other_movieId)
            if similarity is not None:
                rated_similarities.append((other_movieId, similarity))
        return rated_similarities
        
    def top_k_most_similar(self, userId: int, movieId: int, k: int):
        """
        Finds the top-K most similar movies to a target movie rated by a user.
        Only positive similarities are used to compute the top-K most similar
        movies.

        Args:
            userId (int): ID of the target user.
            movieId (int): ID of the target movie.
            k (int): Number of top similar (with positive similarity) movies to retrieve.

        Returns:
            list: List of top-K similar (with positive similarity) movies and their similarity scores.
        """

        # Validate input parameter
        if userId not in self.ratings_dict:
            raise ValueError("userId not found during initialization. Invalid user ID.")
        if movieId not in self.movies_df.index:
            raise ValueError("movieId not found during initialization. Invalid movie ID.")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")

        rated_similarities = self.get_rated_positive_similarities(userId, movieId)

        top_k = heapq.nsmallest(
            k,
            rated_similarities,
            key=lambda x: (-x[1], x[0])  # Negate score to simulate descending order
        )

        return top_k

    def predict_rating(self, userId: int, movieId: int, k: int):
        """
        Predicts the rating for a specific movie by a user.

        Args:
            userId (int): ID of the target user.
            movieId (int): ID of the target movie.
            k (int): Number of top similar movies to consider.

        Returns:
            float: Predicted rating.
        """

        # Validate input parameter
        if userId not in self.ratings_dict:
            raise ValueError("userId not found during initialization. Invalid user ID.")
        if movieId not in self.movies_df.index:
            raise ValueError("movieId not found during initialization. Invalid movie ID.")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")

        top_k_most_similar = self.top_k_most_similar(userId, movieId, k)

        if len(top_k_most_similar) == 0:
            return 0

        # At this point we are sure that the denominator will always be positive, so I don't do any particular check for it.
        # The reason is the similarities are all positive, and summing them will always be positive.
        numerator = sum([self.ratings_dict[userId][other_movieId] * similarity for other_movieId, similarity in top_k_most_similar])
        denominator = sum([similarity for _, similarity in top_k_most_similar]) 

        return numerator / denominator

    def top_n_recommendations(self, userId: int, n: int, k: int):
        """
        Provides the top-N movie recommendations for a user.

        Args:
            userId (int): ID of the target user.
            n (int): Number of recommendations to retrieve.
            k (int): Number of top similar movies to consider.

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

        # Builds predictions and filters out movies already rated by the user ("if movieId not in self.ratings_dict[userId]" at the end).
        predictions = [(movieId, self.predict_rating(userId, movieId, k)) for movieId, _ in self.movies_df.iterrows() if movieId not in self.ratings_dict[userId]]

        # Compute the top-n recommendations
        top_n = heapq.nsmallest(
            n,
            predictions,
            key=lambda x: (-x[1], x[0])  # Negate score to simulate descending order
        )

        return top_n

    def get_similarity(self, movieId_1: int, movieId_2: int):
        """
        Retrieves the positive similarity between two movies.

        Args:
            movieId_1 (int): ID of the first movie.
            movieId_2 (int): ID of the second movie.

        Returns:
            float or None: Positive similarity value if available, otherwise None.
        """

        # Validate input parameters
        if movieId_1 not in self.movies_df.index:
            raise ValueError("movieId_1 not found during initialization. Invalid movie ID.")
        if movieId_2 not in self.movies_df.index:
            raise ValueError("movieId_2 not found during initialization. Invalid movie ID.")

        return self.similarities.get_positive_similarity(movieId_1, movieId_2)