"""
This module implements a basket-based recommender system that provides
personalized movie recommendations based on a user's basket of previously
rated or selected movies.

The recommender uses movie-movie similarities to compute scores for
unrated movies and recommends the top-N movies with the highest scores.

@author: Nicola Maritan
"""
__author__ = "Nicola Maritan"

from collaborative_filtering.recommendations.similarity_model import SimilarityModel
import pandas as pd
import heapq

class BasketRecommender:
    """
    A basket-based recommender system that provides personalized movie
    recommendations based on a user's basket of selected movies.
    """

    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        """
        Initializes the basket recommender with movie and rating data.

        Args:
            movies_df (pd.DataFrame): DataFrame containing movie information.
            ratings_df (pd.DataFrame): DataFrame containing user ratings for movies.
        """

        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.similarities = SimilarityModel(self.movies_df, self.ratings_df)

    def get_score(self, movieId: int, basket, only_positive_similarities: bool = True):
        """
        Computes the score for a movie based on its similarity to movies in the basket.

        Args:
            movieId (int): ID of the target movie.
            basket (list): List of movie IDs in the user's basket.
            only_positive_similarities (bool): Whether to consider only positive similarities.

        Returns:
            float: The computed score for the movie.
        """

        score = 0
        for other_movieId in basket:
            similarity = self.get_similarity(movieId, other_movieId, only_positive_similarities)
            if similarity is not None:
                score += similarity
        return score

    def top_n_recommendations(self, basket, n: int, only_positive_similarities: bool = True):
        """
        Provides the top-N movie recommendations based on the user's basket.

        Args:
            basket (list): List of movie IDs in the user's basket.
            n (int): Number of recommendations to retrieve.
            only_positive_similarities (bool): Whether to consider only positive similarities.

        Returns:
            list: List of top-N recommended movies and their computed scores.
        """

        # Validate input parameter
        if n <= 0:
            raise ValueError("n must be a positive integer.")

        # Builds scores and filters out movies already rated in the basket ("if movieId not in basket" at the end).
        scores = [(movieId, self.get_score(movieId, basket, only_positive_similarities)) for movieId, _ in self.movies_df.iterrows() if movieId not in basket]

        # Compute the top-n recommendations
        top_n = heapq.nsmallest(
            n,
            scores,
            key=lambda x: (-x[1], x[0])  # Negate score to simulate descending order
        )

        return top_n

    def get_similarity(self, movieId_1: int, movieId_2: int, only_positive_similarities: bool = True):
        """
        Retrieves the similarity between two movies.

        Args:
            movieId_1 (int): ID of the first movie.
            movieId_2 (int): ID of the second movie.
            only_positive_similarities (bool): Whether to consider only positive similarities.

        Returns:
            float or None: The similarity value if available, otherwise None.
        """

        # Compute similarity. It depends on whether we want to consider only positive similarities or not.
        if only_positive_similarities:
            return self.similarities.get_positive_similarity(movieId_1, movieId_2)

        return self.similarities.get_similarity(movieId_1, movieId_2)