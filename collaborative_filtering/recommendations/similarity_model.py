"""
This module provides a singleton class `SimilarityModel` to compute and cache
movie-to-movie similarities based on user ratings. It supports efficient
retrieval of positive and non-positive similarities.

@author: Nicola Maritan
"""
__author__ = "Nicola Maritan"

import numpy as np
from collections import defaultdict
from itertools import combinations
import copy
import math

class SimilarityModel:
    """
    A singleton class to compute and cache movie-to-movie similarities
    based on normalized user ratings. It supports efficient similarity
    retrieval and avoids redundant computations.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Ensures that only one instance of the class is created.
        """
        if cls._instance is None:
            cls._instance = super(SimilarityModel, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, movies_df, ratings_df):
        """
        Initializes the similarity cache by computing similarities between movies.

        Args:
            movies_df (pd.DataFrame): DataFrame containing movie information.
            ratings_df (pd.DataFrame): DataFrame containing user ratings for movies.
        """
        if self._initialized:
            return
        self._initialized = True
        self.positive = dict()
        self.negative = dict()
        self.movie_norms = dict() # Cache movie norms to avoid recomputation

        ratings_dict = self._get_movie_user_ratings_dict(movies_df, ratings_df)
        self._init_similarities(ratings_dict)

    def _get_movie_user_ratings_dict(self, movies_df, ratings_df):
        """
        Creates a dictionary mapping movie IDs to user ratings.

        Args:
            movies_df (pd.DataFrame): DataFrame containing movie information.
            ratings_df (pd.DataFrame): DataFrame containing user ratings for movies.

        Returns:
            dict: A dictionary where keys are movie IDs and values are dictionaries
                  of user ratings.
        """
        ratings_dict = defaultdict(dict)

        # Init ratings_dict with all movies, as some of them do not have a rating
        for movieId, row in movies_df.iterrows():
            ratings_dict[movieId] = dict() 

        for _, row in ratings_df.iterrows():
            ratings_dict[int(row['movieId'])][int(row['userId'])] = row['rating']
        return ratings_dict

    def _init_similarities(self, ratings_dict):
        """
        Computes and caches similarities between all pairs of movies.

        Args:
            ratings_dict (dict): Dictionary of movie-user ratings.
        """
        normalized_ratings_dict = self._get_normalized_ratings_dict(ratings_dict)
        for movieId_1, movieId_2 in combinations(normalized_ratings_dict, 2):
            similarity = self._similarity(normalized_ratings_dict, movieId_1, movieId_2)
            min_id = min(movieId_1, movieId_2)
            max_id = max(movieId_1, movieId_2)

            # Save similarity in the appropriate dictionary
            # We do not save similarities = 0. Instead, in the
            # getter function for the similarity, we assume that
            # a missing pair means 0 similarity. This reduces the
            # memory print of the similarity model.
            if similarity > 0:
                self.positive[(min_id, max_id)] = similarity
            elif similarity < 0:
                self.negative[(min_id, max_id)] = similarity


    def _get_normalized_ratings_dict(self, ratings_dict):
        """
        Normalizes user ratings for each movie by subtracting the mean rating.

        Args:
            ratings_dict (dict): Dictionary of movie-user ratings.

        Returns:
            dict: Dictionary of normalized movie-user ratings.
        """
        normalized_ratings_dict = copy.deepcopy(ratings_dict)
        for movieId in ratings_dict.keys():
            movie_ratings = [rating for rating in ratings_dict[movieId].values()]
            movie_mean = np.mean(movie_ratings) if len(movie_ratings) != 0 else 0
            for userId in ratings_dict[movieId]:
                normalized_ratings_dict[movieId][userId] -= movie_mean

            # Cache movie norms for similarity computation.
            # This halved the time of the similarity computation, from ~90 seconds to ~45 seconds.
            self.movie_norms[movieId] = math.sqrt(sum(val * val for val in normalized_ratings_dict[movieId].values()))  

        return normalized_ratings_dict

    def _similarity(self, normalized_ratings_dict, movieId_1, movieId_2):
        """
        Computes the similarity between two movies.

        Args:
            normalized_ratings_dict (dict): Dictionary of normalized movie-user ratings.
            movieId_1 (int): ID of the first movie.
            movieId_2 (int): ID of the second movie.

        Returns:
            float: Similarity between the two movies.
        """
        ZERO_ABS_TOL = 1e-9

        ratings1 = normalized_ratings_dict[movieId_1]
        ratings2 = normalized_ratings_dict[movieId_2]
        overlapping = ratings1.keys() & ratings2.keys()

        if not overlapping:
            return 0
        
        # Use cached norms to avoid recomputation
        norm1 = self.movie_norms.get(movieId_1, None)
        if math.isclose(norm1, 0, abs_tol = ZERO_ABS_TOL):
            return 0
        norm2 = self.movie_norms.get(movieId_2, None)
        if math.isclose(norm2, 0, abs_tol = ZERO_ABS_TOL):
            return 0

        dot_product = sum(ratings1[userId] * ratings2[userId] for userId in overlapping)

        # Return if dot product is close to 0
        if math.isclose(dot_product, 0, abs_tol = ZERO_ABS_TOL):
            return 0

        return dot_product / (norm1 * norm2)

    def get_similarity(self, movieId_1, movieId_2):
        """
        Retrieves the similarity between two movies.

        Args:
            movieId_1 (int): ID of the first movie.
            movieId_2 (int): ID of the second movie.

        Returns:
            float or None: Similarity value if available, otherwise None.
        """
        # Trick: store similarities in a dictionary with the smallest movieId first
        # This way we can always query the similarity with the smallest movieId first
        min_id = min(movieId_1, movieId_2)
        max_id = max(movieId_1, movieId_2)
        if (min_id, max_id) in self.positive:
            return self.positive[(min_id, max_id)]
        elif (min_id, max_id) in self.negative:
            return self.negative[(min_id, max_id)]
        return 0 

    def get_positive_similarity(self, movieId_1, movieId_2):
        """
        Retrieves the positive similarity between two movies.

        Args:
            movieId_1 (int): ID of the first movie.
            movieId_2 (int): ID of the second movie.

        Returns:
            float or None: Positive similarity value if available, otherwise None.
        """
        # Same trick as get_similarity method
        min_id = min(movieId_1, movieId_2)
        max_id = max(movieId_1, movieId_2)
        return self.positive.get((min_id, max_id), None)