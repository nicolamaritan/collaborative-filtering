import sys, os
sys.path.append(os.path.abspath(os.path.join('..', '.')))

from collaborative_filtering.data.movies import * 
from collaborative_filtering.data.ratings import *
from collaborative_filtering.recommendations.uucf import UserUserCollaborativeFilter
from collaborative_filtering.recommendations.iicf import ItemItemCollaborativeFilter 
from collaborative_filtering.recommendations.basket import BasketRecommender 
from collaborative_filtering.recommendations.hybrid import ItemItemUserUserHybridRecommender 
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: python practical.py <movie_path> <rating_path>")
        sys.exit(1)

    movie_path = sys.argv[1]
    rating_path = sys.argv[2]
    
    movies_df = load_movies(movie_path)
    ratings_df = load_ratings(rating_path)
    uucf_rec = UserUserCollaborativeFilter(movies_df, ratings_df)

    GAMMA = 10
    K = 20
    N = 10

    top_n = uucf_rec.top_n_recommendations(userId = 522, n = N, k = K, gamma = GAMMA) 
    top_n_df = pd.DataFrame(top_n, columns=['movieId', 'rating prediction'])
    top_n_df.index = top_n_df.index + 1
    top_n_df.index.name = 'rank'
    top_n_df = top_n_df.join(movies_df, on='movieId')
    print(top_n_df.reset_index()[['rank', 'movieId', 'title', 'rating prediction']].to_string(index = False))

    iicf_rec = ItemItemCollaborativeFilter(movies_df, ratings_df)
    top_n = iicf_rec.top_n_recommendations(userId = 522, n = N, k = K) 
    top_n_df = pd.DataFrame(top_n, columns=['movieId', 'rating prediction'])
    top_n_df.index = top_n_df.index + 1
    top_n_df.index.name = 'rank'
    top_n_df = top_n_df.join(movies_df, on='movieId')
    print(top_n_df.reset_index()[['rank', 'movieId', 'title', 'rating prediction']].to_string(index = False))

    basket_rec = BasketRecommender(movies_df, ratings_df)
    top_n = basket_rec.top_n_recommendations(basket = [1, 48, 239], n = N, only_positive_similarities = True)
    top_n_df = pd.DataFrame(top_n, columns=['movieId', 'score'])
    top_n_df.index = top_n_df.index + 1
    top_n_df.index.name = 'rank'
    top_n_df = top_n_df.join(movies_df, on='movieId')
    print(top_n_df.reset_index()[['rank', 'movieId', 'title', 'score']].to_string(index = False))

    hybrid_rec = ItemItemUserUserHybridRecommender(movies_df, ratings_df)
    top_n = hybrid_rec.top_n_recommendations(userId = 522, n = N, k = K, gamma = GAMMA) 
    top_n_df = pd.DataFrame(top_n, columns=['movieId', 'rating prediction'])
    top_n_df.index = top_n_df.index + 1
    top_n_df.index.name = 'rank'
    top_n_df = top_n_df.join(movies_df, on='movieId')
    print(top_n_df.reset_index()[['rank', 'movieId', 'title', 'rating prediction']].to_string(index = False))

if __name__ == "__main__":
    main()