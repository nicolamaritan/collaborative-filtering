# Collaborative Filtering
Set of collaborative filtering recommender systems for the MovieLens dataset.
## Requirements
Download a version of the MovieLens dataset from [the official site](https://grouplens.org/datasets/movielens/).
## Demo execution 
From the `examples/` directory of the source code run
```
python demo.py <movie_path> <rating_path>
```
## Dependencies
- NumPy
- Pandas

## Fast similarity computation
This module optimizes similarity calculations for IICF through several strategies:
- **Singleton Pattern**: Ensures only one instance is created to avoid redundant data and computations.
- **Precomputation and Caching**: All movie-to-movie similarities and norms are computed once and stored for fast retrieval.
- **Sparse Storage**: Only non-zero similarities are saved, significantly reducing memory usage.
- **Symmetry Exploitation**: Only one direction of each movie pair is stored, leveraging the symmetry of similarity.
- **Normalized Ratings**: Ratings are mean-centered ahead of time to simplify and speed up similarity calculations.
- **Optimized Lookups**: Movie pairs are ordered consistently to allow constant-time dictionary access.