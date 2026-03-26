"""
train.py — Run once to build and save the recommendation model.

Usage:
    python train.py
    python train.py --movies path/to/movies.csv --credits path/to/credits.csv
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Train the CineMatch recommender model')
    parser.add_argument('--movies',  default='tmdb_5000_movies.csv',  help='Path to movies CSV')
    parser.add_argument('--credits', default='tmdb_5000_credits.csv', help='Path to credits CSV')
    args = parser.parse_args()

    if not os.path.exists(args.movies):
        print(f"❌ Movies CSV not found: {args.movies}")
        print("   Download the TMDB 5000 dataset from:")
        print("   https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
        sys.exit(1)

    if not os.path.exists(args.credits):
        print(f"❌ Credits CSV not found: {args.credits}")
        sys.exit(1)

    print("🎬 Starting model training...")
    from recommender import MovieRecommender
    rec = MovieRecommender()
    rec.build(args.movies, args.credits)
    print(f"✅ Model saved to {rec.MODEL_PATH}")
    print("   You can now run:  python app.py")

if __name__ == '__main__':
    main()
