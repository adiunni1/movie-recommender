import os
import zipfile
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pandas as pd
import numpy as np 
from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Small dataset (100k ratings) for fast iteration. 
# Once everything works, swap the URL for ml-25m.zip for production scale. 

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ZIP_NAME = "ml-latest_small.zip"
DATASET_FOLDER = "ml-latest-small"
# ALPHA controls how strongly completion rate boosts confidence.
# Standard value in the ALS literature is 40 (Hu, Koren, Volinsky 2008)
# confidence = 1 + ALPHA * completion_rate
ALPHA = 40

# This is our core assumption: what does a star rating actually imply about
# how much of the content the user actually watched? 

# A 5 star rating almost certainly means they watched the whole thing and
# would watch again. A 1 star means they probably quit early. 
# A 3 star is ambiguous - maybe they finished but they were underwhelmed. 

RATING_TO_COMPLETION = {
    5.0 : 0.95, 
    4.5 : 0.90,
    4.0 : 0.80,
    3.5 : 0.65,
    3.0 : 0.50,
    2.5 : 0.35,
    2.0 : 0.25,
    1.5 : 0.15,
    1.0 : 0.10,
}

def download_movielens():
    """Download and extract the MovieLens dataset."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if (RAW_DIR / DATASET_FOLDER).exists():
        print(f"[1/5] Dataset already downloaded at {RAW_DIR / DATASET_FOLDER}")
        return

    print(f"[1/5] Downloading MovieLens...")
    import ssl
    import urllib.request

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    zip_path = RAW_DIR / ZIP_NAME

    with urllib.request.urlopen(MOVIELENS_URL, context=ctx) as response:
        with open(zip_path, "wb") as f:
            f.write(response.read())

    print("    Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(RAW_DIR)
    zip_path.unlink()
    print(f"    Done — extracted to {RAW_DIR / DATASET_FOLDER}")

# Raw data loading

def load_raw_data():
    """Load ratings and movies CSV files into DataFrames"""
    print("[2/5] Loading raw data...")
    ratings_path = RAW_DIR / DATASET_FOLDER / "ratings.csv"
    movies_path = RAW_DIR / DATASET_FOLDER / "movies.csv"

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    print(f"    Ratings: {len(ratings):,} rows — "
          f"{ratings['userId'].nunique():,} users, "
          f"{ratings['movieId'].nunique():,} movies")
    print(f"    Movies:  {len(movies):,} entries")
    print(f"    Rating distribution:\n"
          f"{ratings['rating'].value_counts().sort_index()}")
 
    return ratings, movies

# Explicit ratings dataset

def build_explicit(ratings, movies):
    """
    Merge ratings with movie metadata and clean up.
    This is the input for our SVD baseline model.
    """
    print("[3/5] Building explicit ratings dataset...")

    df = ratings.merge(movies, on="movieId", how="left")

    df["year"] = df["title"].str.extract(r"\((\d{4})\)").astype(float)

    df["genre_list"] = df["genres"].str.split("|")

    df["rated_at"] = pd.to_datetime(df["timestamp"], unit="s")

    explicit = df[[
        "userId", "movieId", "rating",
        "title", "genres", "genre_list", "year", "rated_at"
    ]].copy()

    print(f"    Explicit dataset: {len(explicit):,} rows")
    print(f"    Average rating: {explicit['rating'].mean():.2f} stars")

    return explicit
def build_implicit(explicit):
    """
    Convert explicit star ratings into implicit confidence scores.
 
    The key idea: instead of 'what did the user rate this?', we ask
    'how confident are we that this user genuinely likes this content?'
 
    confidence = 1 + ALPHA * completion_rate
 
    A confidence of 1.0 means 'no signal'.
    A confidence of 41.0 means 'extremely strong positive signal'.
 
    This is the Hu, Koren & Volinsky (2008) formulation used by Spotify.
    """
    print("[4/5] Converting to implicit confidence scores...")
 
    implicit = explicit.copy()
 
    # Map each rating to an estimated completion rate
    implicit["completion_rate"] = implicit["rating"].map(RATING_TO_COMPLETION)
 
    # Handle any ratings not in our map using linear interpolation
    missing = implicit["completion_rate"].isna()
    if missing.any():
        implicit.loc[missing, "completion_rate"] = (
            implicit.loc[missing, "rating"] - 1
        ) / 4 * 0.85 + 0.10
 
    # Confidence score
    implicit["confidence"] = 1 + ALPHA * implicit["completion_rate"]
 
    # Binary preference: did the user engage meaningfully?
    implicit["preference"] = (implicit["rating"] >= 3.0).astype(int)
 
    print(f"    Implicit dataset: {len(implicit):,} rows")
    print(f"    Confidence range: {implicit['confidence'].min():.1f} "
          f"to {implicit['confidence'].max():.1f}")
    print(f"    Positive preference rate: {implicit['preference'].mean():.1%}")
 
    summary = implicit.groupby("rating").agg(
        completion_rate=("completion_rate", "mean"),
        confidence=("confidence", "mean"),
        count=("userId", "count")
    ).round(2)
    print(f"\n    Rating to implicit signal mapping:\n{summary}")
 
    return implicit
 
 
# ── Step 5: Build movie content features ──────────────────────────────────────
 
def build_movie_features(movies):
    """
    One-hot encode genres and normalise year.
    Used by the LightFM hybrid model in Stage 3.
    """
    print("[5/5] Building movie content features...")
 
    all_genres = set()
    for genres in movies["genres"].str.split("|"):
        all_genres.update(genres)
    all_genres.discard("(no genres listed)")
    all_genres = sorted(all_genres)
 
    for genre in all_genres:
        movies[f"genre_{genre}"] = movies["genres"].str.contains(
            genre, regex=False
        ).astype(int)
 
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)
    year_min = movies["year"].min()
    year_max = movies["year"].max()
    movies["year_norm"] = (movies["year"] - year_min) / (year_max - year_min)
    movies["year_norm"] = movies["year_norm"].fillna(0.5)
 
    genre_cols = [f"genre_{g}" for g in all_genres]
    print(f"    {len(all_genres)} genres + year = "
          f"{len(genre_cols) + 1} feature dimensions per movie")
 
    return movies, genre_cols
 
 
# ── Save ───────────────────────────────────────────────────────────────────────
 
def save_processed(explicit, implicit, movies, genre_cols):
    """Save all processed data to disk for use by train_models.py."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
 
    explicit.to_pickle(PROCESSED_DIR / "explicit_ratings.pkl")
    implicit.to_pickle(PROCESSED_DIR / "implicit_signals.pkl")
    movies.to_pickle(PROCESSED_DIR / "movies_features.pkl")
 
    import json
    with open(PROCESSED_DIR / "genre_cols.json", "w") as f:
        json.dump(genre_cols, f)
 
    print(f"\n    Saved to {PROCESSED_DIR}/")
    print(f"    explicit_ratings.pkl  — {len(explicit):,} rows")
    print(f"    implicit_signals.pkl  — {len(implicit):,} rows")
    print(f"    movies_features.pkl   — {len(movies):,} movies")
    print(f"    genre_cols.json       — {len(genre_cols)} genre features")
 
 
# ── Main ───────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    print("=" * 60)
    print("Movie Recommender — Data Pipeline")
    print("=" * 60)
 
    download_movielens()
    ratings, movies = load_raw_data()
    explicit = build_explicit(ratings, movies)
    implicit = build_implicit(explicit)
    movies, genre_cols = build_movie_features(movies)
    save_processed(explicit, implicit, movies, genre_cols)
 
    print("\n" + "=" * 60)
    print("Pipeline complete! Next step: run train_models.py")
    print("=" * 60)
 