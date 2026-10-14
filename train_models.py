"""
train_models.py
---------------
Trains three recommendation models on the processed MovieLens data.

Stage 1 — SVD on explicit star ratings       (the baseline everyone uses)
Stage 2 — ALS on implicit confidence scores  (Spotify's Discover Weekly algorithm)
Stage 3 — LightFM hybrid: implicit + genre   (content-aware, most sophisticated)

The experiment: do Stages 2 or 3 beat Stage 1?
If yes, we have proved the core hypothesis of this project.

Run:
    python train_models.py
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, lil_matrix

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")

# ── Hyperparameters ────────────────────────────────────────────────────────────
# These control model complexity. Larger = more expressive but slower to train.
N_FACTORS    = 50   # number of latent dimensions (hidden taste features)
N_ITERATIONS = 20   # ALS training iterations
N_EPOCHS     = 30   # LightFM training epochs
K            = 10   # evaluate top-K recommendations
TEST_RATIO   = 0.2  # hold out the most recent 20% of each user's ratings
RANDOM_STATE = 42


# ── Step 1: Load processed data ────────────────────────────────────────────────

def load_data():
    """Load the four files saved by data_pipeline.py."""
    print("Loading processed data...")

    explicit  = pd.read_pickle(PROCESSED_DIR / "explicit_ratings.pkl")
    implicit  = pd.read_pickle(PROCESSED_DIR / "implicit_signals.pkl")
    movies    = pd.read_pickle(PROCESSED_DIR / "movies_features.pkl")

    with open(PROCESSED_DIR / "genre_cols.json") as f:
        genre_cols = json.load(f)

    print(f"  Ratings:  {len(explicit):,}")
    print(f"  Movies:   {len(movies):,} with {len(genre_cols)} genre features")
    return explicit, implicit, movies, genre_cols


# ── Step 2: Chronological train/test split ────────────────────────────────────

def chronological_split(df, test_ratio=TEST_RATIO):
    """
    Split ratings into train and test sets using TIME — not random shuffling.

    Why chronological? If you shuffle and split randomly, the model can
    'see the future' during training — this is called data leakage.
    It makes the model look better than it really is.

    Strategy: sort each user's ratings by date, hold out the MOST RECENT
    ones as the test set. Train on the past, evaluate on the future.
    """
    df = df.sort_values("rated_at")

    train_rows, test_rows = [], []

    for user_id, user_df in df.groupby("userId"):
        n_test = max(1, int(len(user_df) * test_ratio))
        train_rows.append(user_df.iloc[:-n_test])   # older ratings → train
        test_rows.append(user_df.iloc[-n_test:])    # newest ratings → test

    train = pd.concat(train_rows).reset_index(drop=True)
    test  = pd.concat(test_rows).reset_index(drop=True)

    print(f"  Train: {len(train):,} ratings | Test: {len(test):,} ratings")
    return train, test


# ── Step 3: Build ID maps ─────────────────────────────────────────────────────

def build_id_maps(df):
    """
    Create mappings from original IDs to consecutive integers starting at 0.

    The ALS and LightFM libraries require 0-based integer indices.
    MovieLens userId values jump around (1, 5, 12, ...) so we map them
    to a clean sequence (0, 1, 2, ...).

    Example: userId [1, 5, 12] → user2idx = {1:0, 5:1, 12:2}
    """
    user_ids  = sorted(df["userId"].unique())
    movie_ids = sorted(df["movieId"].unique())

    user2idx  = {uid: i for i, uid in enumerate(user_ids)}
    movie2idx = {mid: i for i, mid in enumerate(movie_ids)}
    idx2movie = {i: mid for mid, i in movie2idx.items()}

    print(f"  {len(user2idx):,} users | {len(movie2idx):,} movies")
    return user2idx, movie2idx, idx2movie


# ── Step 4: Build sparse matrix ───────────────────────────────────────────────

def build_sparse_matrix(df, user2idx, movie2idx, value_col="confidence"):
    """
    Build a compressed sparse row (CSR) matrix of shape (n_users, n_movies).

    Why sparse? With 610 users and 9,724 movies, a full dense matrix would
    have 5.9 million cells. But we only have 100K actual ratings — 98% of
    cells would be zero. A sparse matrix only stores the non-zero values.

    value_col: what to put in each cell — "confidence" for ALS, "preference"
               for LightFM, "rating" for anything needing raw stars.
    """
    # Drop rows where the user or movie isn't in our index
    df = df[df["userId"].isin(user2idx) & df["movieId"].isin(movie2idx)].copy()

    rows = df["userId"].map(user2idx).values
    cols = df["movieId"].map(movie2idx).values
    vals = df[value_col].astype(float).values

    shape = (len(user2idx), len(movie2idx))
    return csr_matrix((vals, (rows, cols)), shape=shape)


# ── STAGE 1: SVD on explicit ratings ──────────────────────────────────────────

def train_svd(train, test):
    """
    SVD (Singular Value Decomposition) collaborative filtering.

    What it does: decomposes the user-movie ratings matrix into two smaller
    matrices. One captures 'user taste profiles', one captures 'movie style
    profiles'. The dot product of those two matrices predicts any rating.

    Trained on:  raw star ratings (1.0 to 5.0)
    Predicts:    star ratings for unseen movies
    Evaluated:   RMSE — average prediction error in stars (lower = better)

    This is the approach Netflix used circa 2006. It is our baseline.
    """
    print("\n" + "=" * 60)
    print("STAGE 1 — SVD on Explicit Ratings")
    print("=" * 60)

    from surprise import Dataset, Reader, SVD
    from surprise import accuracy as surp_accuracy

    reader = Reader(rating_scale=(0.5, 5.0))

    # Surprise needs its own Dataset object
    train_data = Dataset.load_from_df(
        train[["userId", "movieId", "rating"]], reader
    )
    trainset = train_data.build_full_trainset()

    test_data = Dataset.load_from_df(
        test[["userId", "movieId", "rating"]], reader
    )
    testset = test_data.build_full_trainset().build_testset()

    print(f"Training SVD (n_factors={N_FACTORS}, n_epochs={N_EPOCHS})...")
    model = SVD(
        n_factors=N_FACTORS,
        n_epochs=N_EPOCHS,
        random_state=RANDOM_STATE,
        verbose=False
    )
    model.fit(trainset)

    predictions = model.test(testset)
    rmse = surp_accuracy.rmse(predictions, verbose=False)
    mae  = surp_accuracy.mae(predictions,  verbose=False)

    print(f"  RMSE: {rmse:.4f}  (average prediction error in stars)")
    print(f"  MAE:  {mae:.4f}")

    return model, {"rmse": rmse, "mae": mae}


# ── STAGE 2: ALS on implicit confidence scores ────────────────────────────────

def evaluate_als_precision(model, train_matrix, test_df,
                            user2idx, movie2idx, idx2movie, k=K):
    """
    Precision@K for ALS: of the top-K movies recommended to a user,
    what fraction did they actually rate positively in the test set?

    A random recommender on this dataset would score about 0.01 (1%).
    A good model should score 0.05-0.15 (5-15%).
    """
    liked_in_test = (
        test_df[test_df["preference"] == 1]
        .groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    precisions = []
    sample_users = list(liked_in_test.keys())[:100]  # sample for speed

    for user_id in sample_users:
        if user_id not in user2idx:
            continue
        user_idx = user2idx[user_id]
        liked    = liked_in_test[user_id]

        try:
            result = model.recommend(
                user_idx,
                train_matrix[user_idx],
                N=k,
                filter_already_liked_items=True
            )
            # newer versions of implicit return (ids, scores) tuple
            if isinstance(result, tuple):
                rec_indices = result[0]
            else:
                rec_indices = result

            rec_movies = {idx2movie[i] for i in rec_indices if i in idx2movie}
            precisions.append(len(rec_movies & liked) / k)
        except Exception:
            continue

    return float(np.mean(precisions)) if precisions else 0.0


def train_als(train, test, user2idx, movie2idx, idx2movie):
    """
    ALS (Alternating Least Squares) on implicit confidence scores.

    What it does: same matrix factorisation idea as SVD, but instead of
    minimising star-rating prediction error, it maximises the likelihood
    that high-confidence items (things the user engaged with a lot) appear
    in the top of the recommendations list.

    Trained on:  confidence scores (1 + 40 × completion_rate)
    Predicts:    a ranked list of movies the user will engage with
    Evaluated:   Precision@10 (higher = better)

    This is the algorithm behind Spotify's Discover Weekly.
    """
    print("\n" + "=" * 60)
    print("STAGE 2 — ALS on Implicit Confidence Scores")
    print("=" * 60)

    import implicit

    user_item = build_sparse_matrix(train, user2idx, movie2idx, "confidence")
    item_user = user_item.T.tocsr()  # implicit expects (items × users)

    print(f"  Matrix: {user_item.shape[0]} users × {user_item.shape[1]} movies")
    density = 100 * user_item.nnz / (user_item.shape[0] * user_item.shape[1])
    print(f"  Density: {density:.2f}% filled (rest are zeros)")
    print(f"Training ALS (factors={N_FACTORS}, iterations={N_ITERATIONS})...")

    model = implicit.als.AlternatingLeastSquares(
        factors=N_FACTORS,
        iterations=N_ITERATIONS,
        regularization=0.1,
        random_state=RANDOM_STATE
    )
    model.fit(item_user)

    precision = evaluate_als_precision(
        model, user_item, test, user2idx, movie2idx, idx2movie, k=K
    )

    print(f"  Precision@{K}: {precision:.4f}  (% of top-{K} recs user actually liked)")

    return model, {"precision_at_k": precision, "user_item": user_item}


# ── STAGE 3: LightFM Hybrid ───────────────────────────────────────────────────

def build_item_features(movies, genre_cols, movie2idx):
    """
    Build the item feature matrix: one row per movie, one column per genre.

    Each cell = 1.0 if the movie has that genre, 0.0 if not.
    Plus a normalised year column.

    This is what makes LightFM 'hybrid' — it learns from both
    who watched what AND what the movie is actually about.
    """
    n_movies   = len(movie2idx)
    n_features = len(genre_cols) + 1   # genres + year

    feat = lil_matrix((n_movies, n_features))

    for _, row in movies.iterrows():
        if row["movieId"] not in movie2idx:
            continue
        idx = movie2idx[row["movieId"]]
        for j, col in enumerate(genre_cols):
            feat[idx, j] = float(row.get(col, 0))
        feat[idx, len(genre_cols)] = float(row.get("year_norm", 0.5))

    return feat.tocsr()


def train_lightfm(train, test, movies, genre_cols, user2idx, movie2idx, idx2movie):
    """
    LightFM hybrid model.

    What it does: combines collaborative filtering (who watched what)
    with content-based filtering (what genre is this movie?).

    Loss = WARP (Weighted Approximate-Rank Pairwise): unlike RMSE which
    penalises star prediction error, WARP directly optimises for getting
    liked items into the top of the ranked list.

    Trained on:  binary preference (liked/disliked) + genre features
    Predicts:    a ranked list of movies
    Evaluated:   Precision@10 and Recall@10
    """
    print("\n" + "=" * 60)
    print("STAGE 3 — LightFM Hybrid (Implicit + Genre Content)")
    print("=" * 60)

    from lightfm import LightFM
    from lightfm.evaluation import precision_at_k, recall_at_k

    # Only positive interactions for training
    train_pos = train[train["preference"] == 1]
    test_pos  = test[test["preference"] == 1]

    interactions      = build_sparse_matrix(train_pos, user2idx, movie2idx, "confidence")
    test_interactions = build_sparse_matrix(test_pos,  user2idx, movie2idx, "confidence")
    item_features     = build_item_features(movies, genre_cols, movie2idx)

    print(f"  Interactions:  {interactions.shape}")
    print(f"  Item features: {item_features.shape}")
    print(f"Training LightFM (loss=warp, epochs={N_EPOCHS})...")

    model = LightFM(
        loss="warp",
        no_components=N_FACTORS,
        random_state=RANDOM_STATE
    )
    model.fit(
        interactions,
        item_features=item_features,
        epochs=N_EPOCHS,
        num_threads=2,
        verbose=False
    )

    p_at_k = precision_at_k(
        model, test_interactions,
        train_interactions=interactions,
        item_features=item_features,
        k=K
    ).mean()

    r_at_k = recall_at_k(
        model, test_interactions,
        train_interactions=interactions,
        item_features=item_features,
        k=K
    ).mean()

    print(f"  Precision@{K}: {p_at_k:.4f}")
    print(f"  Recall@{K}:    {r_at_k:.4f}")

    return model, {
        "precision_at_k": float(p_at_k),
        "recall_at_k":    float(r_at_k),
        "item_features":  item_features
    }


# ── Save all models ────────────────────────────────────────────────────────────

def save_models(svd_model, als_model, als_extras,
                lfm_model, lfm_extras,
                user2idx, movie2idx, idx2movie):
    """Save trained models and ID maps to the models/ directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODELS_DIR / "svd_model.pkl",  "wb") as f:
        pickle.dump(svd_model, f)

    with open(MODELS_DIR / "als_model.pkl",  "wb") as f:
        pickle.dump(als_model, f)

    with open(MODELS_DIR / "als_user_item.pkl", "wb") as f:
        pickle.dump(als_extras["user_item"], f)

    with open(MODELS_DIR / "lightfm_model.pkl", "wb") as f:
        pickle.dump(lfm_model, f)

    with open(MODELS_DIR / "lightfm_item_features.pkl", "wb") as f:
        pickle.dump(lfm_extras["item_features"], f)

    with open(MODELS_DIR / "id_maps.pkl", "wb") as f:
        pickle.dump({
            "user2idx":  user2idx,
            "movie2idx": movie2idx,
            "idx2movie": idx2movie
        }, f)

    print(f"\nAll models saved to {MODELS_DIR}/")
    for p in sorted(MODELS_DIR.iterdir()):
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:<35} {size_kb:>7.1f} KB")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Movie Recommender — Model Training")
    print("=" * 60)

    # Load processed data
    explicit, implicit_df, movies, genre_cols = load_data()

    # Chronological split
    print("\nSplitting data chronologically...")
    train, test = chronological_split(explicit)

    # Build ID maps
    print("\nBuilding ID maps...")
    user2idx, movie2idx, idx2movie = build_id_maps(explicit)

    # Merge implicit signals into train/test
    sig_cols = ["userId", "movieId", "confidence", "completion_rate", "preference"]
    implicit_df = implicit_df[sig_cols]
    train = train.merge(implicit_df, on=["userId", "movieId"], how="left")
    test  = test.merge(implicit_df,  on=["userId", "movieId"], how="left")
    train["preference"] = train["preference"].fillna(0).astype(int)
    test["preference"]  = test["preference"].fillna(0).astype(int)
    train["confidence"] = train["confidence"].fillna(1.0)
    test["confidence"]  = test["confidence"].fillna(1.0)

    # Train all three models
    svd_model, svd_metrics = train_svd(train, test)
    als_model, als_metrics = train_als(train, test, user2idx, movie2idx, idx2movie)
    lfm_model, lfm_metrics = train_lightfm(
        train, test, movies, genre_cols, user2idx, movie2idx, idx2movie
    )

    # Print results comparison
    print("\n" + "=" * 60)
    print("RESULTS — WHICH MODEL WINS?")
    print("=" * 60)
    print(f"\nStage 1 — SVD (explicit star ratings):")
    print(f"  RMSE:          {svd_metrics['rmse']:.4f}  ← avg error in stars (lower better)")
    print(f"  MAE:           {svd_metrics['mae']:.4f}")
    print(f"\nStage 2 — ALS (implicit confidence):")
    print(f"  Precision@10:  {als_metrics['precision_at_k']:.4f}  ← % of recs user liked (higher better)")
    print(f"\nStage 3 — LightFM hybrid:")
    print(f"  Precision@10:  {lfm_metrics['precision_at_k']:.4f}")
    print(f"  Recall@10:     {lfm_metrics['recall_at_k']:.4f}")

    if lfm_metrics["precision_at_k"] >= als_metrics["precision_at_k"]:
        winner = "LightFM hybrid"
    else:
        winner = "ALS implicit"
    print(f"\n→  Best recommendation model: {winner}")
    print(f"→  Hypothesis result: implicit signals {'beat' if als_metrics['precision_at_k'] > 0.01 else 'vs'} explicit ratings")

    # Save
    save_models(
        svd_model, als_model, als_metrics,
        lfm_model, lfm_metrics,
        user2idx, movie2idx, idx2movie
    )

    print("\n" + "=" * 60)
    print("Training complete! Next step: python app.py")
    print("=" * 60)