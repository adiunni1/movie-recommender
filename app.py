"""
app.py
------
Streamlit live demo for the Movie Recommendation System.
Fixed to use item-embedding similarity for all three models,
so results actually change based on what movie you search for.

Run:
    streamlit run app.py
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

MODELS_DIR    = Path("models")
PROCESSED_DIR = Path("data/processed")

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)


@st.cache_resource
def load_everything():
    with open(MODELS_DIR / "id_maps.pkl", "rb") as f:
        maps = pickle.load(f)
    user2idx  = maps["user2idx"]
    movie2idx = maps["movie2idx"]
    idx2movie = maps["idx2movie"]

    with open(MODELS_DIR / "lightfm_model.pkl", "rb") as f:
        lfm_model = pickle.load(f)
    with open(MODELS_DIR / "lightfm_item_features.pkl", "rb") as f:
        item_features = pickle.load(f)
    with open(MODELS_DIR / "als_model.pkl", "rb") as f:
        als_model = pickle.load(f)
    with open(MODELS_DIR / "svd_model.pkl", "rb") as f:
        svd_model = pickle.load(f)

    movies  = pd.read_pickle(PROCESSED_DIR / "movies_features.pkl")
    ratings = pd.read_pickle(PROCESSED_DIR / "explicit_ratings.pkl")

    with open(PROCESSED_DIR / "genre_cols.json") as f:
        genre_cols = json.load(f)

    return (lfm_model, item_features, als_model, svd_model,
            movies, genre_cols, ratings,
            user2idx, movie2idx, idx2movie)


def find_movie(query, movies):
    mask = movies["title"].str.contains(query, case=False, na=False)
    return movies[mask][["movieId", "title", "genres"]].drop_duplicates("movieId")


def get_genres(row, genre_cols):
    genres = [col.replace("genre_", "") for col in genre_cols
              if row.get(col, 0) == 1]
    return ", ".join(genres) if genres else row.get("genres", "Unknown")


# ── LightFM: item embedding cosine similarity ─────────────────────────────────
def recommend_lightfm(seed_id, lfm_model, item_features,
                       movies, genre_cols, movie2idx, idx2movie, n=10):
    """
    Get item representations (latent vectors) from the trained LightFM model,
    then find the movies whose vectors are most similar to the seed movie.
    This is proper item-to-item similarity using learned genre + interaction features.
    """
    if seed_id not in movie2idx:
        return pd.DataFrame()

    seed_idx = movie2idx[seed_id]

    # Get the latent embedding for every movie
    # item_biases: shape (n_movies,)  item_embeddings: shape (n_movies, n_factors)
    _, item_embeddings = lfm_model.get_item_representations(features=item_features)

    seed_vec = item_embeddings[seed_idx].reshape(1, -1)
    sims     = cosine_similarity(seed_vec, item_embeddings)[0]
    sims[seed_idx] = -999  # exclude the seed itself

    top_idx      = np.argsort(-sims)[:n]
    top_movie_ids = [idx2movie[i] for i in top_idx if i in idx2movie]

    results = movies[movies["movieId"].isin(top_movie_ids)].copy()
    score_map = {idx2movie[i]: float(sims[i]) for i in top_idx if i in idx2movie}
    results["score"]       = results["movieId"].map(score_map)
    results["genres_clean"] = results.apply(lambda r: get_genres(r, genre_cols), axis=1)
    results = results.sort_values("score", ascending=False)
    return results[["title", "genres_clean", "year", "score"]].head(n)


# ── ALS: built-in item similarity ────────────────────────────────────────────
def recommend_als(seed_id, als_model, movies, movie2idx, idx2movie, n=10):
    """
    ALS learns a latent vector for every item. similar_items() finds the items
    whose vectors have the highest dot-product similarity with the seed item.
    """
    if seed_id not in movie2idx:
        return pd.DataFrame()

    item_idx = movie2idx[seed_id]

    try:
        result = als_model.similar_items(item_idx, N=n + 5)
        indices = result[0] if isinstance(result, tuple) else result

        recs = []
        for idx in indices:
            mid = idx2movie.get(int(idx))
            if mid and mid != seed_id:
                recs.append(mid)
            if len(recs) >= n:
                break

        if not recs:
            return pd.DataFrame()

        results = movies[movies["movieId"].isin(recs)][["title", "genres", "year"]].copy()
        return results.head(n)

    except Exception:
        return pd.DataFrame()


# ── SVD: item latent factor cosine similarity ────────────────────────────────
def recommend_svd(seed_id, svd_model, movies, idx2movie, n=10):
    """
    SVD learns a latent factor vector (qi) for every item in its trainset.
    We compute cosine similarity between the seed item's qi and all other qi vectors
    to find the most similar movies — much better than the popularity fallback.
    """
    trainset = svd_model.trainset

    # Map movieId → Surprise inner id
    try:
        seed_iid = trainset.to_inner_iid(str(seed_id))
    except ValueError:
        try:
            seed_iid = trainset.to_inner_iid(seed_id)
        except Exception:
            return pd.DataFrame()

    seed_vec = svd_model.qi[seed_iid].reshape(1, -1)
    sims     = cosine_similarity(seed_vec, svd_model.qi)[0]
    sims[seed_iid] = -999

    top_inner = np.argsort(-sims)[:n + 5]

    recs = []
    for iid in top_inner:
        try:
            raw_id   = trainset.to_raw_iid(iid)
            movie_id = int(raw_id)
            if movie_id != seed_id:
                recs.append(movie_id)
        except Exception:
            continue
        if len(recs) >= n:
            break

    if not recs:
        return pd.DataFrame()

    results = movies[movies["movieId"].isin(recs)][["title", "genres", "year"]].copy()
    return results.head(n)


# ── App layout ────────────────────────────────────────────────────────────────
def main():
    st.title("🎬 Movie Recommendation System")
    st.markdown(
        "**Comparing three recommendation approaches** — SVD on explicit ratings, "
        "ALS on implicit behavioral signals, and LightFM hybrid (our winner). "
        "All three use **item embedding similarity**: each model learns a latent "
        "vector per movie, and recommendations are the movies whose vectors are "
        "most similar to the one you searched for."
    )
    st.markdown("---")

    with st.spinner("Loading models..."):
        (lfm_model, item_features, als_model, svd_model,
         movies, genre_cols, ratings,
         user2idx, movie2idx, idx2movie) = load_everything()

    # Experiment results panel
    with st.expander("📊 Model Performance — The Experiment Results", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Stage 1 — SVD", "RMSE: 0.8895",
                      help="Avg star prediction error. Lower = better.")
            st.caption("Explicit star ratings only")
        with col2:
            st.metric("Stage 2 — ALS", "Precision@10: 0.71%",
                      help="% of top-10 recs user actually liked.")
            st.caption("Implicit confidence scores (Spotify's algorithm)")
        with col3:
            st.metric("Stage 3 — LightFM ✓", "Precision@10: 1.31%",
                      delta="+85% vs ALS")
            st.caption("Implicit + genre content — our winner")
        st.info(
            "**Hypothesis proved:** Adding genre content features (LightFM) "
            "produces 85% better recommendations than implicit signals alone (ALS)."
        )

    st.markdown("---")
    st.subheader("🔍 Find a movie you enjoyed")

    query   = st.text_input("Type a movie title (partial match works):",
                             placeholder="e.g. Toy Story, Matrix, Inception...")
    seed_id = None

    if query:
        results = find_movie(query, movies)
        if results.empty:
            st.warning(f"No movies found matching '{query}'. Try a different title.")
        else:
            selected = st.selectbox(
                "Select the movie you meant:",
                options=results["movieId"].tolist(),
                format_func=lambda mid: (
                    results[results["movieId"] == mid]["title"].values[0]
                    + "  —  "
                    + results[results["movieId"] == mid]["genres"].values[0]
                )
            )
            seed_id = selected
            row = results[results["movieId"] == selected].iloc[0]
            st.success(f"Selected: **{row['title']}** | {row['genres']}")

    if seed_id:
        st.markdown("---")
        st.subheader("🎯 Recommendations from each model")
        st.caption(
            "Same seed movie, three different algorithms. "
            "Each model learned a unique latent vector per movie — "
            "recommendations are the closest neighbours in that vector space."
        )

        tab1, tab2, tab3 = st.tabs([
            "🥇 LightFM Hybrid (Winner)",
            "🥈 ALS Implicit",
            "🥉 SVD Explicit"
        ])

        with tab1:
            st.markdown("**Genre-aware + behavioral signals. Our best model.**")
            with st.spinner("Generating LightFM recommendations..."):
                recs = recommend_lightfm(
                    seed_id, lfm_model, item_features,
                    movies, genre_cols, movie2idx, idx2movie
                )
            if recs.empty:
                st.warning("Could not generate recommendations for this movie.")
            else:
                for i, (_, r) in enumerate(recs.iterrows(), 1):
                    yr = int(r["year"]) if pd.notna(r.get("year")) else "?"
                    st.write(f"**{i}. {r['title']}** ({yr})")
                    st.caption(f"Genres: {r['genres_clean']}")

        with tab2:
            st.markdown("**Implicit behavioral confidence scores. Spotify's algorithm.**")
            with st.spinner("Generating ALS recommendations..."):
                recs = recommend_als(seed_id, als_model, movies, movie2idx, idx2movie)
            if recs.empty:
                st.warning("Could not generate recommendations for this movie.")
            else:
                for i, (_, r) in enumerate(recs.iterrows(), 1):
                    yr = int(r["year"]) if pd.notna(r.get("year")) else "?"
                    st.write(f"**{i}. {r['title']}** ({yr})")
                    st.caption(f"Genres: {r['genres']}")

        with tab3:
            st.markdown("**Collaborative filtering on star ratings. The 2006-era baseline.**")
            with st.spinner("Generating SVD recommendations..."):
                recs = recommend_svd(seed_id, svd_model, movies, idx2movie)
            if recs.empty:
                st.warning("Could not generate recommendations for this movie.")
            else:
                for i, (_, r) in enumerate(recs.iterrows(), 1):
                    yr = int(r["year"]) if pd.notna(r.get("year")) else "?"
                    st.write(f"**{i}. {r['title']}** ({yr})")
                    st.caption(f"Genres: {r['genres']}")

    st.markdown("---")
    st.caption(
        "Built by Advait Unni · MovieLens 100K dataset · "
        "Models: scikit-surprise (SVD), implicit (ALS), LightFM · "
        "github.com/adiunni1/movie-recommender"
    )


if __name__ == "__main__":
    main()