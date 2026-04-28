"""
train_and_save_optimized.py
===========================

One-time offline training script.

This version avoids saving:
- pred_df
- content_similarity
- full df
- full train_data

Instead it saves compact inference artifacts:
- SVD user/item factors
- biases
- normalized content embeddings
- user history arrays
- movie metadata
"""

from __future__ import annotations

import os
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")
np.random.seed(42)

# ───────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────

RATING_MIN, RATING_MAX = 1.0, 5.0

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

MAX_SVD_COMPONENTS = 600
MAX_TFIDF_FEATURES = 5000
MAX_CONTENT_COMPONENTS = 100

SAVE_TEXT_TRANSFORMERS = False
# Set this to True only if your deployed backend needs to transform brand-new movie text.
# For recommendations among existing movies, you do NOT need tfidf/content_svd at inference.


def save_joblib(obj, filename: str, compress: int = 3):
    path = os.path.join(ARTIFACTS_DIR, filename)
    joblib.dump(obj, path, compress=compress)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  → {filename}: {size_mb:.2f} MB")


def clip_rating(x: float) -> float:
    return float(np.clip(x, RATING_MIN, RATING_MAX))


# ═══════════════════════════════════════════════════════════════
# 1. LOAD & PREPROCESS DATA
# ═══════════════════════════════════════════════════════════════

print("Loading data...")

df = pd.read_csv("backend/final_dataset.csv")

df = df.dropna(subset=["userid", "movieid", "rating"])
df["userid"] = df["userid"].astype(int)
df["movieid"] = df["movieid"].astype(int)
df["rating"] = df["rating"].clip(RATING_MIN, RATING_MAX)

if "clean_title" not in df.columns:
    raise ValueError("Expected column 'clean_title' not found.")

df["clean_title"] = df["clean_title"].fillna("").astype(str).str.lower().str.strip()

text_cols = ["overview", "keywords", "genres", "tagline", "language_name"]
for col in text_cols:
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].fillna("").astype(str)

df["content"] = (
    df["overview"] + " " +
    (df["genres"] + " ") * 3 +
    (df["keywords"] + " ") * 2 +
    df["tagline"] + " " +
    df["language_name"]
)

print(f"Data shape: {df.shape}")
print(f"Unique users: {df['userid'].nunique()} | Unique movies: {df['movieid'].nunique()}")


# ═══════════════════════════════════════════════════════════════
# 2. TRAIN / VAL / TEST SPLIT
# ═══════════════════════════════════════════════════════════════

ratings = df[["userid", "movieid", "rating"]].copy().reset_index(drop=True)

train_data, temp_data = train_test_split(
    ratings,
    test_size=0.30,
    random_state=42,
)

val_data, test_data = train_test_split(
    temp_data,
    test_size=0.50,
    random_state=42,
)

train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")


# ═══════════════════════════════════════════════════════════════
# 3. TRAINING-SET STATISTICS
# ═══════════════════════════════════════════════════════════════

global_mean = float(train_data["rating"].mean())

user_bias_series = train_data.groupby("userid")["rating"].mean() - global_mean
item_bias_series = train_data.groupby("movieid")["rating"].mean() - global_mean

movie_mean_series = train_data.groupby("movieid")["rating"].mean()
user_mean_series = train_data.groupby("userid")["rating"].mean()

print(f"Global mean: {global_mean:.4f}")


# ═══════════════════════════════════════════════════════════════
# 4. SVD COLLABORATIVE FILTERING — COMPACT VERSION
# ═══════════════════════════════════════════════════════════════

print("Training compact SVD model...")

train_data_norm = train_data.copy()

train_data_norm["user_bias"] = train_data_norm["userid"].map(user_bias_series).fillna(0.0)
train_data_norm["item_bias"] = train_data_norm["movieid"].map(item_bias_series).fillna(0.0)

train_data_norm["rating_norm"] = (
    train_data_norm["rating"]
    - global_mean
    - train_data_norm["user_bias"]
    - train_data_norm["item_bias"]
)

# Important:
# pivot_table averages duplicate user/movie pairs.
# To preserve that behavior, we also average duplicates before creating sparse matrix.
matrix_data = (
    train_data_norm
    .groupby(["userid", "movieid"], as_index=False)["rating_norm"]
    .mean()
)

svd_user_ids = np.sort(matrix_data["userid"].unique()).astype(np.int64)
svd_movie_ids = np.sort(matrix_data["movieid"].unique()).astype(np.int64)

user_to_svd_idx = {int(uid): idx for idx, uid in enumerate(svd_user_ids)}
movie_to_svd_idx = {int(mid): idx for idx, mid in enumerate(svd_movie_ids)}

rows = matrix_data["userid"].map(user_to_svd_idx).to_numpy(dtype=np.int32)
cols = matrix_data["movieid"].map(movie_to_svd_idx).to_numpy(dtype=np.int32)
vals = matrix_data["rating_norm"].to_numpy(dtype=np.float32)

train_matrix = csr_matrix(
    (vals, (rows, cols)),
    shape=(len(svd_user_ids), len(svd_movie_ids)),
    dtype=np.float32,
)

n_svd_components = min(MAX_SVD_COMPONENTS, min(train_matrix.shape) - 1)

if n_svd_components < 1:
    raise ValueError("Not enough users/movies to train SVD.")

svd_model = TruncatedSVD(
    n_components=n_svd_components,
    random_state=42,
)

user_factors = svd_model.fit_transform(train_matrix).astype(np.float32)
item_factors = svd_model.components_.T.astype(np.float32)

user_bias = (
    user_bias_series
    .reindex(svd_user_ids)
    .fillna(0.0)
    .to_numpy(dtype=np.float32)
)

item_bias = (
    item_bias_series
    .reindex(svd_movie_ids)
    .fillna(0.0)
    .to_numpy(dtype=np.float32)
)

print(f"SVD fitted — {n_svd_components} components")
print(f"Explained variance: {svd_model.explained_variance_ratio_.sum():.2%}")


def svd_predict_train(uid: int, mid: int) -> float:
    ui = user_to_svd_idx.get(int(uid))
    mi = movie_to_svd_idx.get(int(mid))

    if ui is None or mi is None:
        return float(global_mean)

    latent = float(user_factors[ui] @ item_factors[mi])
    pred = global_mean + float(user_bias[ui]) + float(item_bias[mi]) + latent

    return clip_rating(pred)


# ═══════════════════════════════════════════════════════════════
# 5. CONTENT-BASED FILTERING — COMPACT VERSION
# ═══════════════════════════════════════════════════════════════

print("Training compact content model...")

movie_content = df.drop_duplicates(subset="movieid").reset_index(drop=True).copy()

content_movie_ids = movie_content["movieid"].to_numpy(dtype=np.int64)
movieid_to_content_idx = {
    int(mid): idx for idx, mid in enumerate(content_movie_ids)
}

tfidf = TfidfVectorizer(
    max_features=MAX_TFIDF_FEATURES,
    stop_words="english",
    min_df=2,
    max_df=0.85,
    ngram_range=(1, 2),
)

tfidf_matrix = tfidf.fit_transform(movie_content["content"])

n_content_components = min(
    MAX_CONTENT_COMPONENTS,
    min(tfidf_matrix.shape) - 1,
)

if n_content_components < 1:
    raise ValueError("Not enough content features to train content SVD.")

content_svd = TruncatedSVD(
    n_components=n_content_components,
    random_state=42,
)

reduced_matrix = content_svd.fit_transform(tfidf_matrix).astype(np.float32)

# This replaces content_similarity.
# Cosine similarity between two movies now becomes:
# content_embeddings[i] @ content_embeddings[j]
content_embeddings = normalize(
    reduced_matrix,
    norm="l2",
    axis=1,
).astype(np.float32)

print(f"Content embeddings: {content_embeddings.shape}")


# ═══════════════════════════════════════════════════════════════
# 6. COMPACT USER HISTORY
# ═══════════════════════════════════════════════════════════════

print("Building compact user history...")

history_df = train_data[["userid", "movieid", "rating"]].copy()

history_df["content_idx"] = (
    history_df["movieid"]
    .map(movieid_to_content_idx)
    .fillna(-1)
    .astype(np.int32)
)

history_df["svd_movie_idx"] = (
    history_df["movieid"]
    .map(movie_to_svd_idx)
    .fillna(-1)
    .astype(np.int32)
)

history_df = history_df.sort_values("userid").reset_index(drop=True)

hist_user_ids, first_indices = np.unique(
    history_df["userid"].to_numpy(dtype=np.int64),
    return_index=True,
)

hist_offsets = np.concatenate(
    [first_indices, np.array([len(history_df)], dtype=np.int64)]
).astype(np.int64)

hist_movie_ids = history_df["movieid"].to_numpy(dtype=np.int64)
hist_content_idx = history_df["content_idx"].to_numpy(dtype=np.int32)
hist_svd_movie_idx = history_df["svd_movie_idx"].to_numpy(dtype=np.int32)
hist_ratings = history_df["rating"].to_numpy(dtype=np.float32)

user_to_hist_pos = {
    int(uid): idx for idx, uid in enumerate(hist_user_ids)
}


def get_user_history_content(uid: int):
    pos = user_to_hist_pos.get(int(uid))

    if pos is None:
        return None, None

    start = hist_offsets[pos]
    end = hist_offsets[pos + 1]

    idxs = hist_content_idx[start:end]
    ratings_arr = hist_ratings[start:end]

    return idxs, ratings_arr


def content_predict_train(uid: int, mid: int) -> float:
    target_idx = movieid_to_content_idx.get(int(mid))

    if target_idx is None:
        return float(global_mean)

    hist_idxs, hist_ratings_arr = get_user_history_content(uid)

    if hist_idxs is None:
        return float(global_mean)

    mask = (hist_idxs >= 0) & (hist_idxs != target_idx)

    if not np.any(mask):
        return float(global_mean)

    valid_hist_idxs = hist_idxs[mask]
    valid_ratings = hist_ratings_arr[mask]

    sims = content_embeddings[valid_hist_idxs] @ content_embeddings[target_idx]

    denom = float(sims.sum()) + 1e-8

    pred = float(sims @ valid_ratings) / denom

    return clip_rating(pred)


# ═══════════════════════════════════════════════════════════════
# 7. ALPHA TUNING
# ═══════════════════════════════════════════════════════════════

print("Tuning hybrid alpha on validation set...")

val_users = val_data["userid"].to_numpy(dtype=np.int64)
val_movies = val_data["movieid"].to_numpy(dtype=np.int64)
val_actuals = val_data["rating"].to_numpy(dtype=np.float32)

svd_base = np.array(
    [svd_predict_train(u, m) for u, m in zip(val_users, val_movies)],
    dtype=np.float32,
)

cb_base = np.array(
    [content_predict_train(u, m) for u, m in zip(val_users, val_movies)],
    dtype=np.float32,
)

best_alpha, best_rmse = 0.7, np.inf

for alpha in np.linspace(0.0, 1.0, 21):
    preds = alpha * svd_base + (1.0 - alpha) * cb_base
    rmse = np.sqrt(mean_squared_error(val_actuals, preds))

    if rmse < best_rmse:
        best_rmse = float(rmse)
        best_alpha = float(alpha)

print(f"Best alpha: {best_alpha:.2f} | Validation RMSE: {best_rmse:.4f}")


# ═══════════════════════════════════════════════════════════════
# 8. METADATA / POPULARITY SUPPORT
# ═══════════════════════════════════════════════════════════════

print("Building metadata artifacts...")

catalog_cols = [
    "movieid",
    "clean_title",
    "genres",
    "poster_path",
    "poster_url",
    "vote_average",
    "release_date",
    "release_year",
    "language_name",
]

catalog_cols = [c for c in catalog_cols if c in movie_content.columns]

movie_catalog = movie_content[catalog_cols].copy()

movie_stats = train_data.groupby("movieid")["rating"].agg(["mean", "count"])

# Bayesian-ish popularity score for cold-start users.
# This is better than pure mean rating because it penalizes movies with very few ratings.
m = max(5, int(movie_stats["count"].quantile(0.50)))
C = global_mean

movie_stats["bayes_score"] = (
    (movie_stats["count"] / (movie_stats["count"] + m)) * movie_stats["mean"]
    + (m / (movie_stats["count"] + m)) * C
)

top_popular_movieids = (
    movie_stats
    .sort_values(["bayes_score", "count"], ascending=[False, False])
    .index
    .to_numpy(dtype=np.int64)
)

user_mean_ids = user_mean_series.index.to_numpy(dtype=np.int64)
user_mean_values = user_mean_series.to_numpy(dtype=np.float32)

movie_mean_ids = movie_mean_series.index.to_numpy(dtype=np.int64)
movie_mean_values = movie_mean_series.to_numpy(dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# 9. SAVE OPTIMIZED ARTIFACTS
# ═══════════════════════════════════════════════════════════════

print("Saving optimized artifacts...")

save_joblib(
    {
        "schema_version": 2,
        "rating_min": RATING_MIN,
        "rating_max": RATING_MAX,
        "global_mean": float(global_mean),
        "user_ids": svd_user_ids,
        "movie_ids": svd_movie_ids,
        "user_factors": user_factors,
        "item_factors": item_factors,
        "user_bias": user_bias,
        "item_bias": item_bias,
        "n_components": int(n_svd_components),
        "explained_variance_ratio_sum": float(
            svd_model.explained_variance_ratio_.sum()
        ),
    },
    "svd_artifacts.joblib",
    compress=3,
)

save_joblib(
    {
        "schema_version": 2,
        "movie_ids": content_movie_ids,
        "content_embeddings": content_embeddings,
        "n_components": int(n_content_components),
    },
    "content_artifacts.joblib",
    compress=3,
)

save_joblib(
    {
        "schema_version": 2,
        "hist_user_ids": hist_user_ids,
        "hist_offsets": hist_offsets,
        "hist_movie_ids": hist_movie_ids,
        "hist_content_idx": hist_content_idx,
        "hist_svd_movie_idx": hist_svd_movie_idx,
        "hist_ratings": hist_ratings,
        "user_mean_ids": user_mean_ids,
        "user_mean_values": user_mean_values,
        "movie_mean_ids": movie_mean_ids,
        "movie_mean_values": movie_mean_values,
    },
    "history_artifacts.joblib",
    compress=3,
)

save_joblib(
    {
        "schema_version": 2,
        "best_alpha": float(best_alpha),
        "validation_rmse": float(best_rmse),
    },
    "hybrid_artifacts.joblib",
    compress=3,
)

save_joblib(
    {
        "schema_version": 2,
        "movie_catalog": movie_catalog,
        "top_popular_movieids": top_popular_movieids,
        "n_users": int(df["userid"].nunique()),
        "n_movies": int(df["movieid"].nunique()),
        "n_ratings": int(len(df)),
    },
    "metadata_artifacts.joblib",
    compress=3,
)

if SAVE_TEXT_TRANSFORMERS:
    save_joblib(
        {
            "schema_version": 2,
            "tfidf": tfidf,
            "content_svd": content_svd,
        },
        "content_transform_artifacts.joblib",
        compress=3,
    )

print("\nOptimized artifacts saved successfully.")
