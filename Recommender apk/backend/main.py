"""
Movie Recommendation Backend — FastAPI Service
===============================================
Implements:
  - SVD Collaborative Filtering
  - Content-Based Filtering (TF-IDF + cosine similarity)
  - Hybrid Model (alpha-tuned on validation set)
  - Cold-Start Fallback (new user / new movie / both unknown)
  - Multi-query search (title, year, keyword, clue)

Endpoints:
  POST /recommend   → personalised recommendations
  GET  /health      → service health
  GET  /stats       → dataset & model statistics
  GET  /search      → search movies by query
"""

from __future__ import annotations

import os
import warnings
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
np.random.seed(42)

# ───────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────
RATING_MIN, RATING_MAX = 1.0, 5.0
K_METRICS = 5

# -------------------------------------------------------
# LOAD DATA TO TRAIN DATA AND BUILD INDEXES
# -------------------------------------------------------

from dotenv import load_dotenv
import os
import urllib.request
import gzip
import shutil

load_dotenv()

DATASET_URL = os.getenv("DATASET_URL")
DATASET_PATH = os.getenv("DATASET_PATH", "/tmp/final_dataset.csv")


def ensure_dataset():
    """
    Ensure dataset exists locally.
    If not, download from remote (.gz), extract, and save as CSV.
    """

    # --- Validate config ---
    if not DATASET_URL:
        raise RuntimeError("❌ DATASET_URL not set.")

    print(f"🌐 DATASET_URL = {DATASET_URL}")
    print(f"📁 DATASET_PATH = {DATASET_PATH}")

    # --- Ensure directory exists (/tmp safe on Render) ---
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)

    # --- Skip if already exists ---
    if os.path.exists(DATASET_PATH):
        print(f"✅ Dataset already exists at {DATASET_PATH}")
        return DATASET_PATH

    tmp_gz_path = DATASET_PATH + ".gz"

    try:
        print(f"⬇️ Downloading dataset from {DATASET_URL}...")

        # Custom request (kept from your version)
        req = urllib.request.Request(
            DATASET_URL,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        with urllib.request.urlopen(req) as response, open(tmp_gz_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)

        # --- Validate download ---
        if os.path.getsize(tmp_gz_path) < 1000:
            raise RuntimeError("❌ Download failed (file too small).")

        print("📦 Extracting dataset...")

        with gzip.open(tmp_gz_path, "rb") as f_in:
            with open(DATASET_PATH, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(tmp_gz_path)

        # --- Verify extraction ---
        if not os.path.exists(DATASET_PATH):
            raise RuntimeError("❌ Extraction failed.")

        print(f"🎉 Dataset ready at {DATASET_PATH}")
        print(f"📦 Size: {os.path.getsize(DATASET_PATH) / (1024**2):.2f} MB")

    except Exception as e:
        if os.path.exists(tmp_gz_path):
            os.remove(tmp_gz_path)
        print(f"❌ Error downloading/extracting: {e}")
        raise

    return DATASET_PATH

def resolve_csv_path():
    """
    Resolve dataset path, ensuring availability.
    Priority:
    1. Explicit env path
    2. Existing local file (common paths)
    3. Download via ensure_dataset()
    """

    # 1. Explicit env path
    env_path = os.getenv("DATASET_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # 2. Search common locations
    candidates = [
        "final_dataset.csv",
        "../final_dataset.csv",
        "../../final_dataset.csv",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    # 3. Download if not found
    return ensure_dataset()


# Final resolved path
CSV_PATH = resolve_csv_path()

# ───────────────────────────────────────────────────────────────
# Global state (populated at startup)
# ───────────────────────────────────────────────────────────────
STATE: dict[str, Any] = {}


# ═══════════════════════════════════════════════════════════════
#  MODEL TRAINING & LOADING
# ═══════════════════════════════════════════════════════════════

def load_and_train(csv_path: str) -> dict[str, Any]:
    """Load data, train all models, return inference-ready state."""

    # ── Load raw data ──────────────────────────────────────────
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["userid", "movieid", "rating"])
    df["userid"] = df["userid"].astype(int)
    df["movieid"] = df["movieid"].astype(int)
    df["clean_title"] = df["clean_title"].str.lower().str.strip()
    df["rating"] = df["rating"].clip(RATING_MIN, RATING_MAX)

    text_cols = ["overview", "keywords", "genres", "tagline", "language_name"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Build boosted content column
    df["content"] = (
        df.get("overview", "") + " " +
        df.get("genres", "") * 3 + " " +
        df.get("keywords", "") * 2 + " " +
        df.get("tagline", "") + " " +
        df.get("language_name", "")
    )

    # ── Train / val / test split ───────────────────────────────
    ratings = df[["userid", "movieid", "rating"]].copy().reset_index()
    train_data, temp = train_test_split(ratings, test_size=0.30, random_state=42)
    val_data, test_data = train_test_split(temp, test_size=0.50, random_state=42)
    train_data = train_data.drop(columns=["index"]).reset_index(drop=True)
    val_data = val_data.drop(columns=["index"]).reset_index(drop=True)
    test_data = test_data.drop(columns=["index"]).reset_index(drop=True)

    # ── Training statistics ────────────────────────────────────
    global_mean = train_data["rating"].mean()
    user_biases = train_data.groupby("userid")["rating"].mean() - global_mean
    item_biases = train_data.groupby("movieid")["rating"].mean() - global_mean
    movie_mean = train_data.groupby("movieid")["rating"].mean()
    user_mean = train_data.groupby("userid")["rating"].mean()

    # ── SVD Collaborative Filtering ────────────────────────────
    train_data_norm = train_data.copy()
    train_data_norm["rating"] = train_data_norm.apply(
        lambda r: r["rating"]
        - global_mean
        - user_biases.get(r["userid"], 0)
        - item_biases.get(r["movieid"], 0),
        axis=1,
    )
    train_matrix = train_data_norm.pivot_table(
        index="userid", columns="movieid", values="rating"
    ).fillna(0)

    n_components = min(600, min(train_matrix.shape) - 1)
    svd_model = TruncatedSVD(n_components=n_components, random_state=42)
    U = svd_model.fit_transform(train_matrix)
    Vt = svd_model.components_
    pred_matrix_norm = U @ Vt
    pred_df = pd.DataFrame(
        pred_matrix_norm,
        index=train_matrix.index,
        columns=train_matrix.columns,
    )

    # ── Content-Based Filtering ────────────────────────────────
    movie_content = df.drop_duplicates(subset="movieid").reset_index(drop=True).copy()
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        min_df=2,
        max_df=0.85,
        ngram_range=(1, 2),
    )
    tfidf_matrix = tfidf.fit_transform(movie_content["content"])
    content_svd = TruncatedSVD(n_components=100, random_state=42)
    reduced_matrix = content_svd.fit_transform(tfidf_matrix)
    content_similarity = cosine_similarity(reduced_matrix)

    movieid_to_idx = {mid: i for i, mid in enumerate(movie_content["movieid"])}
    idx_to_movieid = {i: mid for mid, i in movieid_to_idx.items()}
    movieid_to_title = movie_content.set_index("movieid")["clean_title"].to_dict()
    title_to_movieid = {v: k for k, v in movieid_to_title.items()}
    title_to_idx = {
        t: movieid_to_idx[mid]
        for t, mid in title_to_movieid.items()
        if mid in movieid_to_idx
    }

    user_history_map = (
        train_data.groupby("userid")
        .apply(lambda x: list(zip(x["movieid"], x["rating"])))
        .to_dict()
    )

    # ── Alpha tuning on validation set ─────────────────────────
    def _svd_predict(uid: int, mid: int) -> float:
        if uid not in pred_df.index or mid not in pred_df.columns:
            return float(global_mean)
        latent = pred_df.loc[uid, mid]
        pred = (
            global_mean
            + user_biases.get(uid, 0)
            + item_biases.get(mid, 0)
            + latent
        )
        return float(np.clip(pred, RATING_MIN, RATING_MAX))

    def _content_predict(uid: int, mid: int) -> float:
        if mid not in movieid_to_idx or uid not in user_history_map:
            return float(global_mean)
        t_idx = movieid_to_idx[mid]
        history = user_history_map[uid]
        sims, rts = [], []
        for hm, hr in history:
            if hm == mid or hm not in movieid_to_idx:
                continue
            sims.append(content_similarity[t_idx, movieid_to_idx[hm]])
            rts.append(hr)
        if not sims:
            return float(global_mean)
        sims_arr = np.array(sims)
        rts_arr = np.array(rts)
        pred = np.dot(sims_arr, rts_arr) / (sims_arr.sum() + 1e-8)
        return float(np.clip(pred, RATING_MIN, RATING_MAX))

    val_users = val_data.userid.values
    val_movies = val_data.movieid.values
    val_actuals = val_data["rating"].values
    svd_base = np.array([_svd_predict(u, m) for u, m in zip(val_users, val_movies)])
    cb_base = np.array([_content_predict(u, m) for u, m in zip(val_users, val_movies)])

    best_alpha, best_rmse = 0.7, np.inf
    for a in np.linspace(0.0, 1.0, 21):
        preds = a * svd_base + (1 - a) * cb_base
        rmse = np.sqrt(mean_squared_error(val_actuals, preds))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(a)

    # ── Cold-start support sets ────────────────────────────────
    train_users_set = set(train_data["userid"])
    train_movies_set = set(train_data["movieid"])

    # Package state
    return {
        "df": df,
        "train_data": train_data,
        "pred_df": pred_df,
        "global_mean": global_mean,
        "user_biases": user_biases,
        "item_biases": item_biases,
        "movie_mean": movie_mean,
        "user_mean": user_mean,
        "content_similarity": content_similarity,
        "movieid_to_idx": movieid_to_idx,
        "idx_to_movieid": idx_to_movieid,
        "movieid_to_title": movieid_to_title,
        "title_to_movieid": title_to_movieid,
        "title_to_idx": title_to_idx,
        "user_history_map": user_history_map,
        "best_alpha": best_alpha,
        "train_users_set": train_users_set,
        "train_movies_set": train_movies_set,
        "n_users": df["userid"].nunique(),
        "n_movies": df["movieid"].nunique(),
        "n_ratings": len(df),
    }


# ═══════════════════════════════════════════════════════════════
#  PREDICTION FUNCTIONS (operate on STATE)
# ═══════════════════════════════════════════════════════════════

def svd_predict(uid: int, mid: int) -> float:
    pred_df = STATE["pred_df"]
    global_mean = STATE["global_mean"]
    user_biases = STATE["user_biases"]
    item_biases = STATE["item_biases"]
    if uid not in pred_df.index or mid not in pred_df.columns:
        return float(global_mean)
    latent = pred_df.loc[uid, mid]
    pred = global_mean + user_biases.get(uid, 0) + item_biases.get(mid, 0) + latent
    return float(np.clip(pred, RATING_MIN, RATING_MAX))


def content_predict(uid: int, mid: int) -> float:
    movieid_to_idx = STATE["movieid_to_idx"]
    content_similarity = STATE["content_similarity"]
    user_history_map = STATE["user_history_map"]
    global_mean = STATE["global_mean"]
    if mid not in movieid_to_idx or uid not in user_history_map:
        return float(global_mean)
    t_idx = movieid_to_idx[mid]
    history = user_history_map[uid]
    sims, rts = [], []
    for hm, hr in history:
        if hm == mid or hm not in movieid_to_idx:
            continue
        sims.append(content_similarity[t_idx, movieid_to_idx[hm]])
        rts.append(hr)
    if not sims:
        return float(global_mean)
    sims_arr = np.array(sims)
    rts_arr = np.array(rts)
    pred = np.dot(sims_arr, rts_arr) / (sims_arr.sum() + 1e-8)
    return float(np.clip(pred, RATING_MIN, RATING_MAX))


def hybrid_predict(uid: int, mid: int, alpha: float | None = None) -> float:
    if alpha is None:
        alpha = STATE["best_alpha"]
    svd = svd_predict(uid, mid)
    cb = content_predict(uid, mid)
    return float(np.clip(alpha * svd + (1 - alpha) * cb, RATING_MIN, RATING_MAX))


def cold_start_predict(uid: int, mid: int, alpha: float | None = None):
    """
    Returns (score, mode).
    mode ∈ {"hybrid", "cold_user", "cold_movie", "cold_both", "content"}
    """
    if alpha is None:
        alpha = STATE["best_alpha"]
    train_users_set = STATE["train_users_set"]
    train_movies_set = STATE["train_movies_set"]
    movie_mean = STATE["movie_mean"]
    user_mean = STATE["user_mean"]
    global_mean = STATE["global_mean"]

    u_known = uid in train_users_set
    m_known = mid in train_movies_set

    if u_known and m_known:
        return hybrid_predict(uid, mid, alpha), "hybrid"
    if not u_known and m_known:
        return float(movie_mean.get(mid, global_mean)), "cold_user"
    if u_known and not m_known:
        return float(user_mean.get(uid, global_mean)), "cold_movie"
    return float(global_mean), "cold_both"


# ═══════════════════════════════════════════════════════════════
#  SEARCH & RECOMMENDATION HELPERS
# ═══════════════════════════════════════════════════════════════

def search_movies(query: str, max_results: int = 20) -> list[dict]:
    """
    Ranked movie search by title, year, keyword, genre, overview.
    Returns list of dicts: {"movieid", "title", "score"}
    """
    df = STATE["df"]
    query = query.strip().lower()

    movies = df.drop_duplicates(subset="movieid")[
        ["movieid", "clean_title"]
        + [c for c in ["genres", "keywords", "overview", "release_year", "year"] if c in df.columns]
    ].copy()

    year_col = None
    for col in ["release_year", "year"]:
        if col in movies.columns:
            year_col = col
            break

    results: list[dict] = []
    seen_ids: set[int] = set()

    # 1. Exact title match
    exact = movies[movies["clean_title"] == query]
    for _, row in exact.iterrows():
        results.append({"movieid": int(row["movieid"]), "title": row["clean_title"], "score": 1.0})
        seen_ids.add(int(row["movieid"]))

    # 2. Year-only query
    if query.isdigit() and len(query) == 4 and year_col:
        yr_matches = movies[movies[year_col].astype(str).str.startswith(query)]
        for _, row in yr_matches.iterrows():
            mid = int(row["movieid"])
            if mid not in seen_ids:
                results.append({"movieid": mid, "title": row["clean_title"], "score": 0.9})
                seen_ids.add(mid)

    # 3. Partial title match
    partial = movies[movies["clean_title"].str.contains(query, na=False, regex=False)]
    for _, row in partial.iterrows():
        mid = int(row["movieid"])
        if mid not in seen_ids:
            results.append({"movieid": mid, "title": row["clean_title"], "score": 0.75})
            seen_ids.add(mid)

    # 4. Genre / keyword / overview match
    for col in ["genres", "keywords", "overview"]:
        if col in movies.columns:
            mask = movies[col].str.lower().str.contains(query, na=False, regex=False)
            for _, row in movies[mask].iterrows():
                mid = int(row["movieid"])
                if mid not in seen_ids:
                    results.append({"movieid": mid, "title": row["clean_title"], "score": 0.5})
                    seen_ids.add(mid)

    return results[:max_results]


def content_similar_movies(movie_ids: list[int], top_k: int = 60) -> list[int]:
    """Given seed movie IDs, return content-similar candidates (excluding seeds)."""
    cs = STATE["content_similarity"]
    m2i = STATE["movieid_to_idx"]
    i2m = STATE["idx_to_movieid"]

    valid_seeds = [m for m in movie_ids if m in m2i]
    if not valid_seeds:
        return []

    avg_sim = np.mean([cs[m2i[m]] for m in valid_seeds], axis=0)
    seed_idxs = {m2i[m] for m in valid_seeds}
    ranked = np.argsort(avg_sim)[::-1]
    candidates = [i2m[i] for i in ranked if i not in seed_idxs]
    return candidates[:top_k]


def recommend_for_user(uid: int, candidate_movie_ids: list[int], top_k: int = 10) -> list[dict]:
    """
    Score candidates with cold_start_predict (dispatches to hybrid or fallback).
    Returns sorted list of recommendation dicts.
    """
    scored = []
    for mid in candidate_movie_ids:
        score, mode = cold_start_predict(uid, mid)
        scored.append(
            {
                "movieid": int(mid),
                "title": STATE["movieid_to_title"].get(mid, "Unknown"),
                "score": round(float(score), 4),
                "mode": mode,
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def generate_recommendations(
    query: str, user_id: int | None = None, top_k: int = 8
) -> dict[str, Any]:
    """
    End-to-end recommendation pipeline:
      1. Search for seed movies matching the query.
      2. Expand candidates via content similarity.
      3. Score & rank using hybrid + cold-start logic.
    """
    matches = search_movies(query, max_results=25)
    if not matches:
        return {
            "recommendations": [],
            "seed_matches": [],
            "user_status": "anonymous",
            "best_alpha": STATE["best_alpha"],
        }

    seed_ids = [m["movieid"] for m in matches]
    seed_id_set = set(seed_ids)

    content_candidates = content_similar_movies(seed_ids[:5], top_k=60)
    all_candidates = seed_ids + [c for c in content_candidates if c not in seed_id_set]

    if user_id is not None:
        recs = recommend_for_user(user_id, all_candidates, top_k=top_k)
        is_new_user = user_id not in STATE["train_users_set"]
        user_status = "new" if is_new_user else "existing"
    else:
        recs = []
        for mid in all_candidates:
            score = float(STATE["movie_mean"].get(mid, STATE["global_mean"]))
            recs.append(
                {
                    "movieid": int(mid),
                    "title": STATE["movieid_to_title"].get(mid, "Unknown"),
                    "score": round(score, 4),
                    "mode": "content",
                }
            )
        recs.sort(key=lambda x: x["score"], reverse=True)
        recs = recs[:top_k]
        user_status = "anonymous"

    return {
        "recommendations": recs,
        "seed_matches": matches[:10],
        "user_status": user_status,
        "best_alpha": STATE["best_alpha"],
    }


# ═══════════════════════════════════════════════════════════════
#  FASTAPI APP & LIFECYCLE
# ═══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global STATE
    if not STATE:
        try:
            STATE = load_and_train(CSV_PATH)
            print(f" Models loaded. best_alpha={STATE['best_alpha']:.2f}")
        except FileNotFoundError:
            raise RuntimeError(f"Dataset not found at {CSV_PATH}")
    yield
    # Shutdown cleanup (if needed)


app = FastAPI(
    title="Movie Recommender API",
    description="Hybrid SVD + Content-Based recommendation backend with cold-start support.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic schemas ──────────────────────────────────────────

class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query: title, keyword, year, or clue")
    user_id: int | None = Field(None, ge=1, description="Optional user ID for personalised recommendations")
    top_k: int = Field(8, ge=1, le=50, description="Number of recommendations to return")


class RecommendResponse(BaseModel):
    recommendations: list[dict]
    seed_matches: list[dict]
    user_status: str
    best_alpha: float


class StatsResponse(BaseModel):
    n_users: int
    n_movies: int
    n_ratings: int
    best_alpha: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


# ── Endpoints ─────────────────────────────────────────────────

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(payload: RecommendRequest):
    """Generate movie recommendations for a query and optional user."""
    result = generate_recommendations(
        query=payload.query,
        user_id=payload.user_id,
        top_k=payload.top_k,
    )
    return result


@app.get("/search")
async def search(query: str = Query(..., min_length=1), max_results: int = Query(20, ge=1, le=100)):
    """Search movies by title, keyword, year, or clue."""
    matches = search_movies(query, max_results=max_results)
    return {"matches": matches}


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Return dataset and model statistics."""
    return {
        "n_users": STATE["n_users"],
        "n_movies": STATE["n_movies"],
        "n_ratings": STATE["n_ratings"],
        "best_alpha": STATE["best_alpha"],
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    return {"status": "ok", "models_loaded": bool(STATE)}


# ── Local dev entrypoint ──────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

