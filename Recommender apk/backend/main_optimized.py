"""
Movie Recommendation Backend — FastAPI Service (Optimized Artifacts)
====================================================================
Loads optimized pre-trained serialized artifacts (joblib) instead of training on startup.
Uses factor-based prediction (user/item factors + embeddings) instead of dense matrices.

Implements:
  - SVD Collaborative Filtering (factor-based)
  - Content-Based Filtering (normalized embeddings + cosine similarity via dot-product)
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
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query 
import requests
from requests.exceptions import HTTPError, Timeout
from pydantic import BaseModel, Field


warnings.filterwarnings("ignore")
np.random.seed(42)

# ───────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────
RATING_MIN, RATING_MAX = 1.0, 5.0
K_METRICS = 5

# ───────────────────────────────────────────────────────────────
# Artifact Loading
# ───────────────────────────────────────────────────────────────

from dotenv import load_dotenv

load_dotenv()



REQUIRED_ARTIFACTS = [
    "svd_artifacts.joblib",
    "content_artifacts.joblib",
    "history_artifacts.joblib",
    "hybrid_artifacts.joblib",
    "metadata_artifacts.joblib",
]

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "/tmp/artifacts"))
ARTIFACTS_BASE_URL = os.getenv("ARTIFACTS_BASE_URL", "").rstrip("/")


def has_all_artifacts(path: Path) -> bool:
    """Check whether all required artifact files exist."""
    return path.is_dir() and all((path / name).is_file() for name in REQUIRED_ARTIFACTS)


def download_file(url: str, destination: Path) -> None:
    """Download one artifact safely using streaming."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp_path = destination.with_suffix(destination.suffix + ".tmp")

    print(f"Downloading artifact:")
    print(f"  URL:  {url}")
    print(f"  DEST: {destination}")

    with requests.get(url, stream=True, timeout=(10, 180)) as response:
        response.raise_for_status()

        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    if temp_path.stat().st_size == 0:
        raise RuntimeError(f"Downloaded empty file from {url}")

    os.replace(temp_path, destination)

    size_mb = destination.stat().st_size / (1024 * 1024)
    print(f"Downloaded {destination.name}: {size_mb:.2f} MB")


def ensure_artifacts_available() -> Path:
    """
    Resolve artifacts directory.

    Priority:
    1. Existing local artifacts, useful for local development.
    2. Download from Cloudflare R2 into ARTIFACTS_DIR.
    """

    local_candidates = [
        ARTIFACTS_DIR,
        Path(__file__).resolve().parent.parent / "artifacts",
        Path(__file__).resolve().parent / "artifacts",
        Path.cwd() / "artifacts",
    ]

    for candidate in local_candidates:
        if has_all_artifacts(candidate):
            print(f"Using local artifacts from: {candidate}")
            return candidate

    if not ARTIFACTS_BASE_URL:
        searched = [str(p) for p in local_candidates]
        raise RuntimeError(
            "Artifacts not found locally and ARTIFACTS_BASE_URL is not set. "
            f"Searched: {searched}"
        )

    print(f"Artifacts not found locally. Downloading from R2: {ARTIFACTS_BASE_URL}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    for filename in REQUIRED_ARTIFACTS:
        destination = ARTIFACTS_DIR / filename

        if destination.exists() and destination.stat().st_size > 0:
            print(f"Artifact already exists: {destination}")
            continue

        url = f"{ARTIFACTS_BASE_URL}/{filename}"
        download_file(url, destination)

    if not has_all_artifacts(ARTIFACTS_DIR):
        raise RuntimeError(f"Artifact download failed. Missing files in {ARTIFACTS_DIR}")

    return ARTIFACTS_DIR


def load_artifacts(artifacts_dir: str | None = None) -> dict[str, Any]:
    """Load all optimized pre-trained artifacts from joblib files."""
    if artifacts_dir is None:
        artifacts_path = ensure_artifacts_available()
    else:
        artifacts_path = Path(artifacts_dir)

    print(f"Loading optimized artifacts from: {artifacts_path}")

    svd = joblib.load(artifacts_path / "svd_artifacts.joblib")
    content = joblib.load(artifacts_path / "content_artifacts.joblib")
    history = joblib.load(artifacts_path / "history_artifacts.joblib")
    hybrid = joblib.load(artifacts_path / "hybrid_artifacts.joblib")
    metadata = joblib.load(artifacts_path / "metadata_artifacts.joblib")

    print("✅ All artifacts loaded successfully.")


    # --- Build fast lookup dicts from compact arrays ---
    svd_user_ids = svd["user_ids"]
    svd_movie_ids = svd["movie_ids"]
    content_movie_ids = content["movie_ids"]
    hist_user_ids = history["hist_user_ids"]

    user_to_svd_idx = {int(uid): idx for idx, uid in enumerate(svd_user_ids)}
    movie_to_svd_idx = {int(mid): idx for idx, mid in enumerate(svd_movie_ids)}
    movie_to_content_idx = {int(mid): idx for idx, mid in enumerate(content_movie_ids)}
    user_to_hist_pos = {int(uid): idx for idx, uid in enumerate(hist_user_ids)}

    # Map from SVD movie positions to content embedding positions
    svd_to_content_idx = np.array(
        [movie_to_content_idx.get(int(mid), -1) for mid in svd_movie_ids],
        dtype=np.int32,
    )

    # Build mean dicts from compact arrays
    user_mean = dict(
        zip(history["user_mean_ids"].astype(int), history["user_mean_values"])
    )
    movie_mean = dict(
        zip(history["movie_mean_ids"].astype(int), history["movie_mean_values"])
    )

    # Build title lookup from movie_catalog
    catalog = metadata["movie_catalog"]
    movieid_to_title = {}
    if "movieid" in catalog.columns and "clean_title" in catalog.columns:
        movieid_to_title = (
            catalog.set_index("movieid")["clean_title"].to_dict()
        )

    state = {
        # SVD
        "rating_min": svd["rating_min"],
        "rating_max": svd["rating_max"],
        "global_mean": float(svd["global_mean"]),
        "user_factors": svd["user_factors"],
        "item_factors": svd["item_factors"],
        "user_bias": svd["user_bias"],
        "item_bias": svd["item_bias"],
        "svd_user_ids": svd_user_ids,
        "svd_movie_ids": svd_movie_ids,
        "user_to_svd_idx": user_to_svd_idx,
        "movie_to_svd_idx": movie_to_svd_idx,
        # Content
        "content_embeddings": content["content_embeddings"],
        "content_movie_ids": content_movie_ids,
        "movie_to_content_idx": movie_to_content_idx,
        "svd_to_content_idx": svd_to_content_idx,
        # History
        "hist_user_ids": hist_user_ids,
        "hist_offsets": history["hist_offsets"],
        "hist_content_idx": history["hist_content_idx"],
        "hist_svd_movie_idx": history["hist_svd_movie_idx"],
        "hist_ratings": history["hist_ratings"],
        "user_to_hist_pos": user_to_hist_pos,
        # Means for cold-start
        "user_mean": user_mean,
        "movie_mean": movie_mean,
        "train_users_set": set(int(x) for x in svd_user_ids),
        "train_movies_set": set(int(x) for x in svd_movie_ids),
        # Hybrid
        "best_alpha": float(hybrid["best_alpha"]),
        # Metadata
        "movie_catalog": catalog,
        "top_popular_movieids": metadata["top_popular_movieids"],
        "n_users": int(metadata["n_users"]),
        "n_movies": int(metadata["n_movies"]),
        "n_ratings": int(metadata["n_ratings"]),
        # Convenience
        "movieid_to_title": movieid_to_title,
    }

    print(f"Optimized artifacts loaded. best_alpha={state['best_alpha']:.2f}")
    return state


# ───────────────────────────────────────────────────────────────
# Global state (populated at startup)
# ───────────────────────────────────────────────────────────────
STATE: dict[str, Any] = {}


# ═══════════════════════════════════════════════════════════════
#  PREDICTION FUNCTIONS (operate on STATE)
# ═══════════════════════════════════════════════════════════════


def clip_rating(x: float) -> float:
    return float(np.clip(x, STATE["rating_min"], STATE["rating_max"]))


def svd_predict(uid: int, mid: int) -> float:
    """Factor-based SVD prediction using user/item latent factors."""
    ui = STATE["user_to_svd_idx"].get(int(uid))
    mi = STATE["movie_to_svd_idx"].get(int(mid))

    if ui is None or mi is None:
        return float(STATE["global_mean"])

    latent = float(STATE["user_factors"][ui] @ STATE["item_factors"][mi])
    pred = (
        STATE["global_mean"]
        + float(STATE["user_bias"][ui])
        + float(STATE["item_bias"][mi])
        + latent
    )

    return clip_rating(pred)


def get_user_history(uid: int):
    """Retrieve compact user history arrays for a given user ID."""
    pos = STATE["user_to_hist_pos"].get(int(uid))

    if pos is None:
        return None, None, None

    start = STATE["hist_offsets"][pos]
    end = STATE["hist_offsets"][pos + 1]

    return (
        STATE["hist_content_idx"][start:end],
        STATE["hist_svd_movie_idx"][start:end],
        STATE["hist_ratings"][start:end],
    )


def content_predict(uid: int, mid: int) -> float:
    """Content-based prediction using normalized embedding dot-products."""
    target_idx = STATE["movie_to_content_idx"].get(int(mid))

    if target_idx is None:
        return float(STATE["global_mean"])

    hist_content, _, ratings = get_user_history(uid)

    if hist_content is None:
        return float(STATE["global_mean"])

    mask = (hist_content >= 0) & (hist_content != target_idx)

    if not np.any(mask):
        return float(STATE["global_mean"])

    valid_hist_idx = hist_content[mask]
    valid_ratings = ratings[mask]

    sims = STATE["content_embeddings"][valid_hist_idx] @ STATE["content_embeddings"][target_idx]

    denom = float(sims.sum()) + 1e-8
    pred = float(sims @ valid_ratings) / denom

    return clip_rating(pred)


def hybrid_predict(uid: int, mid: int, alpha: float | None = None) -> float:
    if alpha is None:
        alpha = STATE["best_alpha"]
    svd = svd_predict(uid, mid)
    cb = content_predict(uid, mid)
    return clip_rating(alpha * svd + (1.0 - alpha) * cb)


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
#  OPTIONAL: Efficient top-N recommendation for all movies
# ═══════════════════════════════════════════════════════════════


def recommend_all_movies_for_user(uid: int, n: int = 10):
    """
    Efficiently compute hybrid scores for ALL known movies and return top-N.
    Uses batch processing for content similarity to avoid O(N*M) dense matrix.
    """
    uid = int(uid)

    ui = STATE["user_to_svd_idx"].get(uid)

    if ui is None:
        return STATE["top_popular_movieids"][:n].astype(int).tolist()

    # SVD scores for all known movies
    svd_scores = (
        STATE["global_mean"]
        + float(STATE["user_bias"][ui])
        + STATE["item_bias"]
        + STATE["item_factors"] @ STATE["user_factors"][ui]
    ).astype(np.float32)

    svd_scores = np.clip(svd_scores, STATE["rating_min"], STATE["rating_max"])

    cb_scores = np.full_like(svd_scores, STATE["global_mean"], dtype=np.float32)

    hist_content, hist_svd_idx, ratings = get_user_history(uid)

    if hist_content is not None:
        mask = hist_content >= 0

        if np.any(mask):
            valid_hist_content = hist_content[mask]
            valid_ratings = ratings[mask]

            valid_candidate_mask = STATE["svd_to_content_idx"] >= 0
            valid_candidate_positions = np.where(valid_candidate_mask)[0]
            candidate_content_idx = STATE["svd_to_content_idx"][valid_candidate_positions]

            batch_size = 2048

            for start in range(0, len(valid_candidate_positions), batch_size):
                end = start + batch_size

                batch_positions = valid_candidate_positions[start:end]
                batch_content_idx = candidate_content_idx[start:end]

                sims = (
                    STATE["content_embeddings"][batch_content_idx]
                    @ STATE["content_embeddings"][valid_hist_content].T
                )

                denom = sims.sum(axis=1) + 1e-8
                vals = (sims @ valid_ratings) / denom

                cb_scores[batch_positions] = np.clip(
                    vals, STATE["rating_min"], STATE["rating_max"]
                )

    final_scores = STATE["best_alpha"] * svd_scores + (1.0 - STATE["best_alpha"]) * cb_scores

    # Remove already watched movies
    if hist_svd_idx is not None:
        watched = hist_svd_idx[hist_svd_idx >= 0]
        final_scores[watched] = -np.inf

    n = min(n, len(final_scores))

    top_idx = np.argpartition(-final_scores, n - 1)[:n]
    top_idx = top_idx[np.argsort(-final_scores[top_idx])]

    return STATE["svd_movie_ids"][top_idx].astype(int).tolist()


# ═══════════════════════════════════════════════════════════════
#  SEARCH & RECOMMENDATION HELPERS
# ═══════════════════════════════════════════════════════════════


def search_movies(query: str, max_results: int = 20) -> list[dict]:
    """
    Ranked movie search by title, year, keyword, genre, overview.
    Returns list of dicts: {"movieid", "title", "score"}
    """
    catalog = STATE["movie_catalog"]
    query = query.strip().lower()

    movies = catalog.copy()

    # Ensure we have the required columns
    if "movieid" not in movies.columns or "clean_title" not in movies.columns:
        return []

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
        results.append(
            {"movieid": int(row["movieid"]), "title": row["clean_title"], "score": 1.0}
        )
        seen_ids.add(int(row["movieid"]))

    # 2. Year-only query
    if query.isdigit() and len(query) == 4 and year_col:
        yr_matches = movies[movies[year_col].astype(str).str.startswith(query)]
        for _, row in yr_matches.iterrows():
            mid = int(row["movieid"])
            if mid not in seen_ids:
                results.append(
                    {"movieid": mid, "title": row["clean_title"], "score": 0.9}
                )
                seen_ids.add(mid)

    # 3. Partial title match
    partial = movies[movies["clean_title"].str.contains(query, na=False, regex=False)]
    for _, row in partial.iterrows():
        mid = int(row["movieid"])
        if mid not in seen_ids:
            results.append(
                {"movieid": mid, "title": row["clean_title"], "score": 0.75}
            )
            seen_ids.add(mid)

    # 4. Genre / keyword / overview match
    for col in ["genres", "keywords", "overview"]:
        if col in movies.columns:
            mask = movies[col].str.lower().str.contains(query, na=False, regex=False)
            for _, row in movies[mask].iterrows():
                mid = int(row["movieid"])
                if mid not in seen_ids:
                    results.append(
                        {"movieid": mid, "title": row["clean_title"], "score": 0.5}
                    )
                    seen_ids.add(mid)

    return results[:max_results]


def content_similar_movies(movie_ids: list[int], top_k: int = 60) -> list[int]:
    """Given seed movie IDs, return content-similar candidates (excluding seeds)."""
    embeddings = STATE["content_embeddings"]
    m2c = STATE["movie_to_content_idx"]
    c2m = {idx: int(mid) for mid, idx in m2c.items()}

    valid_seeds = [m for m in movie_ids if m in m2c]
    if not valid_seeds:
        return []

    seed_embeddings = np.array([embeddings[m2c[m]] for m in valid_seeds])
    avg_embedding = seed_embeddings.mean(axis=0)

    # Cosine similarity via dot-product (embeddings are L2-normalized)
    sims = embeddings @ avg_embedding

    seed_idxs = {m2c[m] for m in valid_seeds}
    ranked = np.argsort(sims)[::-1]
    candidates = [c2m[i] for i in ranked if i not in seed_idxs]
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
    """Load artifacts on startup."""
    global STATE
    if not STATE:
        try:
            STATE = load_artifacts()
        except FileNotFoundError as e:
            raise RuntimeError(f"Artifacts not found: {e}")
    yield


app = FastAPI(
    title="Movie Recommender API (Optimized)",
    description="Hybrid SVD + Content-Based recommendation backend with cold-start support. "
                "Loads optimized pre-trained artifacts on startup using factor-based inference.",
    version="2.1.0",
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

    uvicorn.run("main_optimized:app", host="0.0.0.0", port=8000, reload=True)

