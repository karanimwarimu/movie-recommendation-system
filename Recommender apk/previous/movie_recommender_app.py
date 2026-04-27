"""
Movie Recommendation System — Streamlit UI
Implements:
  - Hybrid Recommendation (SVD + Content-Based, alpha-tuned)
  - Cold-Start Handling for new / sparse users
  - Multi-query search: keyword, clue, year, or movie title
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1a1d27; }

    /* Card style for results */
    .movie-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #3d4266;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 12px;
        transition: transform 0.15s ease;
    }
    .movie-card:hover { transform: translateY(-2px); border-color: #6c77e8; }

    .movie-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #c9d1ff;
        margin-bottom: 6px;
        text-transform: capitalize;
    }
    .movie-score {
        font-size: 0.85rem;
        color: #7c85c9;
    }
    .score-bar-bg {
        background: #2a2d42;
        border-radius: 6px;
        height: 8px;
        margin-top: 6px;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 8px;
        border-radius: 6px;
        background: linear-gradient(90deg, #4f5eb8, #9c6fe4);
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 4px;
    }
    .badge-hybrid   { background:#2e3a78; color:#a0acff; }
    .badge-coldstart{ background:#3a2020; color:#ff9e9e; }
    .badge-content  { background:#1e3a2a; color:#7effd4; }

    /* Metric tiles */
    .metric-tile {
        background: #1a1d27;
        border: 1px solid #2c3050;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-label { font-size: 0.78rem; color: #7c85c9; margin-bottom: 2px; }
    .metric-value { font-size: 1.4rem; font-weight: 700; color: #c9d1ff; }

    /* Header */
    .hero {
        background: linear-gradient(135deg, #1c1f35 0%, #252840 100%);
        border-radius: 16px;
        padding: 28px 32px 22px;
        margin-bottom: 24px;
        border: 1px solid #3d4266;
    }
    .hero h1 { margin: 0; font-size: 2rem; color: #c9d1ff; }
    .hero p  { color: #7c85c9; margin: 6px 0 0; font-size: 0.95rem; }

    /* Section headers */
    .section-header {
        font-size: 1rem; font-weight: 600; color: #9ca3f5;
        border-bottom: 1px solid #2c3050; padding-bottom: 6px;
        margin: 18px 0 12px;
    }

    /* Input labels override */
    label { color: #b0b8e8 !important; }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #4f5eb8, #7b5ea7);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* Spinner */
    [data-testid="stSpinner"] { color: #9ca3f5 !important; }

    /* Info / warning boxes */
    .stAlert { border-radius: 10px; }

    /* Number input / text input */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background: #1e2130 !important;
        color: #e0e0e0 !important;
        border: 1px solid #3d4266 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
RATING_MIN, RATING_MAX = 1.0, 5.0
K_METRICS = 5


# ─────────────────────────────────────────────────────────
# DATA & MODEL LOADING  (cached)
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️  Training models — this runs once...")
def load_and_train(csv_path: str):
    """Load data, train all models, return everything needed for inference."""

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["userid", "movieid", "rating"])
    df["userid"]  = df["userid"].astype(int)
    df["movieid"] = df["movieid"].astype(int)
    df["clean_title"] = df["clean_title"].str.lower().str.strip()
    df["rating"] = df["rating"].clip(RATING_MIN, RATING_MAX)

    text_cols = ["overview", "keywords", "genres", "tagline", "language_name"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    df["content"] = (
        df.get("overview", "")       + " " +
        df.get("genres", "")         * 3  + " " +
        df.get("keywords", "")       * 2  + " " +
        df.get("tagline", "")        + " " +
        df.get("language_name", "")
    )

    # ── Splits ──────────────────────────────────────────────
    ratings = df[["userid", "movieid", "rating"]].copy().reset_index()
    train_data, temp  = train_test_split(ratings, test_size=0.30, random_state=42)
    val_data,   test_data = train_test_split(temp,  test_size=0.50, random_state=42)
    train_data = train_data.drop(columns=["index"]).reset_index(drop=True)
    val_data   = val_data.drop(columns=["index"]).reset_index(drop=True)
    test_data  = test_data.drop(columns=["index"]).reset_index(drop=True)

    # ── Training statistics ──────────────────────────────────
    global_mean = train_data["rating"].mean()
    user_biases = train_data.groupby("userid")["rating"].mean() - global_mean
    item_biases = train_data.groupby("movieid")["rating"].mean() - global_mean
    movie_mean  = train_data.groupby("movieid")["rating"].mean()
    user_mean   = train_data.groupby("userid")["rating"].mean()

    # ── SVD model ────────────────────────────────────────────
    train_data_norm = train_data.copy()
    train_data_norm["rating"] = train_data_norm.apply(
        lambda r: r["rating"]
                  - global_mean
                  - user_biases.get(r["userid"],  0)
                  - item_biases.get(r["movieid"], 0),
        axis=1,
    )
    train_matrix = train_data_norm.pivot_table(
        index="userid", columns="movieid", values="rating"
    ).fillna(0)

    N_COMPONENTS = min(200, min(train_matrix.shape) - 1)
    svd_model = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    U  = svd_model.fit_transform(train_matrix)
    Vt = svd_model.components_
    pred_matrix_norm = U @ Vt
    pred_df = pd.DataFrame(
        pred_matrix_norm,
        index=train_matrix.index,
        columns=train_matrix.columns,
    )

    # ── Content-Based model ──────────────────────────────────
    movie_content = df.drop_duplicates(subset="movieid").reset_index(drop=True).copy()
    tfidf = TfidfVectorizer(
        max_features=5000, stop_words="english",
        min_df=2, max_df=0.85, ngram_range=(1, 2),
    )
    tfidf_matrix = tfidf.fit_transform(movie_content["content"])
    content_svd   = TruncatedSVD(n_components=100, random_state=42)
    reduced_matrix = content_svd.fit_transform(tfidf_matrix)
    content_similarity = cosine_similarity(reduced_matrix)

    movieid_to_idx  = {mid: i for i, mid in enumerate(movie_content["movieid"])}
    idx_to_movieid  = {i: mid for mid, i in movieid_to_idx.items()}
    movieid_to_title = movie_content.set_index("movieid")["clean_title"].to_dict()
    title_to_movieid = {v: k for k, v in movieid_to_title.items()}
    title_to_idx     = {
        t: movieid_to_idx[mid]
        for t, mid in title_to_movieid.items()
        if mid in movieid_to_idx
    }

    user_history_map = train_data.groupby("userid").apply(
        lambda x: list(zip(x["movieid"], x["rating"]))
    ).to_dict()

    # ── Alpha tuning on val_data ──────────────────────────────
    def _svd_predict(uid, mid):
        if uid not in pred_df.index or mid not in pred_df.columns:
            return float(global_mean)
        latent = pred_df.loc[uid, mid]
        pred   = global_mean + user_biases.get(uid, 0) + item_biases.get(mid, 0) + latent
        return float(np.clip(pred, RATING_MIN, RATING_MAX))

    def _content_predict(uid, mid):
        if mid not in movieid_to_idx or uid not in user_history_map:
            return float(global_mean)
        t_idx   = movieid_to_idx[mid]
        history = user_history_map[uid]
        sims, rts = [], []
        for hm, hr in history:
            if hm == mid or hm not in movieid_to_idx:
                continue
            sims.append(content_similarity[t_idx, movieid_to_idx[hm]])
            rts.append(hr)
        if not sims:
            return float(global_mean)
        sims = np.array(sims); rts = np.array(rts)
        return float(np.clip(np.dot(sims, rts) / (sims.sum() + 1e-8), RATING_MIN, RATING_MAX))

    val_users  = val_data.userid.values
    val_movies = val_data.movieid.values
    val_act    = val_data["rating"].values
    svd_base   = np.array([_svd_predict(u, m) for u, m in zip(val_users, val_movies)])
    cb_base    = np.array([_content_predict(u, m) for u, m in zip(val_users, val_movies)])

    best_alpha, best_rmse = 0.7, np.inf
    for a in np.linspace(0.0, 1.0, 21):
        preds = a * svd_base + (1 - a) * cb_base
        rmse  = np.sqrt(mean_squared_error(val_act, preds))
        if rmse < best_rmse:
            best_rmse  = rmse
            best_alpha = a

    # ── Cold-start sets ──────────────────────────────────────
    train_users_set  = set(train_data["userid"])
    train_movies_set = set(train_data["movieid"])

    # Package everything
    return dict(
        df               = df,
        train_data       = train_data,
        pred_df          = pred_df,
        global_mean      = global_mean,
        user_biases      = user_biases,
        item_biases      = item_biases,
        movie_mean       = movie_mean,
        user_mean        = user_mean,
        content_similarity = content_similarity,
        movieid_to_idx   = movieid_to_idx,
        idx_to_movieid   = idx_to_movieid,
        movieid_to_title = movieid_to_title,
        title_to_movieid = title_to_movieid,
        title_to_idx     = title_to_idx,
        user_history_map = user_history_map,
        best_alpha       = best_alpha,
        train_users_set  = train_users_set,
        train_movies_set = train_movies_set,
    )


# ─────────────────────────────────────────────────────────
# PREDICTION FUNCTIONS
# ─────────────────────────────────────────────────────────
def svd_predict(uid, mid, state):
    pred_df     = state["pred_df"]
    global_mean = state["global_mean"]
    user_biases = state["user_biases"]
    item_biases = state["item_biases"]
    if uid not in pred_df.index or mid not in pred_df.columns:
        return float(global_mean)
    latent = pred_df.loc[uid, mid]
    pred   = global_mean + user_biases.get(uid, 0) + item_biases.get(mid, 0) + latent
    return float(np.clip(pred, RATING_MIN, RATING_MAX))


def content_predict(uid, mid, state):
    movieid_to_idx     = state["movieid_to_idx"]
    content_similarity = state["content_similarity"]
    user_history_map   = state["user_history_map"]
    global_mean        = state["global_mean"]
    if mid not in movieid_to_idx or uid not in user_history_map:
        return float(global_mean)
    t_idx   = movieid_to_idx[mid]
    history = user_history_map[uid]
    sims, rts = [], []
    for hm, hr in history:
        if hm == mid or hm not in movieid_to_idx:
            continue
        sims.append(content_similarity[t_idx, movieid_to_idx[hm]])
        rts.append(hr)
    if not sims:
        return float(global_mean)
    sims = np.array(sims); rts = np.array(rts)
    return float(np.clip(np.dot(sims, rts) / (sims.sum() + 1e-8), RATING_MIN, RATING_MAX))


def hybrid_predict(uid, mid, alpha, state):
    svd = svd_predict(uid, mid, state)
    cb  = content_predict(uid, mid, state)
    return float(np.clip(alpha * svd + (1 - alpha) * cb, RATING_MIN, RATING_MAX))


def cold_start_predict(uid, mid, state):
    alpha            = state["best_alpha"]
    train_users_set  = state["train_users_set"]
    train_movies_set = state["train_movies_set"]
    movie_mean       = state["movie_mean"]
    user_mean        = state["user_mean"]
    global_mean      = state["global_mean"]

    u_known = uid in train_users_set
    m_known = mid in train_movies_set

    if u_known and m_known:
        return hybrid_predict(uid, mid, alpha, state), "hybrid"
    if not u_known and m_known:
        return float(movie_mean.get(mid, global_mean)), "cold_user"
    if u_known and not m_known:
        return float(user_mean.get(uid, global_mean)), "cold_movie"
    return float(global_mean), "cold_both"


# ─────────────────────────────────────────────────────────
# QUERY MATCHING  (keyword / year / title / clue)
# ─────────────────────────────────────────────────────────
def search_movies(query: str, state: dict, max_results: int = 20) -> list[dict]:
    """
    Return a ranked list of matching movies from the dataset.
    Handles: exact title match, year filter, keyword/clue match.
    """
    df    = state["df"]
    query = query.strip().lower()

    # Deduplicated movie rows
    movies = df.drop_duplicates(subset="movieid")[
        ["movieid", "clean_title"] +
        [c for c in ["genres", "keywords", "overview", "release_year", "year"]
         if c in df.columns]
    ].copy()

    # Try year extraction
    year_col = None
    for col in ["release_year", "year"]:
        if col in movies.columns:
            year_col = col
            break

    results = []

    # 1. Exact title match
    exact = movies[movies["clean_title"] == query]
    for _, row in exact.iterrows():
        results.append({"movieid": row["movieid"], "title": row["clean_title"], "score": 1.0})

    seen_ids = {r["movieid"] for r in results}

    # 2. Year-only query (4 digits)
    if query.isdigit() and len(query) == 4 and year_col:
        yr = int(query)
        yr_matches = movies[movies[year_col].astype(str).str.startswith(query)]
        for _, row in yr_matches.iterrows():
            if row["movieid"] not in seen_ids:
                results.append({"movieid": row["movieid"], "title": row["clean_title"], "score": 0.9})
                seen_ids.add(row["movieid"])

    # 3. Partial title match
    partial = movies[movies["clean_title"].str.contains(query, na=False, regex=False)]
    for _, row in partial.iterrows():
        if row["movieid"] not in seen_ids:
            results.append({"movieid": row["movieid"], "title": row["clean_title"], "score": 0.75})
            seen_ids.add(row["movieid"])

    # 4. Genre / keyword / overview match
    for col in ["genres", "keywords", "overview"]:
        if col in movies.columns:
            mask = movies[col].str.lower().str.contains(query, na=False, regex=False)
            for _, row in movies[mask].iterrows():
                if row["movieid"] not in seen_ids:
                    results.append({
                        "movieid": row["movieid"],
                        "title": row["clean_title"],
                        "score": 0.5,
                    })
                    seen_ids.add(row["movieid"])

    return results[:max_results]


def recommend_for_user(uid: int, candidate_movie_ids: list, state: dict, top_k: int = 10):
    """
    For a given user and set of candidate movies, return top-k recommendations
    using cold_start_predict (which dispatches to hybrid or fallback automatically).
    """
    scored = []
    for mid in candidate_movie_ids:
        score, mode = cold_start_predict(uid, mid, state)
        scored.append({
            "movieid": mid,
            "title"  : state["movieid_to_title"].get(mid, "Unknown"),
            "score"  : score,
            "mode"   : mode,
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def content_similar_movies(movie_ids: list, state: dict, top_k: int = 30) -> list[int]:
    """Given seed movie IDs, find content-similar candidates."""
    cs  = state["content_similarity"]
    m2i = state["movieid_to_idx"]
    i2m = state["idx_to_movieid"]

    if not movie_ids:
        return []

    # Average similarity vectors for seeds that are in the content index
    valid_seeds = [m for m in movie_ids if m in m2i]
    if not valid_seeds:
        return []

    avg_sim = np.mean([cs[m2i[m]] for m in valid_seeds], axis=0)
    seed_idxs = {m2i[m] for m in valid_seeds}
    ranked = np.argsort(avg_sim)[::-1]
    candidates = [i2m[i] for i in ranked if i not in seed_idxs]
    return candidates[:top_k]


# ─────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────
MODE_LABELS = {
    "hybrid"    : ("🔀 Hybrid",       "badge-hybrid"),
    "cold_user" : ("❄️ Cold-Start",   "badge-coldstart"),
    "cold_movie": ("❄️ Cold-Start",   "badge-coldstart"),
    "cold_both" : ("❄️ Cold-Start",   "badge-coldstart"),
    "content"   : ("📖 Content",       "badge-content"),
}

def render_movie_card(rank: int, title: str, score: float, mode: str):
    label_txt, badge_cls = MODE_LABELS.get(mode, ("🔀 Hybrid", "badge-hybrid"))
    fill_pct = int((score / RATING_MAX) * 100)
    st.markdown(f"""
    <div class="movie-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div class="movie-title">#{rank} &nbsp; {title.title()}</div>
            <span class="badge {badge_cls}">{label_txt}</span>
        </div>
        <div class="movie-score">Predicted score: <b>{score:.2f}</b> / 5.00</div>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{fill_pct}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️  Settings")
    csv_path = st.text_input(
        "Dataset path (CSV)",
        value="final_dataset.csv",
        help="Relative or absolute path to your final_dataset.csv file",
    )
    top_k = st.slider("Number of recommendations", min_value=3, max_value=20, value=8)
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
**Models inside:**
- 🤝 SVD Collaborative Filtering
- 📖 Content-Based (TF-IDF + SVD)
- 🔀 Hybrid (alpha-tuned)
- ❄️ Cold-Start fallback

The system auto-selects the right
strategy for each user/movie pair.
""")

# ─────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎬 Movie Recommendation System</h1>
    <p>Hybrid SVD + Content-Based engine with cold-start handling &mdash;
       search by title, keyword, genre, year, or any clue.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# LOAD STATE
# ─────────────────────────────────────────────────────────
try:
    state = load_and_train(csv_path)
except FileNotFoundError:
    st.error(
        f"❌  Dataset not found at **{csv_path}**.  "
        "Update the path in the sidebar and refresh."
    )
    st.stop()
except Exception as e:
    st.error(f"❌  Error loading data: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────
# QUICK STATS
# ─────────────────────────────────────────────────────────
df   = state["df"]
n_u  = df["userid"].nunique()
n_m  = df["movieid"].nunique()
n_r  = len(df)
alpha = state["best_alpha"]

c1, c2, c3, c4 = st.columns(4)
for col, label, val in [
    (c1, "👤 Users",      f"{n_u:,}"),
    (c2, "🎬 Movies",     f"{n_m:,}"),
    (c3, "⭐ Ratings",    f"{n_r:,}"),
    (c4, "🎛️ Best α",    f"{alpha:.2f}"),
]:
    col.markdown(f"""
    <div class="metric-tile">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{val}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔍 Search & Recommend</div>', unsafe_allow_html=True)

col_q, col_u = st.columns([3, 1])
with col_q:
    query = st.text_input(
        "Search query",
        placeholder="e.g.  toy story  ·  1994  ·  space adventure  ·  romantic comedy",
        label_visibility="collapsed",
    )
with col_u:
    user_id_input = st.text_input(
        "User ID",
        placeholder="User ID (optional)",
        label_visibility="collapsed",
    )

run = st.button("🎬 Get Recommendations", use_container_width=False)

# ─────────────────────────────────────────────────────────
# RECOMMENDATION LOGIC
# ─────────────────────────────────────────────────────────
if run:
    if not query.strip():
        st.warning("Please enter a search query.")
        st.stop()

    # Parse user ID
    uid = None
    if user_id_input.strip():
        try:
            uid = int(user_id_input.strip())
        except ValueError:
            st.warning("User ID must be an integer. Ignoring.")

    with st.spinner("Finding the best movies for you…"):

        # Step 1: find matching seed movies from the query
        matches = search_movies(query, state, max_results=25)

        if not matches:
            st.info("No movies matched your query. Try a different keyword or title.")
            st.stop()

        seed_ids   = [m["movieid"] for m in matches]
        seed_id_set = set(seed_ids)

        # Step 2: expand candidates via content similarity
        content_candidates = content_similar_movies(seed_ids[:5], state, top_k=60)
        # Merge: seeds first, then content-similar, deduplicated
        all_candidates = seed_ids + [c for c in content_candidates if c not in seed_id_set]

        # Step 3: score & rank
        if uid is not None:
            recs = recommend_for_user(uid, all_candidates, state, top_k=top_k)
        else:
            # No user ID → pure content similarity, scored via item popularity bias
            recs = []
            for mid in all_candidates:
                score = float(state["movie_mean"].get(mid, state["global_mean"]))
                recs.append({
                    "movieid": mid,
                    "title"  : state["movieid_to_title"].get(mid, "Unknown"),
                    "score"  : score,
                    "mode"   : "content",
                })
            recs.sort(key=lambda x: x["score"], reverse=True)
            recs = recs[:top_k]

    # ── Results ──────────────────────────────────────────
    st.markdown(f'<div class="section-header">🎥 Top {len(recs)} Recommendations</div>',
                unsafe_allow_html=True)

    # Determine user status
    if uid is not None:
        is_new_user = uid not in state["train_users_set"]
        if is_new_user:
            st.info(
                f"ℹ️ **User {uid}** is new (not in training data). "
                "Cold-start fallback is active — recommendations are based on "
                "item popularity and content similarity."
            )
        else:
            st.success(
                f"✅ **User {uid}** found in training data. "
                f"Using Hybrid model (α = {alpha:.2f})."
            )
    else:
        st.info("ℹ️ No User ID provided — showing popularity-weighted content matches.")

    # Render cards
    for i, rec in enumerate(recs, 1):
        render_movie_card(i, rec["title"], rec["score"], rec["mode"])

    # ── Seed summary ─────────────────────────────────────
    with st.expander("🔎 Matched seed movies from your query"):
        seed_df = pd.DataFrame([
            {"Title": m["title"].title(), "Match score": f"{m['score']:.2f}"}
            for m in matches[:10]
        ])
        st.dataframe(seed_df, use_container_width=True, hide_index=True)

    # ── Mode legend ──────────────────────────────────────
    st.markdown("""
    <div style="margin-top:18px; color:#7c85c9; font-size:0.82rem;">
    <b>Legend:</b> &nbsp;
    <span class="badge badge-hybrid">🔀 Hybrid</span> SVD + Content blend &nbsp;
    <span class="badge badge-coldstart">❄️ Cold-Start</span> Mean-based fallback &nbsp;
    <span class="badge badge-content">📖 Content</span> Content similarity only
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color:#3d4266; font-size:0.8rem;'>"
    "Movie Recommendation System · Hybrid SVD + Content-Based · Cold-Start Aware"
    "</div>",
    unsafe_allow_html=True,
)
