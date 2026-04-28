"""
Movie Recommender — Streamlit Frontend (Pro Version)
=====================================================
Designed to work with backend/main_pro.py (artifact-loading backend).

No model training happens in this file or in the backend on startup.
The backend loads pre-trained joblib artifacts from the artifacts/ directory.

Environment variable: BACKEND_URL (default: http://localhost:8000) --- for testing purposes and remote deployment (point to your cloud backend URL).

How to run the full stack:
--------------------------
1. Generate artifacts (one-time):
       cd .. && python train_and_save.py

2. Start the backend (loads artifacts, no training):
       cd ../backend
       uvicorn main_pro:app --host 0.0.0.0 --port 8000

3. Start this frontend (in a new terminal):
       cd ../frontend
       streamlit run streamlit_app_pro.py

Or use docker-compose if configured.
"""

import os
from typing import Any

import requests
import streamlit as st

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
RATING_MAX = 5.0

st.set_page_config(
    page_title="🎬 Movie Recommender Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    [data-testid="stSidebar"] { background-color: #1a1d27; }

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
    .badge-hybrid    { background:#2e3a78; color:#a0acff; }
    .badge-coldstart { background:#3a2020; color:#ff9e9e; }
    .badge-content   { background:#1e3a2a; color:#7effd4; }

    .metric-tile {
        background: #1a1d27;
        border: 1px solid #2c3050;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-label { font-size: 0.78rem; color: #7c85c9; margin-bottom: 2px; }
    .metric-value { font-size: 1.4rem; font-weight: 700; color: #c9d1ff; }

    .hero {
        background: linear-gradient(135deg, #1c1f35 0%, #252840 100%);
        border-radius: 16px;
        padding: 28px 32px 22px;
        margin-bottom: 24px;
        border: 1px solid #3d4266;
    }
    .hero h1 { margin: 0; font-size: 2rem; color: #c9d1ff; }
    .hero p  { color: #7c85c9; margin: 6px 0 0; font-size: 0.95rem; }

    .section-header {
        font-size: 1rem; font-weight: 600; color: #9ca3f5;
        border-bottom: 1px solid #2c3050; padding-bottom: 6px;
        margin: 18px 0 12px;
    }

    label { color: #b0b8e8 !important; }

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

    [data-testid="stSpinner"] { color: #9ca3f5 !important; }
    .stAlert { border-radius: 10px; }

    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background: #1e2130 !important;
        color: #e0e0e0 !important;
        border: 1px solid #3d4266 !important;
        border-radius: 8px !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def fetch_stats() -> dict[str, Any] | None:
    try:
        resp = requests.get(f"{BACKEND_URL}/stats", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=60)
def fetch_health() -> dict[str, Any] | None:
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def fetch_recommendations(query: str, user_id: int | None, top_k: int) -> dict[str, Any] | None:
    payload = {"query": query, "user_id": user_id, "top_k": top_k}
    try:
        resp = requests.post(f"{BACKEND_URL}/recommend", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Is main_pro.py running?")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Backend error: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None


# ─────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────
MODE_LABELS = {
    "hybrid": ("🔀 Hybrid", "badge-hybrid"),
    "cold_user": ("❄️ Cold-Start", "badge-coldstart"),
    "cold_movie": ("❄️ Cold-Start", "badge-coldstart"),
    "cold_both": ("❄️ Cold-Start", "badge-coldstart"),
    "content": ("📖 Content", "badge-content"),
}


def render_movie_card(rank: int, title: str, score: float, mode: str) -> None:
    label_txt, badge_cls = MODE_LABELS.get(mode, ("🔀 Hybrid", "badge-hybrid"))
    fill_pct = int((score / RATING_MAX) * 100)
    st.markdown(
        f"""
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
    """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️  Settings")
    top_k = st.slider("Number of recommendations", min_value=3, max_value=20, value=8)
    backend_input = st.text_input(
        "Backend URL",
        value=BACKEND_URL,
        help="FastAPI backend address (main_pro.py). Change if deploying remotely.",
    )
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        """
**Architecture (Pro):**
- 🤝 FastAPI backend (`main_pro.py`) loads pre-trained artifacts
- 🎨 Streamlit frontend (thin client)
- 🔀 Hybrid SVD + Content-Based
- ❄️ Cold-Start aware
- ⚡ Zero training on startup — instant load

The backend loads serialized models via `joblib`.
Run `python train_and_save.py` once to generate artifacts.
"""
    )

# Override BACKEND_URL if user changed it in sidebar
if backend_input.strip():
    BACKEND_URL = backend_input.strip().rstrip("/")

# ─────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
    <h1>🎬 Movie Recommendation System <span style="font-size:0.6em; color:#6c77e8;">PRO</span></h1>
    <p>Artifact-loaded Hybrid SVD + Content-Based engine with cold-start handling —
       search by title, keyword, genre, year, or any clue.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────
# BACKEND HEALTH CHECK
# ─────────────────────────────────────────────────────────
health = fetch_health()
if health and health.get("models_loaded"):
    st.success("✅ Backend connected — models loaded from artifacts.")
elif health:
    st.warning("⚠️ Backend connected but models not yet loaded.")
else:
    st.error(
        "❌ Backend unreachable. Please start `main_pro.py` first:\n\n"
        "```bash\ncd backend && uvicorn main_pro:app --host 0.0.0.0 --port 8000\n```"
    )

# ─────────────────────────────────────────────────────────
# QUICK STATS (from backend)
# ─────────────────────────────────────────────────────────
stats = fetch_stats()
if stats:
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in [
        (c1, "👤 Users", f"{stats['n_users']:,}"),
        (c2, "🎬 Movies", f"{stats['n_movies']:,}"),
        (c3, "⭐ Ratings", f"{stats['n_ratings']:,}"),
        (c4, "🎛️ Best α", f"{stats['best_alpha']:.2f}"),
    ]:
        col.markdown(
            f"""
        <div class="metric-tile">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
        </div>""",
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)
else:
    st.warning("⚠️ Backend stats unavailable. Check that main_pro.py is running.")

# ─────────────────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-header">🔍 Search & Recommend</div>',
    unsafe_allow_html=True,
)

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

    uid = None
    if user_id_input.strip():
        try:
            uid = int(user_id_input.strip())
        except ValueError:
            st.warning("User ID must be an integer. Ignoring.")

    with st.spinner("Finding the best movies for you…"):
        result = fetch_recommendations(query.strip(), uid, top_k)

    if result is None:
        st.stop()

    recs = result.get("recommendations", [])
    seed_matches = result.get("seed_matches", [])
    user_status = result.get("user_status", "anonymous")
    best_alpha = result.get("best_alpha", 0.7)

    if not recs:
        st.info("No movies matched your query. Try a different keyword or title.")
        st.stop()

    # ── Results header ───────────────────────────────────
    st.markdown(
        f'<div class="section-header">🎥 Top {len(recs)} Recommendations</div>',
        unsafe_allow_html=True,
    )

    # User status banner
    if uid is not None:
        if user_status == "new":
            st.info(
                f"ℹ️ **User {uid}** is new (not in training data). "
                "Cold-start fallback is active — recommendations are based on "
                "item popularity and content similarity."
            )
        else:
            st.success(
                f"✅ **User {uid}** found in training data. "
                f"Using Hybrid model (α = {best_alpha:.2f})."
            )
    else:
        st.info(
            "ℹ️ No User ID provided — showing popularity-weighted content matches."
        )

    # Render cards
    for i, rec in enumerate(recs, 1):
        render_movie_card(i, rec["title"], rec["score"], rec["mode"])

    # ── Seed summary ─────────────────────────────────────
    if seed_matches:
        with st.expander("🔎 Matched seed movies from your query"):
            import pandas as pd

            seed_df = pd.DataFrame(
                [
                    {"Title": m["title"].title(), "Match score": f"{m['score']:.2f}"}
                    for m in seed_matches
                ]
            )
            st.dataframe(seed_df, use_container_width=True, hide_index=True)

    # ── Mode legend ──────────────────────────────────────
    st.markdown(
        """
    <div style="margin-top:18px; color:#7c85c9; font-size:0.82rem;">
    <b>Legend:</b> &nbsp;
    <span class="badge badge-hybrid">🔀 Hybrid</span> SVD + Content blend &nbsp;
    <span class="badge badge-coldstart">❄️ Cold-Start</span> Mean-based fallback &nbsp;
    <span class="badge badge-content">📖 Content</span> Content similarity only
    </div>
    """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color:#3d4266; font-size:0.8rem;'>"
    "Movie Recommendation System Pro · Artifact-Loaded Backend · Hybrid SVD + Content-Based · Cold-Start Aware"
    "</div>",
    unsafe_allow_html=True,
)

