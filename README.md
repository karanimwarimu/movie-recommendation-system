# 🎬 Movie Recommendation System

A production-ready hybrid movie recommendation system that delivers top  personalized movie suggestions to users based on their preferences. The system combines collaborative filtering with content-based techniques, handles cold-start scenarios gracefully, and is packaged for cloud deployment with a modern microservices architecture.



> **Live Dashboard:** [Tableau Storytelling Insights — Movie Recommendation & Analysis](https://public.tableau.com/views/movierecommendation/Story1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Technologies Used](#-technologies-used)
- [System Architecture](#-system-architecture)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Development with Docker](#option-1-docker-recommended)
  - [Local Development without Docker](#option-2-direct-python)
- [How to Use](#-how-to-use)
- [API Reference](#-api-reference)
- [Model Performance](#-model-performance)
- [Deployment](#-deployment)
- [Future Enhancements](#-future-enhancements)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Problem Statement

Streaming platforms contain thousands of movies, making it difficult for users to quickly find content they will enjoy. Traditional recommendation systems rely on limited user–item interactions and struggle with the cold start problem, producing generic and less relevant suggestions.

This project develops a hybrid recommendation system that combines collaborative and content-based techniques to generate accurate personalized  movie recommendations in both data-rich and data-sparse scenarios.

---


## 🎯 Project Overview

This project solves the classic information overload problem on streaming platforms: users face thousands of movies and struggle to find content they enjoy. Additionally, new users and newly added movies often lack sufficient rating history, creating the **cold-start problem** that reduces the effectiveness of traditional recommendation systems.

### What It Does

- **Personalized Recommendations:** Suggests  top  movies tailored to each user's taste
- **Multi-Input Search:** Accepts movie titles, keywords, genres, years, or descriptive clues
- **Hybrid Intelligence:** Blends collaborative filtering (user behavior) with content-based filtering (movie metadata)
- **Cold-Start Resilience:** Gracefully handles new users and new movies using intelligent fallbacks
- **Interactive UI:** Clean, responsive Streamlit interface with real-time backend health monitoring
- **Cloud-Ready:** Containerized services deployable to Render, Streamlit Cloud, or any Docker host

### Key Features

| Feature | Description |
|---------|-------------|
| 🔀 Hybrid Scoring | Combines SVD matrix factorization with TF-IDF content similarity |
| ❄️ Cold-Start Handling | Automatic fallback to global, user, or movie mean ratings |
| 🔍 Smart Search | Ranked multi-field search across titles, genres, keywords, and overviews |
| ⚡ Optimized Inference | Pre-trained artifact loading for sub-second response times |
| 📊 Live Metrics | Real-time display of dataset stats, model health, and recommendation mode |
| 🐳 Docker Orchestration | One-command local deployment with `docker-compose` |

---

## 📁 Repository Structure

```
.
├── README.md                              # Project documentation
├── LICENSE                                # Project license
├── .gitignore                             # Git ignore rules
│
├── Data sets/                             # Raw and processed datasets
│   └── (MovieLens + TMDB datasets)
│
├── Project Proposal/                      # Project planning documents
│   ├── Movie Recommendation System – Phase 4 Proposal.pdf
│   └── Phase 4 Project Proposal.docx
│
|___ presenattion.pdf # non technical presenation
|
├── Data_cleaning.ipynb                    # Data cleaning & preprocessing notebook
├── data_analysis_and_visualisation.ipynb  # EDA & visualization notebook
├── models.ipynb                           # Model training & evaluation notebook ( used as the basis fo creating the application)
│
└── Recommender_application/               # Production application
    ├── docker-compose.yml                 # Local orchestration (backend + frontend)
    ├── requirements.txt                   # Root-level dependencies
    ├── final_dataset.csv.gz              # Compressed production dataset
    ├── TRAIN_SAVE_OPTIMIZED.PY           # Offline model training & artifact generation
    ├── TODO.md                            # Implementation checklist
    ├── DEPLOYMENT.md                      # Detailed deployment & architecture guide
    │
    ├── backend/                           # FastAPI recommendation service
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   ├── main.py                        # Standard backend (trains on startup)
    │   └── main_optimized.py             # Optimized backend (loads pre-trained artifacts)
    │
    └── frontend/                          # Streamlit user interface
        ├── Dockerfile
        ├── requirements.txt
        ├── streamlit_app.py              # Standard frontend
        └── streamlit_app_pro.py          # Production frontend (artifact-aware)
```

### File Descriptions

| Path | Purpose |
|------|---------|
| `Data_cleaning.ipynb` | Merges MovieLens and TMDB datasets, handles missing values, standardizes text, and engineers the final feature set |
| `models.ipynb` | Trains and evaluates SVD, Content-Based, and Hybrid models; performs hyperparameter tuning (alpha grid search) and cold-start analysis (basis for the application build)|
| `TRAIN_SAVE_OPTIMIZED.PY` | One-time offline script that trains models and serializes compact inference artifacts using `joblib` |
| `main.py` | FastAPI backend that trains SVD + Content-Based models on startup (good for experimentation) |
| `main_optimized.py` | FastAPI backend that loads pre-trained artifacts for instant inference (production recommended) |
| `streamlit_app.py` / `streamlit_app_pro.py` | Streamlit frontends that call the backend API and render recommendation cards with custom CSS |
| `DEPLOYMENT.md` | Comprehensive deployment guide with architecture diagrams, environment setup, and security checklists |

---

## 🛠️ Technologies Used

### Core Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Backend** | Python 3.11 + FastAPI | High-performance REST API for model serving |
| **Frontend** | Python 3.11 + Streamlit | Interactive, low-code web interface |
| **Containerization** | Docker + Docker Compose | Reproducible local and cloud deployment |
| **Data Science** | pandas, NumPy, scikit-learn | Data manipulation, model training, and evaluation |
| **Serialization** | joblib | Efficient storage of pre-trained model artifacts |
| **Vectorization** | TF-IDF (scikit-learn) | Text feature extraction from movie metadata |
| **Dimensionality Reduction** | TruncatedSVD | Latent factor extraction for collaborative and content filtering |
| **Similarity** | Cosine Similarity (dot-product on normalized embeddings) | Content-based movie matching |

### Data Sources

- **MovieLens Dataset** — User ratings (`userId`, `movieId`, `rating`)
- **TMDB (The Movie Database) Dataset** — Movie metadata (`title`, `genres`, `overview`, `keywords`, `tagline`, `release_year`)

### Visualization

- **Tableau Public** — Interactive storytelling dashboard for movie performance, user preferences, and recommendation pattern analysis
- **matplotlib + seaborn** (notebooks) — Model comparison charts and EDA plots

---

## 🏗️ System Architecture

The system follows a clean **microservices architecture** that separates model logic from presentation, enabling independent scaling and deployment.

### High-Level Flow

```
┌─────────────────┐      HTTP/JSON       ┌──────────────────┐
│   Streamlit UI  │  ◄────────────────►  │   FastAPI Backend │
│  (Thin Client)  │                      │  (Model + Data)   │
│   Port: 8501    │                      │   Port: 8000      │
└─────────────────┘                      └──────────────────┘
        │                                         │
   Deployed to:                            Deployed to:
   Streamlit Cloud                         Render / Railway / Fly.io
```

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| **Streamlit Frontend** | Captures user input (`query`, `user_id`, `top_k`), calls backend API, renders styled recommendation cards, displays user status and system health |
| **FastAPI Backend** | Loads dataset & artifacts, serves `/recommend`, `/search`, `/stats`, `/health`, handles cold-start logic, tunes and stores the optimal hybrid `alpha` |
| **Artifact Store** (Cloudflare R2 / Local) | Hosts serialized `joblib` artifacts so the backend can skip training on startup |

### Request Lifecycle

1. **User submits a query** (e.g., "toy story") and optional `user_id` via the Streamlit UI
2. **Frontend POSTs to `/recommend`** with the query payload
3. **Backend executes the pipeline:**
   - Search: Finds seed movies matching the query across titles, genres, keywords, and overviews
   - Expansion: Generates content-similar candidates using cosine similarity on TF-IDF embeddings
   - Scoring: Dispatches to SVD, Content-Based, or Hybrid prediction based on user/movie familiarity
   - Fallback: Uses cold-start strategy (global mean, user mean, or movie mean) when data is sparse
4. **Backend returns JSON** with ranked recommendations, seed matches, user status, and `best_alpha`
5. **Frontend renders** interactive movie cards with predicted scores, mode badges, and visual score bars




### Production Evolution Roadmap

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│  Streamlit  │────►│  Cloudflare │────►│  FastAPI (xN)   │
│    Cloud    │     │   / Nginx   │     │  (auto-scaled)  │
└─────────────┘     └─────────────┘     └─────────────────┘
                                                │
                                          ┌─────┴─────┐
                                          ▼           ▼
                                    ┌─────────┐  ┌────────┐
                                    │  Redis  │  │  S3/   │
                                    │  Cache  │  │ GCS    │
                                    └─────────┘  │ (data) │
                                                 └────────┘
```

For full deployment instructions, environment variables, and security checklists, see [`Recommender_application/DEPLOYMENT.md`](Recommender_application/DEPLOYMENT.md).

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+ (for direct execution)
- Docker + Docker Compose (for containerized execution)
- Git

### Option 1: Docker (Recommended)

From the `Recommender_application` directory:

```bash
cd Recommender_application
docker-compose up --build
```

- **Backend API:** http://localhost:8000
- **Frontend UI:** http://localhost:8501
- **API Docs (Swagger):** http://localhost:8000/docs

The frontend waits for the backend health check to pass before starting.

### Option 2: Direct Python

**Step 1 — Start the Backend**

```bash
cd Recommender_application/backend
pip install -r requirements.txt

# Standard version (trains on startup; takes 30–90s)
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# OR — Optimized version (loads artifacts instantly)
python -m uvicorn main_optimized:app --host 0.0.0.0 --port 8000 --reload
```

**Step 2 — Start the Frontend** (in a new terminal)

```bash
cd Recommender_application/frontend
pip install -r requirements.txt

# Windows Command Prompt
set BACKEND_URL=http://localhost:8000

# Windows PowerShell
$env:BACKEND_URL="http://localhost:8000"

# Mac / Linux
export BACKEND_URL=http://localhost:8000

streamlit run streamlit_app_pro.py
```

- **Frontend UI:** http://localhost:8501

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_URL` | `http://localhost:8000` | FastAPI endpoint used by the Streamlit frontend |
| `DATASET_PATH` | `final_dataset.csv` | Path to the movie ratings dataset |
| `ARTIFACTS_BASE_URL` | — | Cloudflare R2 base URL for remote artifact loading |
| `ARTIFACTS_DIR` | `/tmp/artifacts` | Local directory for cached artifacts |

---

## 🎮 How to Use

### Web Interface

1. **Open the UI** in your browser [(https://movie-recommendation-system-kvvu9kbkkvtcrjucrnpw8r.streamlit.app)]

2. **Enter a search query** in the search box:
   - Movie title: `toy story`
   - Release year: `1994`
   - Genre or keyword: `space adventure`, `romantic comedy`
   - Descriptive clue: `dinosaurs park`

   ![Search Input Placeholder](docs/images/placeholder_search_input.png)

3. **(Optional) Enter a User ID** to enable personalized recommendations

   ![User ID Placeholder](docs/images/placeholder_user_id.png)

4. **Click "Get Recommendations"** and wait for the backend response

5. **Review your recommendations:**
   - Each movie card shows the predicted score (out of 5.00)
   - A color-coded badge indicates which model mode was used:
     - 🔀 **Hybrid** — SVD + Content blend (known user & movie)
     - ❄️ **Cold-Start** — Mean-based fallback (new user or movie)
     - 📖 **Content** — Popularity-weighted content match (anonymous user)

6. **Expand "Matched seed movies"** to see which movies matched your query

### HOW TO USE THE APPLICATION
 * https://scribehow.com/viewer/How_To_Get_Movie_Recommendations_Using_The_Streamlit_App__JWcStoTsTrCTjU0bFMyPvg

---

### API Direct Usage

```bash
# Health check
curl http://localhost:8000/health

# Dataset stats
curl http://localhost:8000/stats

# Search movies
curl "http://localhost:8000/search?query=space&max_results=10"

# Get recommendations
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "query": "toy story",
    "user_id": 123,
    "top_k": 5
  }'
```

---

## 📡 API Reference

### `POST /recommend`

Generate personalized movie recommendations.

**Request Body:**
```json
{
  "query": "toy story",
  "user_id": 123,
  "top_k": 8
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "movieid": 862,
      "title": "toy story",
      "score": 4.5123,
      "mode": "hybrid"
    }
  ],
  "seed_matches": [
    {"movieid": 862, "title": "toy story", "score": 1.0}
  ],
  "user_status": "existing",
  "best_alpha": 0.65
}
```

### `GET /search`

Search movies by title, keyword, genre, year, or clue.

### `GET /stats`

Return dataset and model statistics (`n_users`, `n_movies`, `n_ratings`, `best_alpha`).

### `GET /health`

Service health check.

---

## 📊 Model Performance

The hybrid model was evaluated on a held-out test set using both regression and ranking metrics. The optimal blending parameter `alpha` was determined via grid search on a validation set.

### Final Model Comparison

| Model | RMSE ↓ | MAE ↓ | Precision@5 ↑ | Recall@5 ↑ | HitRate@5 ↑ | NDCG@5 ↑ |
|-------|--------|-------|---------------|------------|-------------|----------|
| **SVD Collaborative Filtering** | 0.3495 | 0.1000 | 0.9163 | 0.6076 | 1.0000 | 0.9896 |
| **Content-Based Filtering** | 0.9091 | 0.7122 | 0.5197 | 0.3202 | 0.8117 | 0.6853 |
| **Untuned Hybrid (α=0.7)** | 0.4203 | 0.2792 | 0.9183 | 0.6085 | 1.0000 | 0.9900 |
| **Tuned Hybrid (α=1.0)** | 0.3495 | 0.1000 | 0.9163 | 0.6076 | 1.0000 | 0.9896 |

**Key Takeaway:** The SVD component dominates predictive accuracy, while the content-based layer provides critical coverage for cold-start and low-data scenarios. The system dynamically selects the best strategy per user-movie pair.

### Cold-Start Performance

| Scenario | Fallback Strategy | RMSE |
|----------|-------------------|------|
| New user, known movie | Movie mean rating | ~0.99 |
| Known user, new movie | User mean rating | ~0.99 |
| Both unknown | Global mean rating | ~0.99 |

---

## 🌐 Deployment

The application is designed for easy cloud deployment:

- **Backend:** Deploy `backend/` to [Render](https://render.com), [Railway](https://railway.app), or [Fly.io](https://fly.io)
- **Frontend:** Deploy `frontend/` to [Streamlit Cloud](https://streamlit.io/cloud)
- **Artifacts:** Host pre-trained `joblib` files on Cloudflare R2, AWS S3, or GCS for instant backend startup

For step-by-step deployment instructions, architecture diagrams, scaling recommendations, and security hardening, refer to the dedicated [`DEPLOYMENT.md`](Recommender_application/DEPLOYMENT.md).

---

## 🔮 Future Enhancements

- **Redis Caching:** Cache frequent queries and user embeddings to reduce latency under load
- **Auto-Scaling:** Deploy multiple FastAPI replicas behind a load balancer (Nginx / Cloudflare)
- **Authentication:** Add OAuth2 or API-key middleware for public deployments
- **Retraining Pipeline:** Schedule nightly/weekly jobs to retrain on new ratings and hot-swap artifacts
- **Deep Learning:** Experiment with neural collaborative filtering (NCF) or transformer-based recommenders
- **A/B Testing Framework:** Serve multiple model variants and track engagement metrics

---

## 🛠️ Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Backend fails to start | Dataset or artifacts not found | Ensure `final_dataset.csv` is present or `ARTIFACTS_BASE_URL` is configured |
| Frontend shows "Backend unreachable" | `BACKEND_URL` is incorrect or backend is sleeping | Verify the URL; wake the backend if on a free tier |
| Recommendations are empty | Query has no matches | Try broader keywords or check dataset contents |
| Cold-start always triggers | User ID not in training data | Expected behavior — the system falls back gracefully |
| Slow first request | Model loading on startup | Normal for `main.py`; use `main_optimized.py` for instant load |

---

## 📄 License

This project is licensed under the terms provided in the [`LICENSE`](LICENSE) file.

---

*Built with Python, FastAPI, Streamlit, and scikit-learn. Data sourced from MovieLens and TMDB.*

