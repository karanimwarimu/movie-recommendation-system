# Movie Recommender — Deployment & Architecture Guide

## 1. Architecture Overview

```
┌─────────────────┐      HTTP/JSON       ┌──────────────────┐
│   Streamlit UI  │  ◄────────────────►  │   FastAPI Backend │
│  (Thin Client)  │                      │  (Model + Data)   │----- Cloudflare R datasetstorage(stores the )                                    ↓
│   Port: 8501    │                      │   Port: 8000      │
└─────────────────┘                      └──────────────────┘
        │                                         │
        │                                         │
   Deploy to:                              Deployed to:
   Streamlit Cloud                         Render 
```

### Responsibilities

| Layer | Responsibility |
|-------|---------------|
| **Frontend (Streamlit)** | Capture user input (`query`, `user_id`, `top_k`), call backend API, render recommendation cards, display user status |
| **Backend (FastAPI)** | Load dataset, train SVD + Content-Based models, tune `alpha`, serve `/recommend`, `/search`, `/stats`, `/health`, handle cold-start logic |

---

## 2.  Repository overview

### Single Repository (for this project)

**Structure:**
```
the rest of the repository/
|______recommender apk
|
├── backend/          # FastAPI service
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/         # Streamlit app
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
├── DEPLOYMENT.md
└── artefacts   #this is for local tests on deployment itll be fetched from cloudflare storage
```

**Advantages:**
- **Version synchronization**: Model changes and UI updates stay in lockstep.
- **Shared data artifacts**: One dataset file, one source of truth.
- **Simpler CI/CD**: One pipeline can test integration end-to-end.
- **Easier onboarding**: New developers clone one repo and run `docker-compose up`.

---

## 3. Local Development

### Option A: Docker (Recommended if Docker is installed)

**Prerequisites:**
- Docker + Docker Compose installed
- artefacts present in the project root(recommernder apk)

```bash
cd e:/DSF/flatiron/phase4   # or your project root
docker-compose up --build
```

- **Backend API**: http://localhost:8000
- **Frontend UI**: http://localhost:8501
- **API Docs (Swagger)**: http://localhost:8000/docs

### Option B: Direct Python (No Docker required)

If Docker is unavailable or you prefer running natively:

**Step 1 — Install dependencies and start the backend**

```bash
# From the project root directory (e:/DSF/flatiron/phase4/movie deploy)
pip install -r backend/requirements.txt

# The backend auto-detects artefacts in the parent directory
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend will:
- Search for `artefacts` in the current directory, parent directory, or grandparent directory
- Load the artefacts, train SVD + Content-Based models, and tune alpha automatically
- Print `✅ Models loaded. best_alpha=0.XX` when ready

**Step 2 — In a new terminal, start the frontend**

```bash
# From the project root directory
pip install -r frontend/requirements.txt

# Set the backend URL and launch Streamlit
set BACKEND_URL=http://localhost:8000
streamlit run frontend/streamlit_app.py
```

- **Frontend UI**: http://localhost:8501

**Notes:**
- On Windows, use `set BACKEND_URL=http://localhost:8000` (Command Prompt) or `$env:BACKEND_URL="http://localhost:8000"` (PowerShell)
- On Mac/Linux, use `export BACKEND_URL=http://localhost:8000`
- The first backend startup may take 30–90 seconds while models train in memory

### Run Backend Only (for API testing)

```bash
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload
```

Then visit http://localhost:8000/docs for the interactive Swagger UI.

### Run Frontend Only (pointing to a deployed backend)

```bash
pip install -r frontend/requirements.txt
set BACKEND_URL=https://your-deployed-backend.com
streamlit run frontend/streamlit_app.py
```

---

## 4. API Reference

### `POST /recommend`
Generate personalised movie recommendations.

**Request body:**
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

**Modes explained:**
| Mode | Meaning |
|------|---------|
| `hybrid` | SVD + Content-Based blend (user & movie both known) |
| `cold_user` | New user → falls back to item mean rating |
| `cold_movie` | New movie → falls back to user mean rating |
| `cold_both` | Both unknown → falls back to global mean |
| `content` | Anonymous user → content similarity + popularity |

### `GET /search?query=space&max_results=20`
Search movies by title, keyword, genre, year, or clue.

### `GET /stats`
Dataset statistics: users, movies, ratings, best alpha.

### `GET /health`
Service health check.

---

## 6. Scaling & Production Considerations

### Current Architecture Limits
- **Backend loads everything into memory** on startup. This is fast for inference but limits horizontal scaling.
- **No caching layer**: Every request recomputes predictions. For high traffic, add Redis to cache user embeddings or recommendation results.
- **No authentication**: The API is open. Add OAuth2 or API-key middleware if exposing publicly.

### Recommended Production Evolution

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

1. **Add a load balancer** (Nginx, Cloudflare, AWS ALB) in front of multiple FastAPI replicas.
2. **Move dataset to object storage** (S3, GCS) and download on container startup instead of baking into the image.
3. **Add Redis** for caching frequent queries and user embeddings.
4. **Add request logging & monitoring** (Prometheus + Grafana, or Datadog).
5. **Model retraining pipeline**: Schedule a nightly/weekly job to retrain on new data and hot-swap the model artifact.

---

## 8. Security Checklist

- [ ] Enable HTTPS on both frontend and backend (handled automatically by most cloud platforms).
- [ ] Add CORS restrictions in FastAPI (`allow_origins=["https://your-streamlit-app.streamlit.app"]`).
- [ ] Add rate limiting on `/recommend` to prevent abuse.
- [ ] Never commit secrets or API keys to Git; use platform secret managers.
- [ ] Validate `user_id` and `top_k` ranges (already enforced by Pydantic schemas).

---

## 9. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Backend fails to start | artefacts not found | Ensure they are saved or loaded fro cloudflare
| Frontend shows "Backend unreachable" | `BACKEND_URL` incorrect or backend sleeping | Check URL, wake backend, or use paid tier |
| Recommendations are empty | Query has no matches | Try broader keywords or check dataset contents |
| Cold-start always triggers | User ID not in training data | This is expected; system falls back gracefully |
| Slow first request | Model loading on startup | Normal; subsequent requests are fast |

---

## 10. Summary

- **FastAPI backend** handles all data science logic; **Streamlit frontend** is a pure presentation layer.
- **Deploy backend** to Render/Railway/Fly.io; **deploy frontend** to Streamlit Cloud.
- **Connect them** via the `BACKEND_URL` environment variable.
- **Evolve** by adding caching, load balancing, and object storage as traffic grows.



