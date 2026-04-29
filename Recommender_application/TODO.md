# Movie Recommender — Production Architecture Implementation Plan

## Goal
Refactor the monolithic Streamlit app into a cloud-deployable architecture:
- **FastAPI backend**: hosts model logic, data loading, hybrid recommendations, cold-start handling
- **Streamlit frontend**: thin client, calls backend API, displays results
- **Deployment docs**: single vs. separate repos, cloud deployment guide

## Steps

- [x] Step 0: Analyze notebook cells 62–67, 72, 74–77 to extract hybrid + cold-start logic
- [x] Step 1: Read existing `movie_recommender_app.py` to understand current UI/UX
- [x] Step 2: Design API contract and folder structure
- [x] Step 3: Create FastAPI backend (`backend/main.py`, `requirements.txt`, `Dockerfile`)
- [x] Step 4: Create Streamlit frontend (`frontend/streamlit_app.py`, `requirements.txt`, `Dockerfile`)
- [x] Step 5: Create `docker-compose.yml` for local orchestration
- [x] Step 6: Create `DEPLOYMENT.md` with architecture decisions and cloud deployment guide
- [x] Step 7: Verify file completeness and provide summary

