# Backend Dockerfile — FastAPI Recommender Service
FROM python:3.11-slim

WORKDIR /app

# Install system dependancies if needed 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

#use renders dynamic port for cloud deploy
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]

# The dataset is mounted at runtime or copied in via build context.
# For cloud deploy, ensure final_dataset.csv is in the build context.

# fix this copy command to not fail if the file is not present, since it will be mounted at runtime in cloud deploy
#COPY final_dataset.csv . 2>/dev/null || echo "Dataset will be mounted at runtime"
#COPY final_dataset.csv . 2>/dev/null || echo "Dataset will be mounted at runtime"

# Expose FastAPI port (uncomment if running locally)
#EXPOSE 8000

# Run the app(if running locally uncomment the EXPOSE command above)
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

