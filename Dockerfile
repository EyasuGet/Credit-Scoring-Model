# Use a lightweight Python base image
FROM python:3.12-slim-bullseye

# Set the working directory in the container
# WORKDIR /app
WORKDIR /home/eyuleo/Documents/kifya/Credit-Scoring-Model

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: Don't cache pip packages, reduces image size
# --upgrade pip: Ensure pip is up-to-date
# --default-timeout=100: Increase timeout for potentially slow package downloads
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --default-timeout=100

# Copy the entire project directory into the container
# This includes src/, tests/, data/ (though data/ will be empty in production image if .gitignore is respected)
# and mlruns/ (which contains the registered models)
COPY . .

# Set environment variables for MLflow tracking
# This ensures the containerized application knows where to find the mlruns data
# relative to its own /app directory.
# ENV MLFLOW_TRACKING_URI="file:///app/mlruns"
# ENV PYTHONPATH="/app" 
ENV MLFLOW_TRACKING_URI="file:///home/eyuleo/Documents/kifya/Credit-Scoring-Model/mlruns"
ENV PYTHONPATH="/home/eyuleo/Documents/kifya/Credit-Scoring-Model"

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn
# The --host 0.0.0.0 is crucial for allowing external connections to the container
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

