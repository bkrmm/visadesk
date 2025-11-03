# Use Python 3.10 slim image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffering output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (needed for faiss, numpy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip wheel setuptools
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Create data directory if it doesn't exist
RUN mkdir -p /app/data

# Expose the FastAPI port
EXPOSE 8000

# Avoid hardcoding API keys inside the image
# (Instead, set GOOGLE_API_KEY in Cloud Run > Variables & Secrets)
ENV GOOGLE_API_KEY="AIzaSyDCnAFTq5tS3rCrYb7M5jP90IuvitcgFLQ"

# Start the FastAPI app using Uvicorn (production config)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
