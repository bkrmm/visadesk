# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create data directory if it doesn't exist
RUN mkdir -p data

# Expose port
EXPOSE 8000

# Set environment variables
ENV GOOGLE_API_KEY=AIzaSyDCnAFTq5tS3rCrYb7M5jP90IuvitcgFLQ

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]