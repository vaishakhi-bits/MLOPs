# Build stage
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt xgboost

# Runtime stage
FROM python:3.9-slim

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code (excluding files in .dockerignore)
COPY . .

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# Create necessary directories
RUN mkdir -p /app/notebooks/saved_models

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
