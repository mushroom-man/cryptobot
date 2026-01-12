# CryptoBot Platform - Python Application
#
# Build: docker build -t cryptobot .
# Run:   docker run cryptobot

FROM python:3.12.4-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY cryptobot/ ./cryptobot/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create directories for runtime data
RUN mkdir -p /app/logs /app/models /app/data/csv

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "-m", "cryptobot.main"]
