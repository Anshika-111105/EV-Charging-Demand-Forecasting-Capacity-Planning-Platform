FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml first for caching
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy source code and configs
COPY src/ src/
COPY configs/ configs/
COPY pipelines/ pipelines/
COPY app/ app/
COPY Makefile .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p data reports models mlruns && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8501

CMD ["uvicorn", "src.ev_forecasting.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
