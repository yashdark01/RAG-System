FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv from official binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml requirements.txt ./

# Install dependencies using uv into system environment
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY . .

# Ensure data directories exist
RUN mkdir -p data/vector_store data/uploads logs

# Expose application port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
