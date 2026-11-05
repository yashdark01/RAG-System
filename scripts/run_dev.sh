#!/usr/bin/env bash
set -e

# ==============================================================================
# RAG System Development Server Runner
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 Starting RAG System in development mode..."

# Check for .env file
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "⚠️  No .env file found. Copying from .env.example..."
        cp .env.example .env
    else
        echo "⚠️  No .env or .env.example found. Creating minimal .env..."
        cat <<EOF > .env
APP_ENV=development
DEBUG=true
PORT=8000
HOST=0.0.0.0
EOF
    fi
fi

# Create required directories
mkdir -p data/vector_store data/uploads logs

# Check for uv
if command -v uv &> /dev/null; then
    echo "📦 Using uv package manager..."
    exec uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
else
    echo "📦 Using python / uvicorn..."
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
fi

