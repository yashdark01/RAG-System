#!/bin/bash

echo "🚀 Setting up RAG System with uv..."

# Check if uv is installed
if command -v uv &> /dev/null; then
    echo "⚡ 'uv' detected! Setting up ultra-fast virtual environment..."
    uv venv .venv
    source .venv/bin/activate
    uv pip install -e ".[dev]"
else
    echo "📦 'uv' not found, falling back to standard python venv & pip..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Create runtime directories
mkdir -p data/vector_store data/uploads logs

# Copy environment file
if [ ! -f .env ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys (GOOGLE_API_KEY, TAVILY_API_KEY, etc.)"
fi

echo "✅ Setup complete!"
echo ""
echo "To run with uv directly (no manual activation needed):"
echo "  uv run uvicorn app.main:app --reload"
echo ""
echo "Or with activated virtual environment:"
echo "  source .venv/bin/activate"
echo "  uvicorn app.main:app --reload"
echo ""
echo "Documentation will be available at: http://localhost:8000/docs"
