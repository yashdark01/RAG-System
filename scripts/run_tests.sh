#!/usr/bin/env bash
set -e

# ==============================================================================
# RAG System Test Suite Runner
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🧪 Running RAG System Test Suite..."

# Export test environment variables
export APP_ENV=development
export DEBUG=false
export DATABASE_URL="sqlite+aiosqlite:///:memory:"
export RATE_LIMIT_ENABLED=false

if command -v uv &> /dev/null; then
    uv run pytest tests/ -v "$@"
else
    pytest tests/ -v "$@"
fi

echo "✅ All tests completed successfully!"

