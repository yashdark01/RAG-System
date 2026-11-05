#!/usr/bin/env bash
set -e

# ==============================================================================
# Code Quality & Linting Script
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🔍 Running code quality checks with ruff..."

if command -v uv &> /dev/null; then
    uv run ruff check app/ tests/ scripts/
else
    ruff check app/ tests/ scripts/
fi

echo "✅ All linting checks passed!"

