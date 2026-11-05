#!/usr/bin/env bash
set -e

# ==============================================================================
# Docker Build & Smoke Test Script
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

IMAGE_NAME="rag-backend"
TAG="${1:-latest}"

echo "🐳 Building Docker image '${IMAGE_NAME}:${TAG}'..."
docker build -t "${IMAGE_NAME}:${TAG}" .

echo "✅ Docker image built successfully!"
echo "Run with: docker run -p 8000:8000 ${IMAGE_NAME}:${TAG}"

