# Virtual Streamer Makefile
# Common commands for development

.PHONY: help tests test lint format install clean \
        docker-login docker-build-virtual_streamer_api \
        docker-push-virtual_streamer_api docker-release-virtual_streamer_api

# Docker variables
REGISTRY := ghcr.io/admor
IMAGE_WORKER := $(REGISTRY)/virtual_streamer_api

help:
	@echo "Available commands:"
	@echo "  make tests      - Run all tests"
	@echo "  make test       - Alias for 'make tests'"
	@echo "  make test-v     - Run tests with verbose output"
	@echo "  make lint       - Run linter (ruff)"
	@echo "  make format     - Format code with ruff"
	@echo "  make install    - Install dependencies with uv"
	@echo "  make clean      - Remove cache and build artifacts"
	@echo ""
	@echo "Docker commands:"
	@echo "  make docker-login                    - Login to GitHub Container Registry"
	@echo "  make docker-build-virtual_streamer_api - Build virtual_streamer_api image"
	@echo "  make docker-push-virtual_streamer_api  - Push virtual_streamer_api image"
	@echo "  make docker-release-virtual_streamer_api - Build and push in one command"

# Run all tests
tests:
	uv run python -m pytest tests/ -v

# Alias for tests
test: tests

# Run tests with verbose output and show local variables on failure
test-v:
	uv run python -m pytest tests/ -v --tb=long -l

# Run a specific test file (usage: make test-file FILE=tests/test_stateful_callbacks.py)
test-file:
	uv run python -m pytest $(FILE) -v

# Run linter
lint:
	uv run ruff check .

# Format code
format:
	uv run ruff format .

# Install dependencies
install:
	uv sync

# Clean cache and build artifacts
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ============================================================================
# Docker Commands
# ============================================================================

# Login to GitHub Container Registry
docker-login:
	docker login ghcr.io

# Build virtual_streamer_api image
docker-build-virtual_streamer_api:
	docker build -t $(IMAGE_WORKER):latest -f docker/docker_worker/Dockerfile .

# Push virtual_streamer_api image
docker-push-virtual_streamer_api:
	docker push $(IMAGE_WORKER):latest

# Build and push in one command
docker-release-virtual_streamer_api: docker-build-virtual_streamer_api docker-push-virtual_streamer_api

adk-server:
	@adk web ./virtual_streamer/agents --port 8005
