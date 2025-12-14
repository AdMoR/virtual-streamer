# Virtual Streamer Makefile
# Common commands for development

.PHONY: help tests test lint format install clean

help:
	@echo "Available commands:"
	@echo "  make tests      - Run all tests"
	@echo "  make test       - Alias for 'make tests'"
	@echo "  make test-v     - Run tests with verbose output"
	@echo "  make lint       - Run linter (ruff)"
	@echo "  make format     - Format code with ruff"
	@echo "  make install    - Install dependencies with uv"
	@echo "  make clean      - Remove cache and build artifacts"

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

