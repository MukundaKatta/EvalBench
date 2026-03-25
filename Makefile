.PHONY: install dev test lint format clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -v --tb=short

test-cov:
	pytest --cov=evalbench --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
