.PHONY: install install-dev seed run test lint clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

seed:
	python -m acra.data.seed

run:
	python -m acra.main

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

clean:
	rm -rf __pycache__ .pytest_cache chroma_data/ src/acra/__pycache__ tests/__pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
