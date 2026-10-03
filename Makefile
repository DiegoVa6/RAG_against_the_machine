.PHONY: install run debug clean lint lint-strict

all: run

run: install
# uv run python src/chunker.py
	uv run python -m src

install: 
	uv sync

debug: install
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache

lint: install
	uv run python -m flake8 src
	uv run python -m mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: install
	python3 -m flake8 src
	python3 -m mypy src --strict
