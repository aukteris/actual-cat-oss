.PHONY: lint test check

lint:
	venv/bin/ruff check actual_cat/ tests/ scripts/
	venv/bin/mypy actual_cat/

test:
	venv/bin/pytest -q

check: lint test
