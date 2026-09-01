Run the full local verification sequence in order:
1. `.venv/bin/ruff format --check src/ tests/`
2. `.venv/bin/ruff check src/ tests/`
3. `.venv/bin/mypy src/`
4. `.venv/bin/pytest tests/ -m "not slow" --tb=short -q`

Report the result of each step. Stop and explain on the first failure.
