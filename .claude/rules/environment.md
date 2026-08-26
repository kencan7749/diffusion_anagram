# Environment and Reproducibility

- Execution assumes a uv-managed virtual environment at `.venv/`.
- Agents must not:
  - install packages implicitly (`uv sync`, `pip install`, etc.)
  - rely on global Python
  - rely on system packages
- Prefer top-level imports.
- Avoid runtime ImportErrors.
- Dependencies should fail at import time rather than runtime.