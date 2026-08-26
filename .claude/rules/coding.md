# Code Style and Structure
- Follow PEP8 strictly.
- Maximum line length: 88.
- Use ruff for linting and formatting.
---
## Static Typing
- Type hints are mandatory.
- Prefer explicit types.
- Use modern syntax:
  - list[str] instead of List[str]
---
## Code Organization
- Single Responsibility Principle
- No script-style logic in modules
- Separate concerns (data / models / training / evaluation)
- Reusable logic must be extracted to utility modules.

### Code Placement (shared vs paper-specific)
- Cross-paper reusable logic must live on `main`:
  - reusable **functions/modules** → `src/laion_revision/...`
  - reusable **CLI entry points** → `scripts/`
- Paper-specific code stays in `analyses/<paper>/` on that paper's branch.
- Never bury cross-paper logic inside a single paper's folder or branch; lift
  it to `src/`/`scripts/` on `main` and import it.

### Separate analysis from figure generation (mandatory)
- **Analysis** (the expensive, deterministic computation: model fitting,
  permutations, PCA, statistics) and **figure generation** (rendering plots from
  results) must be **separate steps that communicate only through persisted
  artifacts** (e.g. JSON / `.npy` / `.npz` / `.csv` under `results/`).
- An analysis run must **persist everything a figure needs** to its result file,
  so figures can be (re)drawn **without re-running the analysis**.
- A figure/plotting step must **only read persisted artifacts** — never refit,
  resample, reload betas, or otherwise recompute the analysis. Changing a plot's
  layout, colors, labels, or style must never trigger recomputation.
- Concrete pattern (see `analyses/huth2012_semantic_space/`):
  - `run_*.py` → analysis → writes `results/*.json` (numbers + everything the
    figures consume).
  - `render_*.py` (and the `analysis/*figures*.py` plot functions) → read that
    JSON → write `results/figures/...`. Fast, idempotent, iterable.
- Rationale: reproducibility (figures are a pure function of saved results),
  fast iteration (seconds, not a refit), and clean separation of concerns. If a
  figure tweak forces a recompute, the split is wrong — fix it by persisting the
  missing inputs in the analysis step.

---

## Dependency Injection
- Dependencies must be passed explicitly.
- Avoid hidden globals.
- Pass configuration, loggers, and resources as parameters.