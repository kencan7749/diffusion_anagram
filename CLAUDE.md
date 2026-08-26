# CLAUDE.md

- This file defines how code agents should operate within this repository.
- General coding standards and behavioral rules are in `.claude/rules/`.

---
## Session Startup (Run at Every Session Start)
1. Read `.claude/tasks/lessons.md` if it exists — apply all lessons before
   touching any code. (Local working state; not tracked in this repo — each
   contributor accumulates their own.)
2. Read `.claude/tasks/todo.md` if it exists — only work on confirmed tasks.
3. Do not begin implementation until any present startup files are reviewed.

---
# Project Goals
- This repository contains research and data-analysis code.
- Generated code must be suitable for:
  - Long-term research projects
  - Re-analysis by third parties
  - Code review and open science contexts
- Code should prioritize clarity, reproducibility, and correctness.

---
# Replication Routine (re:vision)

This repo works through vision/fMRI papers for the **re:vision** initiative:
reproduce each paper's method, then replicate it on the LAION-fMRI dataset
(`laion_fmri`, installed from `external/LAION-fMRI`).

- **Routine**: reproduce-then-replicate, following the principles below.
  Progress is tracked in `analyses/README.md`. (A detailed internal phase
  checklist is kept as local working state and is not part of this repo.)
- **One folder per paper**: each paper is self-contained under
  `analyses/<paper_slug>/` (copy `analyses/_template/`). It holds `proposal.md`
  and `report.md` in re:vision format, paper-specific code in `analysis/`,
  `run_replication.py`, `tests/`, `results/`, and `run_log.md`.
- **Shared primitives**: cross-paper, re:vision-mandated logic lives in
  `src/laion_revision/replication/` (within-subject permutation testing
  in `permutation.py`; generalization splits in `generalization.py`). Import
  these instead of duplicating them per paper. This package follows the
  `__init__.py` policy below (no eager heavy imports; import from submodules).
- **Shared code lives on `main` (iron rule)**: any logic reused across papers
  belongs on `main`, never buried in a paper folder or paper branch. Reusable
  **functions/modules** go in `src/laion_revision/...`; reusable **CLI/scripts**
  go in `scripts/` (e.g. `scripts/build_beta_store.py` builds the canonical beta
  store for any subject). Only paper-specific code lives in `analyses/<paper>/`
  on that paper's branch. If you find yourself copying a helper into a second
  paper, lift it to `src/`/`scripts/` on `main` and import it instead.
- **re:vision constraints**: only 5 subjects → all statistics are
  within-subject (group-level tests become non-parametric permutation tests,
  corrected per subject). OOD images are decided per study (default: excluded
  for replication, since OOD is for generalization) — record the decision and
  reason in the paper's `README.md`.
- **Default routine principles** (apply to every paper):
  1. **Reproduce the original on its own data first** when a public
     implementation/dataset exists (nilearn, PyMVPA, OpenNeuro), then port to
     LAION-fMRI. Fall back to "re-implement + light sanity check" only when the
     original data is hard to obtain.
  2. **Reproduction and replication share the same core analysis functions** —
     implement the analysis once and feed both original-data and LAION-fMRI
     inputs through it; only the data-loading differs.
  3. **Run sub-01 only first**, verify the pipeline, then scale to all 5
     subjects (sub-01 finishes downloading first, so no need to wait for the
     full dataset).

---
# Workflow

## Plan Mode Default
- For any non-trivial task:
    1. Enter plan mode.
    2. Write the plan to `.claude/tasks/todo.md` based on the template in `.claude/tasks/todo_template.md`.
    3. Wait for user confirmation before implementation.
- Tasks are considered non-trivial if they involve:
    - architectural changes
    - multi-step modifications
    - changes across multiple modules

---
## Task Lifecycle
Every task follows this lifecycle:
1. **Plan first** — Write plan to `.claude/tasks/todo.md` with checkable items.
2. **Verify plan** — User confirms before implementation.
3. **Track progress** — Mark items complete as you go.
4. **Explain changes** — Provide high-level explanation after each step.
5. **Document results** — Add review section when task is completed.
6. **Capture lessons** — Add lessons to `.claude/tasks/lessons.md`.

---
## Verification Before Done
- A task is not complete until:
    - `.venv/bin/ruff check src/ tests/` passes
    - `.venv/bin/mypy src/` passes
    - tests pass: `.venv/bin/pytest tests/ -m "not slow" --tb=short -q`
    - functionality is demonstrated
- Always verify: logs, outputs, CI results.
- If verification reveals a bug NOT caused by the current task:
    - Report it to the user. Do not fix it independently.
- Ask: "Would a staff engineer approve this change?"

---
## Version Control Workflow

- Assume Git-based workflow.
- Guidelines:
    - small, frequent commits with clear messages
    - use feature branches
    - each logical change in a separate commit
    - prefer squash-merge for PRs to keep main history linear
    - avoid large rewrites unless requested
- Changes must be: reviewable, revertible, independently testable.

---
## Context Management
- At approximately 50% of the context window, pause and ask the user to `/compact`.
- Summarize the current state of `.claude/tasks/todo.md` before compacting.
- Do not attempt to complete large multi-file tasks in a single context window.

---
## Development Environment

- Execution assumes a **uv-managed virtual environment** at `.venv/`.
- Use `.venv/bin/` executables directly — **never `uv run`**:
    ```
    .venv/bin/python   .venv/bin/pytest
    .venv/bin/ruff     .venv/bin/mypy
    ```
- Agents must not install packages (`uv sync`, `uv add`, `pip install`).
    - After editing `pyproject.toml`: tell the user which sync command to run manually.
- Do not run `nox` sessions (they rebuild environments and take several minutes).
    - After changes, instruct the user: "Run `nox -s <session>` to verify."
- Full details: `.claude/rules/environment.md`

---
## Research Constraints

- This repository supports scientific research.
- Agents must ensure: reproducibility and experiment traceability.
- Random seeds must be fixed whenever randomness is used.
- Hyperparameters and model versions must be recorded.

---
## `__init__.py` Policy

`__init__.py` files must follow these principles:

1. **Keep them lightweight.** Avoid eagerly importing modules that pull
   in heavy dependencies (e.g. `torch`, `umap-learn`, `scipy`, `pot`) or
   optional extras (`dreamsim`, `torchcodec`, `laion_clap`, etc.). A user
   running `import laion_revision.<subpackage>` should not pay the cost of every
   downstream module being imported.
2. **Avoid side effects at import time.** No network I/O, no filesystem
   writes, no environment-variable mutation, no logger configuration.
3. **Avoid circular imports.** When in doubt, prefer importing from the
   exact submodule (`from laion_revision.embedders.base import BaseEmbedder`)
   rather than re-exporting through `__init__.py`.

When `__init__.py` does re-export names, the re-exports must come from
modules that themselves obey rule (1). The regression test in
`tests/test_lazy_imports.py` enforces this for the top-level packages.

---
# Agent Responsibility
- The human developer is responsible for the overall design.
- Agents must assist but **must not steer architecture decisions**.
- Agents should refuse tasks that violate rules in `.claude/rules/`.
