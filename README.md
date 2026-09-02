# auto_visual_analgram

A screening pipeline for multi-view optical illusions: one picture that reads
as one subject in one view and as another in a second (or third, or fourth)
view, generated with [Visual Anagrams and Factorized
Diffusion](dev/visual_anagrams) on DeepFloyd IF.

**The system does not judge whether an image is interesting.** That judgement
stays with a person. What is automated is discarding candidates where the
illusion did not form at all, so a human looks at a sheet of images that work
rather than a sheet that is mostly failures. The loop optimises *yield*.

Two things come out of a run:

- `contact_sheet.png` — every candidate that held, best first, with every view
  the judge looked at, its prompts and its score. This is where interesting
  images are harvested.
- `components.md` — a ranking of prompt components per task and role, with
  their posteriors. This is where knowledge about *which words work in which
  view* accumulates, across runs.

## Tasks

Every example type in the two papers is a *task* (`ava/image/tasks.py`). A
task names its views, which paper's mechanism it uses, and one prompt *slot*
per view.

| paper | tasks | mechanism |
|---|---|---|
| Visual Anagrams (Geng et al. 2024) | `flip` `rotate_cw` `rotate_ccw` `rotate_180` `skew` `jigsaw` `inner_circle` `negate` `patch_permute` `pixel_permute` `square_hinge` `three_view` `four_view` | the view transforms the noisy image, so it must be orthogonal (a permutation or a sign flip); noise estimates are averaged |
| Factorized Diffusion (Geng et al. 2025) | `hybrid` `triple_hybrid` `color_hybrid` `motion_hybrid` `inverse_hybrid` | the view extracts one linear component of the noise estimate; the components sum to the identity, so estimates are summed |

Not registered, on purpose: the ambigram (CLIP cannot read cursive well enough
to judge it), the spatial-mask and scaling decompositions (not illusions a
judge can score), and the latent-space orthogonal transform (needs Stable
Diffusion). View parameters (blur sigma, patch grid, ...) are the upstream
defaults and are not search axes; they are recorded in every run's
`config.yaml`.

## What is searched

A candidate is `(task, one prompt per slot, style)`. A style is a prefix
(`"an oil painting of"`) or a template with a placeholder
(`"oil painting style, {}"`). For `inverse_hybrid` the first slot is pinned to a
reference image and only the other slot is searched.

Vocabulary arms are `(word, task, role)`. Factorized Diffusion slots are
asymmetric (`low` / `high`, `gray` / `color`, `moving` / `still`), so each has
its own role; Visual Anagrams slots are symmetric, so every slot shares the
role `subject`. Seeds come from two traceable sources: strings that occur in
the upstream checkout (`author`), and prompts quoted verbatim from the papers
with a figure citation (`paper`, `ava/image/paper_examples.py`). Paper seeds
start from a uniform Beta(1, 1); the data decides.

## The score

Each candidate is rendered into the N views a person sees — the transformed
picture for Visual Anagrams, a blurred / grayscale / motion-blurred picture for
Factorized Diffusion — and CLIP is asked which of the N prompts each view
matches:

```
S[i][j] = cos(CLIP(view_i(x)), CLIP(prompt_j))
p_i     = softmax_j(S[i, :])[i]          does view i read as its own prompt?
J       = min_i p_i
```

`min`, not a mean: an illusion only exists when every view holds. A candidate
that renders one view perfectly and another not at all is not a near miss.

Candidates are ordered by `sep_min`, the weakest view's raw CLIP margin over
its runner-up prompt. With two views that is the same ordering `J` gives —
`J = sigmoid(logit_scale * sep_min)` — but it stays legible where `J` has
saturated to 0.99. With three or more views the two orderings can differ; `J`
decides pass/fail, `sep_min` carries the order. The papers' own alignment
(`min diag S`) and concealment (`tr softmax(S/τ)`, both directions) scores are
recorded alongside for comparison.

Nothing is ever deleted on the score's say-so: every candidate is written to
`scores.jsonl` with its full N×N matrix, so a threshold can be moved and the
figures redrawn without regenerating an image.

## Two searches

`--proposer bandit` (v1, the default) treats a candidate as a sum of
independent components: Thompson sampling over the `(word, task, role)`
posteriors, a fixed quota of never-tried words, and targeted swaps of the slot
that failed. Its limitation is that what works is the *pair*, and a component
bandit cannot represent that.

`--proposer evolve` (v2, `ava/search/`) searches over pairs while keeping the
v1 posteriors as its knowledge layer:

- **Archive (MAP-Elites).** One elite per cell, a cell being the task plus the
  semantic cluster of each prompt (k-means over CLIP text embeddings of the
  vocabulary, seeded and persisted as `clusters.npz`). Diversity is a property
  of the archive, not a term in the objective. Everything ever evaluated stays
  in `archive.jsonl`.
- **Operators.** `swap_slot`, `crossover`, `restyle`, `transpose`,
  `transfer_task` (same pair, another view) and `inject` (a never-tried word),
  chosen by a UCB1 bandit rewarded by the child's improvement over its parent.
  Replacement words come from the v1 posteriors.
- **Duplicate rejection.** A child whose prompts are within cosine `eta` of an
  evaluated pair of the same task and style is dropped before it costs a
  generation. The distribution `eta` was applied to is recorded.
- **Racing.** Each pair carries `Beta(held + 1, failed + 1)`; a share of every
  round re-seeds the pairs whose lower quartile of success rate is still at or
  above one half, up to `--max-seeds`. This replaces the fixed harvest.
- **Surrogate.** A numpy Gaussian process over the embeddings, slot cosines,
  task and posteriors ranks the children by UCB, but only while its
  leave-one-out Spearman rho over recent rounds clears a threshold. Otherwise
  it steps aside and selection is uniform; its predictions are recorded
  either way, in `scores.jsonl` under `extra`.

Nothing in `ava/search/` imports torch; embeddings are injected, and in a run
they are the judge's own CLIP. A run is reproducible from its seed byte for
byte (`archive.jsonl` included), and resumes from its persisted state.

```bash
scripts/run_flip_evolve.sh      # flip with v2; ROUNDS, K, SEED, RACE_FRACTION, ... as env vars
.venv/bin/python -m ava.loop --run-id evo --tasks flip,hybrid,jigsaw --rounds 12 --k 8 \
    --proposer evolve --harvest-seeds 0 --harvest-top 0
.venv/bin/python -m scripts.search_summary --run runs/evo   # what the archive learned
scripts/run_ab_search.sh        # v1 vs v2 at equal budget -> results/stepB/ab.md
```

A v2 run leaves `archive.jsonl` (every pair with every seed, elite flag per
cell), `clusters.npz` (the cell map) and `search_state.json` (operator
statistics, the surrogate's skill log, duplicate-rejection statistics, a
per-round summary) beside the usual files; `search_summary.md` reads them
back as tables.

v1 stays the default. Whether v2 should replace it is a question for
measurement, not for the design: `scripts/ab_compare.py` applies the rule
from the design note (yield at least v1's, more cells held) to two runs'
files, and a v2 run's own `archive.jsonl` and `search_state.json` (coverage,
surrogate skill per round, operator rewards) say how it is doing in use.

## Layout

```
ava/          the pipeline
  spec.py       candidates, verdicts, run state, fixed view parameters
  metric.py     J = min(every view holds), shared by both tracks
  rankstats.py  Spearman's rho and the ROC AUC, for checking predictions
  vocab.py      the (word, task, role) bandit arms and their Beta posteriors
  propose.py    what to try next (v1 bandit), and per-slot credit assignment
  report.py     contact sheet and component ranking
  loop.py       orchestration (CLI entry point)
  search/       the v2 evolutionary proposer (archive, operators, novelty,
                racing, surrogate, evolve); torch-free
  image/        the multi-view illusion track
    tasks.py      the task registry (pure data)
    views.py      builds the upstream view objects; perceptual views per task
    perceive.py   blur / grayscale / motion blur / VA transform of a picture
    engine.py     resident DeepFloyd generator, both papers' samplers
    judge.py      CLIP scoring, BLIP captions as evidence only
    paper_examples.py  prompts quoted from the papers, with figure citations
  audio/        the time-reversal anagram track
ava_vocab/    vocabulary generation from GPT-2 (see its README)
scripts/      one-off analyses
tests/
```

## Running

```bash
.venv/bin/python -m ava.loop --run-id myrun --tasks hybrid,flip,jigsaw --rounds 5 --k 8
```

`--tasks` takes any comma-separated subset of the registry; tasks are visited
round-robin within a round. `inverse_hybrid` additionally needs `--ref-image`
and `--ref-prompt` (what the reference shows, for the judge).

Resumable: a stopped run continues from `state.json`, and a randomised view's
permutation is reloaded from `runs/<run>/views/` rather than redrawn. The
vocabulary database at `runs/vocab.db` is shared across runs on purpose — the
accumulated estimate of which prompts work is the asset the system builds.
Databases from before the task split are migrated in place, posteriors intact.

Grow the vocabulary between runs:

```bash
.venv/bin/python -m ava_vocab.generate_vocab --db runs/vocab.db --seed 1 --arm flip:subject --arm hybrid:low
```

## Verification

```bash
.venv/bin/ruff check ava ava_vocab scripts tests
.venv/bin/mypy
.venv/bin/pytest tests/ -m "not slow" -q
```

`-m slow` additionally runs the metric's regression test, which needs CUDA and
the CLIP weights. `scripts/stepA_paper_examples.py` generates one paper example
per task on the GPU and checks that each task's own view reads best.
`scripts/stepB_fidelity.py` checks whether a cheaper generation (15 steps, or
the 64 px stage) predicts whether the full one holds, and
`scripts/run_ab_search.sh` runs the v1/v2 comparison; both write only under
`results/stepB/`, and `scripts/render_stepB.py` draws from those files.
