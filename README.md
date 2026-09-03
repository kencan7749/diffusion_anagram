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
role `subject`. The *words* are pooled across tasks: a prompt that only
`negate` knows (a paper example, a generated word) is offered to `flip` at the
uniform prior and registered under `flip` when first proposed there
(`source = pooled`). The *evidence* is not pooled; that is the point of the
key. Seeds come from two traceable sources: strings that occur in
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
TASKS=hybrid scripts/run_evolve.sh   # any task(s) with v2, then search_summary and clips; ROUNDS, K, SEED, ANIMATE=held|all|none, ... as env vars
scripts/run_flip_evolve.sh           # the same for flip, named flip_evo_<date>
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

## The audio flip

The same search runs on the time-reversal anagram (`ava/audio/`): one sound
that reads as one thing played forwards and another played backwards, sampled
from Stable Audio Open with the Visual Anagrams construction and time reversal
as the view, and scored by CLAP in both directions. The task `time_reverse` is
registered beside the papers' tasks, its vocabulary is authored envelope
prompts (decays and swells; there is no paper to quote from), and
`ava.audio.loop` shares the proposers, the run layout and the reports with
the image loop. The listening list is `audition.md`.

```bash
scripts/run_audio_flip_evolve.sh     # v2 search of the audio flip; ~12 s per candidate
.venv/bin/python -m ava.audio.loop --run-id audio --rounds 6 -k 6 --proposer bandit
BACKEND=audioldm2 scripts/run_audio_flip_evolve.sh   # the same search in AudioLDM 2's latent
```

The anagram can be sampled in either of two latents, chosen by `--backend`
(a run setting, recorded in `config.yaml`; the default is `stable_audio`).
Stable Audio Open's latent is a waveform codec's, a stack of 1D convolutions
whose learned kernels do not commute with reversal; Step 0b measured the
flipped latent against the latent of the reversed audio at a cosine of 0.70
on a percussive probe. AudioLDM 2 (`cvssp/audioldm2`, 16 kHz mono) denoises
the latent of a log-mel spectrogram, and a symmetric-window magnitude
spectrogram reverses exactly with the audio up to its frame grid, so there
only the 2D VAE can fail to commute. `ava/audio/mel.py` rebuilds the mel
front end diffusers does not ship (checked by playing it through the
vocoder: `scripts.smoke_audioldm2`), `AudioLDM2Codec` puts it behind the same
codec interface, and `ava/audio/engine_audioldm2.py` runs the same sampler on
the (B, 8, T/4, 16) latent with time as the second-to-last axis.

Step 0b on both codecs (`results/step0b/`, `results/step0b_audioldm2/`;
`headroom` is the flipped latent's decode against the reversed audio, minus
the codec's own round-trip error, and `lat_cos` compares the flipped latent
with the latent of the reversed audio directly):

| probe | Stable Audio headroom | lat_cos | AudioLDM 2 headroom | lat_cos |
|---|---|---|---|---|
| percussive burst | +0.63 dB | 0.70 | +0.31 dB | 0.99 |
| rising chirp | +1.11 dB | 0.80 | -0.42 dB | 1.00 |
| generated clips (3) | +0.22 to +4.72 dB | 0.44 to 0.75 | +0.11 to +0.61 dB | 0.94 to 0.98 |

```bash
.venv/bin/python -m scripts.smoke_audioldm2                        # load, vocoder round trip, three clips
.venv/bin/python -m scripts.step0b_view_validity --codec audioldm2 # is the flip the reversal, in this latent?
.venv/bin/python -m scripts.step1_first_anagram --backend audioldm2
```

### Two more audio illusions

The audio loop is task-driven (`--task`): the task names its slots, its
perceptual views (`ava/audio/perceive.py`, a registry keyed by view name),
the files a candidate is written to (`<view>.wav`) and the columns of
`audition.md`.

**The frequency hybrid** (`freq_hybrid_750`, AudioLDM 2 only) is Factorized
Diffusion's hybrid image with a low-pass in place of the blur: slot `near` is
heard in the room, slot `far` through a wall (below 750 Hz). Hearing has no
"close up means high frequencies" mechanism the way vision does, so the near
view is the whole signal, not its high band. The split has to be a map on the
latent, and zeroing the latent's frequency rows is not one (it decodes to
something further from a low-passed signal than the untouched latent). What
works is a per-frame affine map fitted by least squares from
`(Enc(x), Enc(lowpass(x)))` pairs (`ava/audio/bands.py`); Step 0c fits it,
checks it on held-out clips and persists it with the digests of the clips it
came from. The projector is data, so every run names it (`--projector`) and
records its digest in `config.yaml`.

| held-out clip (750 Hz) | measured | null | floor | energy above 750 Hz |
|---|---|---|---|---|
| percussive burst | 4.7 dB | 25.3 | 3.9 | -29 dB |
| AudioLDM 2 "match being struck" | 4.9 dB | 14.6 | 3.0 | -32 dB |
| Stable Audio "hammer on anvil" | 7.6 dB | 28.1 | 5.9 | -38 dB |

Step 1 on three authored pairs held 6/6 at a median J of 0.90. A control
with the same seeds and sampler but the near prompt in both slots shows
where the hybrid does the work: the low band of plain "rain" does not read
as "thunder" (margin -0.01) but the hybrid's does (+0.17, +0.11), whereas
low-passed "sizzling" already reads as "a truck idling" on its own.

**The time jigsaw** (`time_jigsaw_4`, either backend) is Visual Anagrams'
jigsaw with time in place of the plane: slot `whole` is the recording as
made, slot `shuffled` the same recording cut into four equal blocks and
spliced in a fixed order (`ava/audio/permute.py`, seed 0, no block stays
put). The engines return exactly the span the latent view acted on, so the
listener's cuts and the latent's fall at the same instants. Step 0b with
`--view jigsaw_4` reads better than the flip on the waveform codec (latent
cosine 0.96 to 0.99 against 0.44 to 0.80): a block permutation only breaks
the codec's convolutions at three boundaries.

**The time mosaic** (`time_mosaic_40ms`, AudioLDM 2 only) is the jigsaw at
the finest cut the codec can follow: every latent frame (40 ms) moves, under
a permutation drawn from the frame count and a fixed seed, and the
listener's view cuts the waveform into the same 40 ms blocks with a 5 ms
fade at each cut. The granularity was chosen by measurement, not taste: CLAP
does not hear a shuffle of blocks longer than about 150 ms (cosine to the
original 0.90, as for reversal) and hears 40 ms blocks clearly (0.69), while
the waveform codec cannot follow cuts finer than about 200 ms (+8 dB over
its floor at 93 ms) and the mel codec follows 40 ms cuts within 1 dB on
generated clips once the cuts are faded. A 40 ms shuffle turns anything
structured -- speech, a melody, a rhythm -- into a grainy texture and leaves
a texture a texture, so the vocabulary pairs the two.

```bash
.venv/bin/python -m scripts.step0c_freq_view_validity --cutoff 750        # fit + persist the projector
.venv/bin/python -m scripts.step1_first_anagram --task freq_hybrid_750 --backend audioldm2 \
    --projector results/step0c_freq_view/projector_750hz --guidance-scale 3.5 --steps 200
TASK=freq_hybrid_750 BACKEND=audioldm2 PROJECTOR=results/step0c_freq_view/projector_750hz \
    scripts/run_audio_flip_evolve.sh
.venv/bin/python -m scripts.step0b_view_validity --view jigsaw_4           # the jigsaw on Stable Audio
TASK=time_jigsaw_4 scripts/run_audio_flip_evolve.sh
TASK=time_mosaic_40ms BACKEND=audioldm2 GUIDANCE=3.5 STEPS=200 scripts/run_audio_flip_evolve.sh
```

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
  webui/        the run viewer (Flask; reads persisted files only)
  search/       the v2 evolutionary proposer (archive, operators, novelty,
                racing, surrogate, evolve); torch-free
  image/        the multi-view illusion track
    tasks.py      the task registry (pure data)
    views.py      builds the upstream view objects; perceptual views per task
    perceive.py   blur / grayscale / motion blur / VA transform of a picture
    engine.py     resident DeepFloyd generator, both papers' samplers
    judge.py      CLIP scoring, BLIP captions as evidence only
    animate.py    transition clips via the upstream animate.py
    paper_examples.py  prompts quoted from the papers, with figure citations
  audio/        the audio illusions (tasks: time_reverse, freq_hybrid_750,
                time_jigsaw_4, time_mosaic_40ms; vocab, loop: the search; engine, engine_audioldm2:
                the samplers on Stable Audio's and AudioLDM 2's latents; codec,
                mel: the autoencoders behind one interface; bands: the fitted
                low-pass on the latent; permute: the jigsaw's cut; judge,
                perceive: CLAP and the listener's views)
ava_vocab/    vocabulary generation from GPT-2 (see its README)
scripts/      one-off analyses, and webui.py to watch runs in the browser
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

## Watching a run

```bash
.venv/bin/python -m scripts.webui --runs runs --port 8765
```

then open <http://127.0.0.1:8765/> (from a laptop: `ssh -L 8765:127.0.0.1:8765 host`).
The viewer reads only what a run persisted — `config.yaml`, `state.json`,
`round_*/scores.jsonl`, and for an evolve run `archive.jsonl`,
`search_state.json` and `clusters.npz` — so it can be started beside a search
that is still going; the page polls every five seconds and follows new rounds.
Nothing is regenerated, re-judged or refitted, and no model is loaded.

Tabs, modelled on ShinkaEvolve's WebUI:

| tab | what it shows | from |
|---|---|---|
| Overview | the run's numbers, the best candidate, the config | `config.yaml`, `scores.jsonl` |
| Rounds | best / mean `sep_min`, held rate, archive coverage, surrogate rho, operator reward, duplicate rejection, per round | `scores.jsonl`, `search_state.json` |
| Candidates | every scored candidate with its views (or wavs), filterable and sortable; click for the N×N CLIP matrix and captions | `scores.jsonl`, `view_<slot>.png` |
| Lineage | pairs as nodes, `parents → child` as edges labelled by operator; colour is fitness, a gold ring is an elite | `archive.jsonl` |
| Archive | the MAP-Elites grid per task, cluster × cluster, with the elite of each cell | `archive.jsonl`, `clusters.npz` |
| Compare | a child beside its parents, changed slots highlighted, with the improvement | `archive.jsonl` |

**Transition clips.** The upstream `animate.py` turns a candidate into the
clip the papers show: the plain view with its prompt, an eased transition
into the other view, that prompt, and back. One clip per non-plain view is
written beside the stills as `anim_<slot>.mp4` (`ava/image/animate.py`), from
the persisted `sample_256.png` and the run's rebuilt view objects — nothing
is regenerated. Pre-render a run:

```bash
.venv/bin/python -m scripts.animate_run --run runs/flip_evo_20260902 --held-only
```

or click *render transition clip* on a candidate in the viewer (or *render
clips for held* above the candidate grid); a worker thread renders one clip at
a time, about 2.5 s each, and the page picks them up. The *clips* toggle
switches between clips and stills. The triple hybrid has no `make_frame`
upstream and stays as stills.

The code is `ava/webui/` (Flask application factory, `data.py` for the
reading, one JSON endpoint per function, `jobs.py` for the clip queue, a
single page with no JavaScript dependencies). `flask --app ava.webui run`
serves the same app.

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
