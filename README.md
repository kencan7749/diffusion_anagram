# auto_visual_analgram

A screening pipeline for hybrid images: pictures that read as one subject from
across the room and as another up close, generated with
[Factorized Diffusion](dev/visual_anagrams) on DeepFloyd IF.

**The system does not judge whether an image is interesting.** That judgement
stays with a person. What is automated is discarding candidates where the
illusion did not form at all, so a human looks at a sheet of images that work
rather than a sheet that is mostly failures. The loop optimises *yield*.

Two things come out of a run:

- `contact_sheet.png` — every candidate that held, best first, with both
  prompts and its score. This is where interesting images are harvested.
- `components.md` — a ranking of prompt components with their posteriors. This
  is where knowledge about *which words work* accumulates, across runs.

## What is searched

Only the prompt triple `(prompt_low, prompt_high, style)`. The blur width is
fixed at `sigma = 2.0` and is not a search axis.

## The score

Each candidate is rendered into two perceptual views — blurred (far) and
untouched (near) — and CLIP is asked which prompt each view matches:

```
p_far  = P(prompt_low  | far view)      does it survive blurring?
p_near = P(prompt_high | near view)     is it readable up close?
J      = min(p_far, p_near)
```

`min`, not a mean: an illusion only exists when both views hold. A candidate
that renders one side perfectly and the other not at all is not a near miss.

Candidates are ordered by `sep_min`, the weaker of the two raw CLIP margins.
That is the same ordering `J` gives — `J = sigmoid(logit_scale * sep_min)` — but
it stays legible where `J` has saturated to 0.99.

Nothing is ever deleted on the score's say-so: every candidate is written to
`scores.jsonl` with its full breakdown, so a threshold can be moved and the
figures redrawn without regenerating an image.

## Layout

```
ava/          the pipeline
  spec.py       candidates, verdicts, run state
  engine.py     resident DeepFloyd generator (bit-identical to generate.py)
  perceive.py   the far / near views a human sees
  judge.py      CLIP scoring, BLIP captions as evidence only
  vocab.py      the (word, role) bandit arms and their Beta posteriors
  propose.py    what to try next, and per-component credit assignment
  report.py     contact sheet and component ranking
  loop.py       orchestration (CLI entry point)
ava_vocab/    vocabulary generation from GPT-2 (see its README)
scripts/      one-off analyses
tests/
```

## Running

```bash
.venv/bin/python -m ava.loop --run-id myrun --rounds 5 --k 8
```

Resumable: a stopped run continues from `state.json`. The vocabulary database
at `runs/vocab.db` is shared across runs on purpose — the accumulated estimate
of which prompts work is the asset the system builds.

Grow the vocabulary between runs:

```bash
.venv/bin/python -m ava_vocab.generate_vocab --db runs/vocab.db --seed 1
```

## Verification

```bash
.venv/bin/ruff check ava ava_vocab scripts tests
.venv/bin/mypy
.venv/bin/pytest tests/ -m "not slow" -q
```

`-m slow` additionally runs the metric's regression test, which needs CUDA and
the CLIP weights.
