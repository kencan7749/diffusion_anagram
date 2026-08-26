# ava_vocab — vocabulary generation

Samples list continuations from GPT-2 and inserts the resulting subjects into
`runs/vocab.db` as untried arms. The search loop picks them up through its
new-word injection quota.

**No separate environment.** GPT-2 runs under the `transformers 4.35.2` that
`diffusers 0.24.0` pins, so this uses the main `.venv`. The directory stays
separate only so the model can be swapped without touching `ava/`; the script
does not import `ava`, and the loop only ever reads `vocab.db`. Inference is on
CPU by default so it does not contend with DeepFloyd for the GPU.

## Use

The database must already exist — running the loop once seeds the schema and
the author vocabulary:

```bash
.venv/bin/python -m ava_vocab.generate_vocab --db runs/vocab.db --draws 40
```

`--dry-run` prints the words without touching the database.

## Why this reaches where caption mining cannot

This is the only supply route that gets outside the vocabulary the system
already has. Caption mining reads captions of images the system generated from
prompts it already knew, so it fills in neighbours of existing words — useful,
but it cannot introduce a subject nobody has tried.

That is also why decoding is **sampled, not greedy**. Greedy returns the argmax,
so the same prompt gives the same continuation forever and a second invocation
adds nothing. With greedy decoding the reachable vocabulary would be fixed at
the first draw.

To accumulate, **vary `--seed`**:

```bash
for seed in 0 1 2 3; do
  .venv/bin/python -m ava_vocab.generate_vocab --db runs/vocab.db --seed "$seed"
done
```

Measured: `--draws 30 --seed 0` produced 64 words and added 122 arms; repeating
it added 0; `--seed 1` added a further 118.

`--greedy` reproduces a deterministic run and is refused together with
`--draws > 1`, which would repeat one identical continuation.

## Continuation, not instruction

An instruct model could be asked for "subjects recognisable from their
silhouette", but that role judgement is not trusted anyway — `vocab.py`
registers both `(word, 'low')` and `(word, 'high')` and lets Thompson sampling
decide, because the upstream readme says intuition about which prompts work is
less reliable than search. What is actually needed is noun phrases people
really depict, which is what a plain language model continuing a list produces.

Three details were found by running it, not by reasoning:

- **No trailing space in the prompt.** GPT-2's BPE attaches a leading space to
  the following token, so a prompt ending in `" "` tokenises to an orphan `Ġ`
  and continuations collapse into fragments (`ichthyosis`, `ileo-moe`).
- **A list, not free continuation.** Free continuation never says where the
  subject ends; it runs into prose, leaving `an extinct species called` or bare
  adjectives like `an unbroken`. In a comma-separated list the comma is the
  terminator, every item is already a noun phrase, and one forward pass yields
  several words.
- **Exemplars must rotate.** A fixed few-shot prefix anchors every draw on the
  same handful of subjects. They are drawn from the visual_anagrams seed
  vocabulary so the conditioning introduces nothing the project has not already
  committed to.

## Filtering

GPT-2 is unfiltered web text. `extract_phrase` rejects sentence fragments
(`and an ancient`), definite phrases (`the same name`), possessive riffs
(`a duck's feathers`, `a duck's tongue`, …), bare adjectives (`a black`),
positional words (`the background`), the medium itself (`an oil painting`), and
a blocklist of violent and sexual terms — one such continuation appeared in the
first trial run. Anything reaching the database is eventually rendered as an
image, so it is refused at the source rather than left for a human to catch on
a contact sheet. Plain age words are deliberately **not** blocked: `a boy` is an
ordinary painting subject, and every harmful combination carries one of the
blocked terms.

Proper nouns are **kept**. The seed vocabulary contains `albert einstein` and
`marilyn monroe`, and the upstream readme calls faces the best subjects to hide,
so rejecting names would discard exactly the category reported to work best.

## Reproducibility

`SamplingConfig.seed_for(prompt, draw)` derives a distinct seed per prompt and
draw, so adding draws or changing the exemplar set does not shift the seeds of
earlier ones and break the provenance of words already stored. Every draw
appends its model id, prompt, seed, temperature, top-p and usable-word count to
`runs/vocab_generation_log.jsonl`. The `arm` table records only `source='llm'`;
that log is what ties a word back to the call that made it.

## What lands in the database

Words arrive with an uninformative `Beta(1, 1)` prior and `n_trials = 0`, under
**both** roles. No role is inferred: list continuation carries no signal about
spatial frequency, and a guess would not be trusted if it did.

Insertion is `INSERT OR IGNORE`, so a word already carrying trial evidence is
never reset to the prior.

`source='llm'` lets this route's hit rate be measured against `author` and
`mined` in `components.md`.

## Cost of a new word

Every new arm is guaranteed a slot in the loop's injection quota (25% of a
round), so it consumes GPU time. The 64 words above became 122 arms, which is
roughly 122 screening generations — about an hour. Prefer several modest draws
over one enormous one.
