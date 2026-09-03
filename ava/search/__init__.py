"""Search v2: quality-diversity evolution with racing and a surrogate front end.

The v1 proposer (`ava.propose.BanditProposer`) treats a candidate as a sum of
independent components. This package treats it as a *pair*: the unit that is
archived, raced across seeds and predicted is the whole candidate, and the
component posteriors of v1 are kept as a knowledge layer underneath.

Nothing here imports torch. Text embeddings enter as a callable
(`ava.search.embed.Embedder`), which in a real run is the judge's CLIP, so the
proposer and the judge share one model without this package paying for it.

  embed.py       the embedding callable and a per-string cache
  archive.py     MAP-Elites archive: cells over semantic clusters, per task
  operators.py   the six mutation operators and the UCB1 bandit over them
  novelty.py     embedding-cosine rejection of near-duplicate children
  racing.py      per-pair Beta posteriors and the seed-promotion rule
  surrogate.py   numpy Gaussian process with a leave-one-out skill check
  evolve.py      EvolutionaryProposer: one round, step by step
"""
