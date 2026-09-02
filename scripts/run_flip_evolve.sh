#!/usr/bin/env bash
# Search flip illusions with the v2 evolutionary proposer (ava/search).
#
# The v2 counterpart of run_flip_search.sh. Runs ava.loop on the `flip` task
# with --proposer evolve: a MAP-Elites archive over (flip, cluster, cluster)
# cells, six mutation operators under a UCB1 bandit, embedding-cosine
# duplicate rejection, racing (extra seeds only for pairs that keep holding)
# and a surrogate that ranks children only while its skill check passes.
#
# Harvest is off: racing already re-seeds the pairs that hold, inside the
# round budget, up to MAX_SEEDS per pair. Everything lands under
# runs/<RUN_ID>/ and the run is resumable: the same RUN_ID continues from
# state.json, archive.jsonl and search_state.json and adds ROUNDS more rounds.
#
# Budget at the defaults: 8 rounds x 8 candidates = 64 generations, of which
# about 30% are racing seeds of pairs that held. Roughly 30 s each on the
# RTX 4090 with BLIP captions, so about 35 minutes.
#
#   scripts/run_flip_evolve.sh                        # defaults
#   ROUNDS=12 K=8 scripts/run_flip_evolve.sh          # longer
#   RACE_FRACTION=0.5 scripts/run_flip_evolve.sh      # spend more on re-seeding
#   RUN_ID=flip_evo_a scripts/run_flip_evolve.sh      # a separate run directory
#
# Afterwards, `scripts/search_summary.py` prints what the search learned:
# which cells hold an elite, how many seeds each survived, whether the
# surrogate ever earned its keep, and which operators produced improvements.
#
# Extra arguments are passed straight to ava.loop (see `-m ava.loop --help`).
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ID="${RUN_ID:-flip_evo_$(date +%Y%m%d)}"
ROUNDS="${ROUNDS:-8}"
K="${K:-8}"
SEED="${SEED:-0}"
CLUSTERS="${CLUSTERS:-8}"
ETA="${ETA:-0.95}"
RACE_FRACTION="${RACE_FRACTION:-0.3}"
MAX_SEEDS="${MAX_SEEDS:-4}"

mkdir -p runs
LOG="runs/${RUN_ID}.log"
echo "[run_flip_evolve] run_id=${RUN_ID} rounds=${ROUNDS} k=${K} seed=${SEED}" \
     "clusters=${CLUSTERS} eta=${ETA} race_fraction=${RACE_FRACTION}" \
     "max_seeds=${MAX_SEEDS}" | tee -a "$LOG"

.venv/bin/python -m ava.loop \
    --run-id "$RUN_ID" \
    --tasks flip \
    --rounds "$ROUNDS" \
    -k "$K" \
    --seed "$SEED" \
    --proposer evolve \
    --clusters "$CLUSTERS" \
    --eta "$ETA" \
    --race-fraction "$RACE_FRACTION" \
    --max-seeds "$MAX_SEEDS" \
    --harvest-top 0 \
    --harvest-seeds 0 \
    "$@" 2>&1 | tee -a "$LOG"

.venv/bin/python -m scripts.search_summary --run "runs/${RUN_ID}" 2>&1 | tee -a "$LOG"

echo
echo "contact sheet : runs/${RUN_ID}/contact_sheet.png"
echo "archive       : runs/${RUN_ID}/archive.jsonl   (every pair, every seed; elite flag)"
echo "search state  : runs/${RUN_ID}/search_state.json"
echo "summary       : runs/${RUN_ID}/search_summary.md"
echo "components    : runs/${RUN_ID}/components.md   (see task = \`flip\` sections)"
echo "log           : ${LOG}"
