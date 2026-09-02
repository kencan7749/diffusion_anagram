#!/usr/bin/env bash
# Search time-reversal anagrams (the audio flip) with the v2 evolutionary proposer.
#
# The audio counterpart of run_flip_evolve.sh: ava.audio.loop on the
# `time_reverse` task with --proposer evolve. The vocabulary is the authored
# envelope prompts in ava/audio/vocab.py (decays and swells), registered in
# the shared runs/vocab.db on first use. Each candidate is one Stable Audio
# Open generation (two branches in one batch, 100 steps, 10 s) scored by CLAP
# in both directions; both directions are written as WAV beside the scores.
#
# Budget at the defaults: 6 rounds x 6 candidates = 36 generations, of which
# about 30% are racing seeds of pairs that held.
#
#   scripts/run_audio_flip_evolve.sh                  # defaults
#   ROUNDS=10 K=6 scripts/run_audio_flip_evolve.sh    # longer
#   STEPS=50 scripts/run_audio_flip_evolve.sh         # cheaper generations
#
# Afterwards: runs/<RUN_ID>/audition.md lists what held, with the files to
# play, and scripts/search_summary.py shows what the archive learned.
#
# Extra arguments are passed straight to ava.audio.loop.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ID="${RUN_ID:-audio_evo_$(date +%Y%m%d)}"
ROUNDS="${ROUNDS:-6}"
K="${K:-6}"
SEED="${SEED:-0}"
STEPS="${STEPS:-100}"
CLUSTERS="${CLUSTERS:-4}"
ETA="${ETA:-0.95}"
RACE_FRACTION="${RACE_FRACTION:-0.3}"
MAX_SEEDS="${MAX_SEEDS:-4}"

mkdir -p runs
LOG="runs/${RUN_ID}.log"
echo "[run_audio_flip_evolve] run_id=${RUN_ID} rounds=${ROUNDS} k=${K} seed=${SEED}" \
     "steps=${STEPS} clusters=${CLUSTERS} eta=${ETA} race_fraction=${RACE_FRACTION}" \
     "max_seeds=${MAX_SEEDS}" | tee -a "$LOG"

.venv/bin/python -m ava.audio.loop \
    --run-id "$RUN_ID" \
    --rounds "$ROUNDS" \
    -k "$K" \
    --seed "$SEED" \
    --steps "$STEPS" \
    --proposer evolve \
    --clusters "$CLUSTERS" \
    --eta "$ETA" \
    --race-fraction "$RACE_FRACTION" \
    --max-seeds "$MAX_SEEDS" \
    "$@" 2>&1 | tee -a "$LOG"

.venv/bin/python -m scripts.search_summary --run "runs/${RUN_ID}" 2>&1 | tee -a "$LOG"

echo
echo "audition      : runs/${RUN_ID}/audition.md   (what held, with forward.wav / reverse.wav)"
echo "archive       : runs/${RUN_ID}/archive.jsonl"
echo "summary       : runs/${RUN_ID}/search_summary.md"
echo "components    : runs/${RUN_ID}/components.md   (task = \`time_reverse\`)"
echo "log           : ${LOG}"
