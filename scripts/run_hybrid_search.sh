#!/usr/bin/env bash
# Search hybrid images with the current (Phase A) bandit proposer.
#
# Runs ava.loop on the `hybrid` task only: Thompson sampling over the hybrid
# low / high / style arms, 25% new-word injection, 25% targeted swaps, then a
# harvest of the top candidates across several seeds. Everything a report
# needs lands under runs/<RUN_ID>/ and the run is resumable: invoking the same
# RUN_ID again continues from state.json and adds ROUNDS more rounds.
#
# Budget at the defaults: 6 rounds x 8 candidates = 48 screening generations
# (~30 s each on the RTX 4090) plus 8 x 4 = 32 harvest generations, roughly
# 40 minutes in total.
#
#   scripts/run_hybrid_search.sh                    # defaults
#   ROUNDS=10 K=8 scripts/run_hybrid_search.sh      # longer screening
#   scripts/run_hybrid_search.sh --uniform          # validation sweep, not a search
#   RUN_ID=hybrid_a scripts/run_hybrid_search.sh      # a separate run directory
#
# Extra arguments are passed straight to ava.loop (see `-m ava.loop --help`).
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ID="${RUN_ID:-hybrid_$(date +%Y%m%d)}"
ROUNDS="${ROUNDS:-6}"
K="${K:-8}"
HARVEST_TOP="${HARVEST_TOP:-8}"
HARVEST_SEEDS="${HARVEST_SEEDS:-4}"
SEED="${SEED:-0}"

mkdir -p runs
LOG="runs/${RUN_ID}.log"
echo "[run_hybrid_search] run_id=${RUN_ID} rounds=${ROUNDS} k=${K}" \
     "harvest_top=${HARVEST_TOP} harvest_seeds=${HARVEST_SEEDS} seed=${SEED}" | tee -a "$LOG"

.venv/bin/python -m ava.loop \
    --run-id "$RUN_ID" \
    --tasks hybrid \
    --rounds "$ROUNDS" \
    -k "$K" \
    --seed "$SEED" \
    --harvest-top "$HARVEST_TOP" \
    --harvest-seeds "$HARVEST_SEEDS" \
    "$@" 2>&1 | tee -a "$LOG"

echo
echo "contact sheet : runs/${RUN_ID}/contact_sheet.png"
echo "components    : runs/${RUN_ID}/components.md   (see task = \`hybrid\` sections)"
echo "per round     : runs/${RUN_ID}/round_*/report.md"
echo "log           : ${LOG}"
