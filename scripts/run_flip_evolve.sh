#!/usr/bin/env bash
# Search flip illusions with the v2 evolutionary proposer, then summarise and
# animate. A thin wrapper over run_evolve.sh that keeps the flip_evo_<date>
# run naming; every knob (ROUNDS, K, SEED, CLUSTERS, ETA, RACE_FRACTION,
# MAX_SEEDS, ANIMATE, DRY_RUN) and every extra argument pass through.
#
#   scripts/run_flip_evolve.sh                        # defaults
#   ROUNDS=12 K=8 scripts/run_flip_evolve.sh          # longer
#   RUN_ID=flip_evo_a scripts/run_flip_evolve.sh      # a separate run directory
set -euo pipefail
RUN_ID="${RUN_ID:-flip_evo_$(date +%Y%m%d)}" TASKS=flip \
    exec "$(dirname "$0")/run_evolve.sh" "$@"
