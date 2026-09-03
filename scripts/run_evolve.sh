#!/usr/bin/env bash
# Search any task(s) with the v2 evolutionary proposer, then summarise and
# animate the result. The task-independent form of run_flip_evolve.sh.
#
# Three steps, each reading only what the previous one persisted:
#   1. ava.loop --proposer evolve      generation + scoring  -> runs/<RUN_ID>/
#   2. scripts.search_summary          what the search learned -> search_summary.md
#   3. scripts.animate_run             transition clips (anim_<slot>.mp4) beside
#                                      the stills, via the upstream animate.py
#
# The run is resumable: the same RUN_ID continues from state.json,
# archive.jsonl and search_state.json and adds ROUNDS more rounds. Clips that
# already exist are kept.
#
#   TASKS=hybrid scripts/run_evolve.sh                      # one task
#   TASKS=rotate_cw,jigsaw,negate scripts/run_evolve.sh     # round-robin over tasks
#   TASKS=four_view ROUNDS=12 K=8 scripts/run_evolve.sh     # longer
#   TASKS=hybrid ANIMATE=all scripts/run_evolve.sh          # clips for every candidate
#   TASKS=hybrid ANIMATE=none scripts/run_evolve.sh         # no clips
#   TASKS=inverse_hybrid scripts/run_evolve.sh --ref-image x.png --ref-prompt "..."
#   RUN_ID=my_run TASKS=flip scripts/run_evolve.sh          # a named run directory
#
# Task names are those of ava.image.tasks.TASKS (flip, rotate_cw, rotate_ccw,
# rotate_180, skew, jigsaw, inner_circle, negate, patch_permute, pixel_permute,
# square_hinge, three_view, four_view, hybrid, triple_hybrid, color_hybrid,
# motion_hybrid, inverse_hybrid). triple_hybrid has no clip upstream and is
# reported as unsupported by step 3; everything else animates.
#
# Budget at the defaults: 8 rounds x 8 candidates = 64 generations, roughly
# 30 s each on the RTX 4090 with BLIP captions, about 35 minutes; clips add
# about 2.5 s per held candidate. Extra arguments go straight to ava.loop
# (see `-m ava.loop --help`). DRY_RUN=1 prints the commands instead.
#
# Watch it while it runs:  .venv/bin/python -m scripts.webui --runs runs
set -euo pipefail
cd "$(dirname "$0")/.."

TASKS="${TASKS:?set TASKS, e.g. TASKS=hybrid or TASKS=rotate_cw,jigsaw}"
RUN_ID="${RUN_ID:-evo_${TASKS//,/+}_$(date +%Y%m%d)}"
ROUNDS="${ROUNDS:-8}"
K="${K:-8}"
SEED="${SEED:-0}"
CLUSTERS="${CLUSTERS:-8}"
ETA="${ETA:-0.95}"
RACE_FRACTION="${RACE_FRACTION:-0.3}"
MAX_SEEDS="${MAX_SEEDS:-4}"
ANIMATE="${ANIMATE:-held}"   # held | all | none

case "$ANIMATE" in
  held|all|none) ;;
  *) echo "ANIMATE must be held, all or none (got '$ANIMATE')" >&2; exit 2 ;;
esac

run() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "$@"; echo
  else
    "$@"
  fi
}

mkdir -p runs
LOG="runs/${RUN_ID}.log"
[[ "${DRY_RUN:-0}" == "1" ]] && LOG=/dev/null  # a dry run leaves no trace
echo "[run_evolve] run_id=${RUN_ID} tasks=${TASKS} rounds=${ROUNDS} k=${K}" \
     "seed=${SEED} clusters=${CLUSTERS} eta=${ETA} race_fraction=${RACE_FRACTION}" \
     "max_seeds=${MAX_SEEDS} animate=${ANIMATE}" | tee -a "$LOG"

run .venv/bin/python -m ava.loop \
    --run-id "$RUN_ID" \
    --tasks "$TASKS" \
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

run .venv/bin/python -m scripts.search_summary --run "runs/${RUN_ID}" 2>&1 | tee -a "$LOG"

case "$ANIMATE" in
  held) run .venv/bin/python -m scripts.animate_run --run "runs/${RUN_ID}" --held-only 2>&1 | tee -a "$LOG" ;;
  all)  run .venv/bin/python -m scripts.animate_run --run "runs/${RUN_ID}" 2>&1 | tee -a "$LOG" ;;
  none) ;;
esac

echo
echo "contact sheet : runs/${RUN_ID}/contact_sheet.png"
echo "archive       : runs/${RUN_ID}/archive.jsonl   (every pair, every seed; elite flag)"
echo "search state  : runs/${RUN_ID}/search_state.json"
echo "summary       : runs/${RUN_ID}/search_summary.md"
echo "components    : runs/${RUN_ID}/components.md"
echo "clips         : runs/${RUN_ID}/round_*/<uid>/anim_<slot>.mp4  (animate=${ANIMATE})"
echo "log           : ${LOG}"
echo "viewer        : .venv/bin/python -m scripts.webui --runs runs   (then open http://127.0.0.1:8765/#${RUN_ID})"
