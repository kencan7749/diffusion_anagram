#!/usr/bin/env bash
# A/B: the v1 component bandit against the v2 evolutionary search, same budget.
#
# Both arms get the same tasks, the same number of screening evaluations
# (ROUNDS x K), the same proposal seed, the same diffusion seed, and a private
# copy of the same vocabulary database, so neither arm's credit assignment can
# feed the other's. Harvest is off for both: v2 re-seeds inside its budget
# (racing) and v1 would otherwise spend extra generations after the count.
#
# What the design document says decides adoption (design_search_v2.md Sec. 6):
# v2 is adopted only if, at equal budget, its yield (pairs that held per
# evaluation) is at least v1's and it fills more archive cells. The numbers
# come from scripts/ab_compare.py, which reads only the two runs' files.
#
# Budget at the defaults: 2 x 12 rounds x 8 = 192 generations, ~30 s each on
# the RTX 4090, about 1 h 45 min in total, plus model loading.
#
#   scripts/run_ab_search.sh                        # defaults
#   TASKS=flip ROUNDS=6 scripts/run_ab_search.sh    # smaller
#   AB_DIR=runs_ab_2 scripts/run_ab_search.sh       # another comparison
#
# Extra arguments are passed to both ava.loop invocations.
set -euo pipefail
cd "$(dirname "$0")/.."

AB_DIR="${AB_DIR:-runs_ab}"
TASKS="${TASKS:-flip,hybrid,jigsaw}"
ROUNDS="${ROUNDS:-12}"
K="${K:-8}"
SEED="${SEED:-0}"
SOURCE_DB="${SOURCE_DB:-runs/vocab.db}"
OUT="${OUT:-results/stepB}"

mkdir -p "$AB_DIR/v1" "$AB_DIR/v2" "$OUT"
if [[ -f "$SOURCE_DB" ]]; then
    # Each arm gets its own copy of the same starting knowledge.
    cp -n "$SOURCE_DB" "$AB_DIR/v1/vocab.db" || true
    cp -n "$SOURCE_DB" "$AB_DIR/v2/vocab.db" || true
else
    echo "[run_ab_search] no $SOURCE_DB; both arms start from the seed vocabulary"
fi
LOG="$AB_DIR/ab.log"
echo "[run_ab_search] tasks=$TASKS rounds=$ROUNDS k=$K seed=$SEED db=$SOURCE_DB" | tee -a "$LOG"

for arm in v1 v2; do
    if [[ "$arm" == "v1" ]]; then proposer=bandit; else proposer=evolve; fi
    echo "[run_ab_search] === $arm ($proposer) ===" | tee -a "$LOG"
    .venv/bin/python -m ava.loop \
        --runs-dir "$AB_DIR/$arm" \
        --run-id "$arm" \
        --tasks "$TASKS" \
        --rounds "$ROUNDS" \
        -k "$K" \
        --seed "$SEED" \
        --proposer "$proposer" \
        --harvest-top 0 \
        --harvest-seeds 0 \
        "$@" 2>&1 | tee -a "$LOG"
done

echo "[run_ab_search] === compare ===" | tee -a "$LOG"
.venv/bin/python -m scripts.ab_compare \
    --v1 "$AB_DIR/v1/v1" --v2 "$AB_DIR/v2/v2" --out "$OUT" 2>&1 | tee -a "$LOG"
echo
echo "results : $OUT/ab.json, $OUT/ab.md"
echo "log     : $LOG"
