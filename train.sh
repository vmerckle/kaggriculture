#!/bin/bash
# Usage: bash train.sh OUT [GPUS]
# GPUS (default 1) share the clone stages at 1,792 rows per step; PPO runs on one GPU.
# Train from OUT/corpus; settings are in README.md.
# Exit 75 means rerun to continue.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(realpath -m "$1")
PY=${PY:-python}
export PYTHONPATH=$HERE OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
START=$(date +%s)
TIME_LIMIT=${TIME_LIMIT:-0}
[ "$TIME_LIMIT" -gt 0 ] || TIME_LIMIT=10000000
left() { echo $(( TIME_LIMIT - ($(date +%s) - START) / 60 )); }
mkdir -p $OUT
[ -f $OUT/seed ] || echo $(( RANDOM * 32768 + RANDOM )) > $OUT/seed
SEED=$(cat $OUT/seed)
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $OUT/train.txt; }
pause() { log "time limit: $1 stopped; run the same command again to continue"; exit 75; }
NPROC=${2:-1}
PPO_ITERATIONS=${PPO_ITERATIONS:-15}
BC_ACCUM=${BC_ACCUM:-$(( (7 + NPROC - 1) / NPROC ))}
if [ -z "${BC_BATCH:-}" ]; then
  BC_BATCH=$(( 1792 / (NPROC * BC_ACCUM) ))
  [ $(( NPROC * BC_ACCUM * BC_BATCH )) -eq 1792 ] || { echo "$NPROC GPUs cannot split 1,792 rows per step evenly; use 1, 2, 4, 7 or 8"; exit 1; }
fi
run() { $PY -B -u -m torch.distributed.run --standalone --nproc_per_node "$@"; }

clone() {
  local DIR=$1 rows=$2 steps=$3 lr=$4 warmup=$5 drop=$6 every=$7 init=${8:-} resume=""
  [ -f $DIR/ckpt-$(printf %06d $steps).pt ] && return 0
  [ $(left) -ge 5 ] || pause $(basename $DIR)
  [ -f $OUT/corpus/train/meta.json ] && [ -f $OUT/corpus/val/meta.json ] || { echo "$OUT/corpus has no packed train/ and val/; run corpus.py first"; exit 1; }
  [ -f $DIR/latest.pt ] && resume="--resume $DIR/latest.pt"
  mkdir -p $DIR
  run $NPROC $HERE/bc.py --data $OUT/corpus/train --rows $rows --val $OUT/corpus/val --out $DIR ${init:+--init $init} $resume \
    --steps $steps --minibatch $BC_BATCH --accum $BC_ACCUM --lr $lr --warmup $warmup --weight-decay 0.1 --dropout 0.1 \
    --team-drop $drop --eval-every $every --val-stride 100 --ckpt-every $every --keep-checkpoints 3 \
    --seed $SEED --time-limit $(( $(left) - 3 )) >> $DIR/train.log 2>&1
  log "$(basename $DIR): $(tail -1 $DIR/metrics.jsonl | cut -c1-160)"
  [ -f $DIR/ckpt-$(printf %06d $steps).pt ] || pause $(basename $DIR)
}

ppo() {
  local DIR=$OUT/ppo last=$OUT/ppo/ckpt-$(printf %06d $PPO_ITERATIONS).pt stop=$OUT/ppo/STOP resume="" timer FINETUNED_STEP400=$OUT/players/finetuned_step400.pt FINETUNED_STEP600=$OUT/players/finetuned_step600.pt
  [ -f $last ] && [ -f $DIR/player.pt ] && return 0
  [ $(left) -ge 25 ] || pause ppo
  mkdir -p $DIR
  rm -f $stop
  [ -f $DIR/latest.pt ] && { resume="--resume $DIR/latest.pt"; mv $DIR/train.log $DIR/train_$(date +%s).log; }
  # Allow 20min for the final iteration.
  (sleep $(( ($(left) - 20) * 60 )); touch $stop) & timer=$!
  $PY -B -u $HERE/ppo.py --init $FINETUNED_STEP600 --out $DIR $resume --opponents frozen+A1:$FINETUNED_STEP400,frozen+A1:$FINETUNED_STEP400,frozen:$FINETUNED_STEP400,frozen+A1:$FINETUNED_STEP600,frozen:$FINETUNED_STEP600,frozen+A1:$FINETUNED_STEP600,frozen:$FINETUNED_STEP600 \
    --iterations $PPO_ITERATIONS --games ${PPO_GAMES:-688} --accum ${PPO_ACCUM:-44} --minibatch 512 --lr 2e-5 --warmup 20 --gae-lambda 1.0 --target-kl 0.05 \
    --temperature 0.5 --replay-frac 0.25 --workers ${PPO_WORKERS:-12} --worker-sets 2 \
    --ckpt-every 1 --keep-checkpoints 3 --seed $SEED --stop-file $stop >> $DIR/train.log 2>&1
  kill $(pgrep -P $timer) $timer 2>/dev/null || true
  log "ppo: $(tail -1 $DIR/metrics.jsonl | cut -c1-200)"
  [ -f $last ] || pause ppo
}

log "train.sh start (seed $SEED, time limit $TIME_LIMIT min)"
clone $OUT/bc rows.npy 114000 1e-3 300 0.1 1000
clone $OUT/finetune finetune.npy 600 5e-5 50 0.0 200 $OUT/bc/latest.pt
mkdir -p $OUT/players
for k in 400 600; do [ -f $OUT/players/finetuned_step$k.pt ] || $PY -B $HERE/bc.py export $OUT/finetune/ckpt-$(printf %06d $k).pt $OUT/corpus/train/meta.json $OUT/players/finetuned_step$k.pt; done
[ "${UNTIL:-}" = players ] && { log "players exported; UNTIL=players stops here"; exit 75; }
ppo
[ -f $OUT/player.pt ] || cp $OUT/ppo/player.pt $OUT/player.pt
log "done: $OUT/player.pt"
