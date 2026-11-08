#!/bin/bash
# Fixing NanoVLA-M's closed-loop drift: (1) scheduled sampling, (2) DAgger (policy drives, expert recovers).
V=~/VLA; R=$V/research/results/fix; CKL1=$V/ckpt/nanovla_m_l1/vlam.pt
log() { echo "$(date +%H:%M) $*" | tee -a $R/pipeline.log; }
while pgrep -f "eval_m.sh $CKL1 pick 24" > /dev/null; do sleep 30; done           # let eval C of M-L1 finish
log "C M-L1: $(grep RESULT $V/research/results/nanovla_m/C_l1.eval.log | cut -c1-160)"
# 0. DAgger smoke test (2 envs, 2 episodes) so a bug shows up early
log "dagger smoke test"
NE=2 $V/research/tools/gpu.sh 900 $V/research/tools/dagger_gen.sh $CKL1 pick pick 2 $V/data/dagger_smoke 6710 7
log "smoke: $(grep -c '"kept"' $V/data/dagger_smoke.log) episodes, $(grep -E 'Error|Traceback' $V/data/dagger_smoke.log | head -2)"
# 1. scheduled sampling fine-tune from M-L1
log "train M+SS (scheduled sampling, 8k steps)"
mkdir -p $V/ckpt/nanovla_m_ss
( cd $V/research/vla && $V/research/tools/gpu.sh 6000 ~/VLM/venv/bin/python -u nanovla_m.py --head l1 --init $CKL1 --ss_max 0.5 --steps 8000 --lr 1e-4 --enc_lr 1e-5 \
    --eval_every 2000 --out $V/ckpt/nanovla_m_ss/vlam.pt ) > $V/ckpt/nanovla_m_ss/train.log 2>&1
log "M+SS: $(grep val_rel $V/ckpt/nanovla_m_ss/train.log | tail -1)"
$V/research/tools/gpu.sh 3600 $V/research/tools/eval_m.sh $V/ckpt/nanovla_m_ss/vlam.pt pick,pad,tray,container,push,topple,rotate,rel 48 $R/A_ss.json 6711
log "A M+SS: $(grep RESULT $R/A_ss.eval.log | cut -c1-200)"
# 2. DAgger round(s): data with the current policy, aggregate, fine-tune (with scheduled sampling)
CUR=$V/ckpt/nanovla_m_ss/vlam.pt; BASE=isaac_all_v2
for RD in 1 2; do
  D=$V/data/dagger_r$RD; mkdir -p $D; SRC=""
  for J in "place pad,tray,container,rel" "pick pick" "push push,topple" "rotate rotate"; do set -- $J
    log "DAgger r$RD gen $1"
    $V/research/tools/gpu.sh 5400 $V/research/tools/dagger_gen.sh $CUR $1 $2 ${EPJ:-60} $D/$1 67$RD$((${#1} % 10)) $((5000 + 100 * RD + ${#1}))
    log "  $1: $(python3 -c "import json; d=json.load(open('$D/$1/episodes.json')); print(d['kept'], 'kept of', d['attempts'])" 2>&1 | tail -1)"
    [ -f $D/$1/episodes.json ] && SRC="$SRC $D/$1"
  done
  ~/VLM/venv/bin/python $V/research/vla/merge_sim_bench.py $V/data/isaac_dagger_r$RD $V/data/$BASE $SRC > $D/merge.log 2>&1
  log "merged r$RD: $(tail -1 $D/merge.log)"
  CK=$V/ckpt/nanovla_m_dagger$RD/vlam.pt; mkdir -p $(dirname $CK)
  log "train M+SS+DAgger r$RD"
  ( cd $V/research/vla && $V/research/tools/gpu.sh 7200 ~/VLM/venv/bin/python -u nanovla_m.py --head l1 --init $CUR --bench isaac_dagger_r$RD --ss_max 0.5 --steps 10000 \
      --lr 1e-4 --enc_lr 1e-5 --eval_every 2000 --out $CK ) > $(dirname $CK)/train.log 2>&1
  log "M+SS+DAgger r$RD: $(grep val_rel $(dirname $CK)/train.log | tail -1)"
  $V/research/tools/gpu.sh 3600 $V/research/tools/eval_m.sh $CK pick,pad,tray,container,push,topple,rotate,rel 48 $R/A_dagger$RD.json 672$RD
  log "A dagger r$RD: $(grep RESULT $R/A_dagger$RD.eval.log | cut -c1-200)"
  EVAL_X="--objects glue_stick,spool,toy_brick" $V/research/tools/gpu.sh 3000 $V/research/tools/eval_m.sh $CK pick 24 $R/C_dagger$RD.json 673$RD
  log "C dagger r$RD: $(grep RESULT $R/C_dagger$RD.eval.log | cut -c1-200)"
  CUR=$CK; BASE=isaac_dagger_r$RD
done
log "ALL DONE"
