#!/bin/bash
# continuation: eval M+SS, train the user's variant (velocity actions + timestamp noise), pick the better one, DAgger rounds
V=~/VLA; R=$V/research/results/fix; CKL1=$V/ckpt/nanovla_m_l1/vlam.pt
log() { echo "$(date +%H:%M) $*" | tee -a $R/pipeline.log; }
while pgrep -f "nanovla_m.py --head l1 --init $CKL1 --ss_max 0.5 --steps 8000" > /dev/null; do sleep 30; done
log "M+SS: $(grep val_rel $V/ckpt/nanovla_m_ss/train.log | tail -1)"
$V/research/tools/gpu.sh 3600 $V/research/tools/eval_m.sh $V/ckpt/nanovla_m_ss/vlam.pt pick,pad,tray,container,push,topple,rotate,rel 48 $R/A_ss.json 6711
log "A M+SS: $(grep RESULT $R/A_ss.eval.log | cut -c1-200)"
# user's idea: velocity actions + timestamp noise (+ scheduled sampling), fine-tuned from M-L1
log "train M+SS+vel+jitter (12k steps)"
mkdir -p $V/ckpt/nanovla_m_vel
( cd $V/research/vla && $V/research/tools/gpu.sh 7200 ~/VLM/venv/bin/python -u nanovla_m.py --head l1 --init $CKL1 --act_mode vel --jitter 2 --ss_max 0.5 --steps 12000 \
    --lr 2e-4 --enc_lr 1e-5 --eval_every 2000 --out $V/ckpt/nanovla_m_vel/vlam.pt ) > $V/ckpt/nanovla_m_vel/train.log 2>&1
log "M+SS+vel+jitter: $(grep val_rel $V/ckpt/nanovla_m_vel/train.log | tail -1)"
$V/research/tools/gpu.sh 3600 $V/research/tools/eval_m.sh $V/ckpt/nanovla_m_vel/vlam.pt pick,pad,tray,container,push,topple,rotate,rel 48 $R/A_vel.json 6712
log "A M+SS+vel+jitter: $(grep RESULT $R/A_vel.eval.log | cut -c1-200)"
sr() { python3 -c "import json; print(json.load(open('$1'))['success_rate'])" 2>/dev/null || echo 0; }
if python3 -c "import sys; sys.exit(0 if $(sr $R/A_vel.json) >= $(sr $R/A_ss.json) else 1)"; then
  CUR=$V/ckpt/nanovla_m_vel/vlam.pt; XA="--act_mode vel --jitter 2"; log "DAgger starts from M+SS+vel+jitter"
else
  CUR=$V/ckpt/nanovla_m_ss/vlam.pt; XA=""; log "DAgger starts from M+SS"
fi
BASE=isaac_all_v2
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
  log "train DAgger r$RD $XA"
  ( cd $V/research/vla && $V/research/tools/gpu.sh 7200 ~/VLM/venv/bin/python -u nanovla_m.py --head l1 --init $CUR --bench isaac_dagger_r$RD --ss_max 0.5 --steps 10000 \
      --lr 1e-4 --enc_lr 1e-5 --eval_every 2000 $XA --out $CK ) > $(dirname $CK)/train.log 2>&1
  log "DAgger r$RD: $(grep val_rel $(dirname $CK)/train.log | tail -1)"
  $V/research/tools/gpu.sh 3600 $V/research/tools/eval_m.sh $CK pick,pad,tray,container,push,topple,rotate,rel 48 $R/A_dagger$RD.json 672$RD
  log "A dagger r$RD: $(grep RESULT $R/A_dagger$RD.eval.log | cut -c1-200)"
  EVAL_X="--objects glue_stick,spool,toy_brick" $V/research/tools/gpu.sh 3000 $V/research/tools/eval_m.sh $CK pick 24 $R/C_dagger$RD.json 673$RD
  log "C dagger r$RD: $(grep RESULT $R/C_dagger$RD.eval.log | cut -c1-200)"
  CUR=$CK; BASE=isaac_dagger_r$RD
done
log "ALL DONE"
