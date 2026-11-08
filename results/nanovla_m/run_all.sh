#!/bin/bash
# NanoVLA-M: train head A (l1) and B (bins), each followed by the same closed-loop evals as results/final_v1 (A multitask, C unseen objects)
V=~/VLA; R=$V/results/nanovla_m
log() { echo "$(date +%H:%M) $*" | tee -a $R/run.log; }
for HEAD in l1 bins; do
  CK=$V/ckpt/nanovla_m_$HEAD/vlam.pt; mkdir -p $(dirname $CK)
  ( cd $V/vla && $V/tools/gpu.sh 12000 ~/VLM/venv/bin/python -u nanovla_m.py --head $HEAD --steps 20000 --out $CK ) > $(dirname $CK)/train.log 2>&1
  log "train $HEAD: $(grep -v -i warn $(dirname $CK)/train.log | grep val_l1 | tail -1)"
  $V/tools/gpu.sh 3600 $V/tools/eval_m.sh $CK pick,pad,tray,container,push,topple,rotate,rel 48 $R/A_$HEAD.json 66${#HEAD}1
  log "A $HEAD: $(grep RESULT $R/A_$HEAD.eval.log | cut -c1-260)"
  EVAL_X="--objects glue_stick,spool,toy_brick" $V/tools/gpu.sh 3000 $V/tools/eval_m.sh $CK pick 24 $R/C_$HEAD.json 66${#HEAD}2
  log "C $HEAD: $(grep RESULT $R/C_$HEAD.eval.log | cut -c1-260)"
done
log "ALL DONE"
