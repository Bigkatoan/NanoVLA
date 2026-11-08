#!/bin/bash
# after nanovla_r training: same closed-loop evals as results/final_v1 (A multitask, C unseen objects)
V=~/VLA; R=$V/results/nanovla_r; CK=$V/ckpt/nanovla_r/vlar.pt
log() { echo "$(date +%H:%M) $*" | tee -a $R/evals.log; }
while pgrep -f "nanovla_r.py --steps 20000" > /dev/null; do sleep 60; done
log "train: $(grep -v -i warn $V/ckpt/nanovla_r/train.log | tail -2 | tr '\n' ' ')"
$V/tools/gpu.sh 3600 $V/tools/eval_r.sh $CK pick,pad,tray,container,push,topple,rotate,rel 48 $R/A_multitask.json 6501
log "A: $(grep RESULT $R/A_multitask.eval.log | cut -c1-250)"
EVAL_X="--objects glue_stick,spool,toy_brick" $V/tools/gpu.sh 3000 $V/tools/eval_r.sh $CK pick 24 $R/C_unseen.json 6502
log "C: $(grep RESULT $R/C_unseen.eval.log | cut -c1-250)"
log "EVALS DONE"
