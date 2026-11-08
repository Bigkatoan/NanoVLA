#!/bin/bash
# NanoVLA-C (closed loop): train, then closed-loop evals A (8 tasks, 48 eps) and C (unseen objects, 24 eps); all in TensorBoard
V=~/VLA; R=$V/results/closed; RUN=closed_v1; CK=$V/ckpt/nanovla_c/vlac.pt; TBR=$V/tb/$RUN
log() { echo "$(date +%H:%M) $*" | tee -a $R/pipeline.log; }
mkdir -p $(dirname $CK)
log "train $RUN"
( cd $V/vla && $V/tools/gpu.sh 12000 ~/VLM/venv/bin/python -u nanovla_c.py --run $RUN --steps 20000 --out $CK ) > $(dirname $CK)/train.log 2>&1
log "trained: $(grep val_rel $(dirname $CK)/train.log | tail -1)"
( cd $V/vla && ~/VLM/venv/bin/python $V/tools/tb_eval.py $R/A.eval.log $TBR A_multitask ) &
$V/tools/gpu.sh 3600 $V/tools/eval_c.sh $CK pick,pad,tray,container,push,topple,rotate,rel 48 $R/A.json 6801
log "A: $(grep RESULT $R/A.eval.log | cut -c1-220)"
( cd $V/vla && ~/VLM/venv/bin/python $V/tools/tb_eval.py $R/C.eval.log $TBR C_unseen ) &
EVAL_X="--objects glue_stick,spool,toy_brick" $V/tools/gpu.sh 3000 $V/tools/eval_c.sh $CK pick 24 $R/C.json 6802
log "C: $(grep RESULT $R/C.eval.log | cut -c1-220)"
wait; log "ALL DONE"
