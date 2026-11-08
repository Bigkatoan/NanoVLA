#!/bin/bash
# grounding module + loc-conditioned decision + stronger anti-copycat, fine-tuned from the stage-1 checkpoint
V=~/VLA; R=$V/results/gct2; CK=$V/ckpt/nanovla_g2/vlag.pt
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $R/log.txt; }
log "train gct_g2"
( cd $V/vla && $V/tools/gpu.sh 9000 ~/VLM/venv/bin/python -u nanovla_g.py --run gct_g2 --bench isaac_gct_v1 --init $V/ckpt/nanovla_g/vlag.pt --steps 12000 --eval_every 2000 --out $CK ) > $(dirname $CK)/train.log 2>&1
log "trained: $(grep val_rel $(dirname $CK)/train.log | tail -1)"
( cd $V/vla && ~/VLM/venv/bin/python $V/tools/tb_eval.py $R/pick.eval.log $V/tb/gct_g2 pick ) &
EVAL_X="--diag 1" $V/tools/gpu.sh 3000 $V/tools/eval_c.sh $CK pick 24 $R/pick.json 7501; log "pick: $(grep RESULT $R/pick.eval.log | cut -c1-200)"
( cd $V/vla && ~/VLM/venv/bin/python $V/tools/tb_eval.py $R/A.eval.log $V/tb/gct_g2 A_multitask ) &
EVAL_X="--diag 1" $V/tools/gpu.sh 3600 $V/tools/eval_c.sh $CK pick,pad,tray,container,push,topple,rotate,rel 48 $R/A.json 7502; log "A: $(grep RESULT $R/A.eval.log | cut -c1-200)"
log DONE
