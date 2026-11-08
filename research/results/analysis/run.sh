#!/bin/bash
# hypothesis tests on NanoVLA-C, pick task, 24 episodes each, same seed: baseline / expert warm-up (start state) / JPEG parity
V=~/VLA; R=$V/research/results/analysis; CK=$V/ckpt/nanovla_c/vlac.pt
log() { echo "$(date +%H:%M) $*" | tee -a $R/log.txt; }
EVAL_X="--diag 1" $V/research/tools/gpu.sh 3000 $V/research/tools/eval_c.sh $CK pick 24 $R/E1_base.json 6951; log "E1 base: $(grep RESULT $R/E1_base.eval.log | cut -c1-150)"
EVAL_X="--diag 1 --warmup_spec specs/pick.json --warmup_steps 30" $V/research/tools/gpu.sh 3000 $V/research/tools/eval_c.sh $CK pick 24 $R/E2_warmup.json 6952; log "E2 warmup: $(grep RESULT $R/E2_warmup.eval.log | cut -c1-150)"
EVAL_X="--diag 1 --jpeg 90" $V/research/tools/gpu.sh 3000 $V/research/tools/eval_c.sh $CK pick 24 $R/E3_jpeg.json 6953; log "E3 jpeg: $(grep RESULT $R/E3_jpeg.eval.log | cut -c1-150)"
log DONE
