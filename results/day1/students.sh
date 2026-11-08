#!/bin/bash
V=~/VLA; R=$V/results/day1; PY=$V/envs/lerobot/bin/python
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $R/pipeline.log; }
for t in gt teacher both; do
  ( cd $V/vla && $V/tools/gpu.sh 6000 $PY student.py --train $R/labels_train.npz --test $R/labels_test.npz --target $t --out $V/ckpt/student/ttt_$t.pt --tb $V/tb/student_$t ) > $R/student_$t.log 2>&1
  log "student $t: $(tail -1 $R/student_$t.log)"
done
log "STUDENTS DONE"
