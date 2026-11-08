#!/bin/bash
# GCT stage 1: merge (old v2 demos + DAgger place chunk + new GCT data with labels), train, gate G1 evals. All on TensorBoard (run gct_s1).
V=~/VLA; R=$V/results/gct; CK=$V/ckpt/nanovla_g/vlag.pt; TBR=$V/tb/gct_s1
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $R/pipeline.log; }
until grep -q "GEN GCT DONE" $V/data/isaac_gct/gen.log 2>/dev/null; do sleep 60; done
touch $V/data/isaac_gct/STOP_TB
SRC=$(ls -d $V/data/isaac_gct/*_c* | while read d; do [ -f $d/episodes.json ] && echo $d; done | tr '\n' ' ')
log "stage0 data: $(grep -c done $V/data/isaac_gct/gen.log) chunks, $(python3 -c "import json,glob; print(sum(json.load(open(f))['kept'] for f in glob.glob('$V/data/isaac_gct/*_c*/episodes.json')))") new episodes"
~/VLM/venv/bin/python $V/vla/merge_sim_bench.py $V/data/isaac_gct_v1 $V/data/isaac_all_v2 $V/data/dagger_c1/place $SRC > $R/merge.log 2>&1
log "merged: $(tail -1 $R/merge.log)"
mkdir -p $(dirname $CK)
log "stage1 train"
( cd $V/vla && $V/tools/gpu.sh 14000 ~/VLM/venv/bin/python -u nanovla_g.py --run gct_s1 --bench isaac_gct_v1 --steps 20000 --out $CK ) > $(dirname $CK)/train.log 2>&1
log "stage1 trained: $(grep val_rel $(dirname $CK)/train.log | tail -1)"
( cd $V/vla && ~/VLM/venv/bin/python $V/tools/tb_eval.py $R/G1_pick.eval.log $TBR G1_pick_diag ) &
EVAL_X="--diag 1" $V/tools/gpu.sh 3000 $V/tools/eval_c.sh $CK pick 24 $R/G1_pick.json 7101
log "G1 pick: $(grep RESULT $R/G1_pick.eval.log | cut -c1-200)"
( cd $V/vla && ~/VLM/venv/bin/python $V/tools/tb_eval.py $R/G1_A.eval.log $TBR A_multitask ) &
EVAL_X="--diag 1" $V/tools/gpu.sh 3600 $V/tools/eval_c.sh $CK pick,pad,tray,container,push,topple,rotate,rel 48 $R/G1_A.json 7102
log "G1 A: $(grep RESULT $R/G1_A.eval.log | cut -c1-200)"
( cd $V/vla && ~/VLM/venv/bin/python $V/tools/tb_eval.py $R/G1_C.eval.log $TBR C_unseen ) &
EVAL_X="--diag 1 --objects glue_stick,spool,toy_brick" $V/tools/gpu.sh 3000 $V/tools/eval_c.sh $CK pick 24 $R/G1_C.json 7103
log "G1 C: $(grep RESULT $R/G1_C.eval.log | cut -c1-200)"
log "STAGE1 DONE"
